# src/open_deep_research/mcp_2.py
from typing import List, Annotated, TypedDict, operator, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
import asyncio
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from langgraph.types import Command, Send
from langgraph.graph import START, END, StateGraph

from open_deep_research.configuration import Configuration
from open_deep_research.utils import get_config_value, tavily_search, duckduckgo_search
from open_deep_research.prompts import SUPERVISOR_INSTRUCTIONS, RESEARCH_INSTRUCTIONS
from langchain_mcp_adapters.client import MultiServerMCPClient

import logging
logger = logging.getLogger(__name__)

# Using context variables to store MCP tools
from contextvars import ContextVar
_mcp_tools = ContextVar('mcp_tools', default=None)

@tool
class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Research scope for this section of the report.",
    )
    content: str = Field(
        description="The content of the section."
    )

@tool
class Sections(BaseModel):
    sections: List[str] = Field(
        description="Sections of the report.",
    )

@tool
class Introduction(BaseModel):
    name: str = Field(
        description="Name for the report.",
    )
    content: str = Field(
        description="The content of the introduction, giving an overview of the report."
    )

@tool
class Conclusion(BaseModel):
    name: str = Field(
        description="Name for the conclusion of the report.",
    )
    content: str = Field(
        description="The content of the conclusion, summarizing the report."
    )

## State
class ReportStateOutput(TypedDict):
    final_report: str # Final report

class ReportState(MessagesState):
    sections: list[str] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    final_report: str # Final report

class SectionState(MessagesState):
    section: str # Report section  
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

## Tools factory - will be initialized based on configuration
def get_search_tool(config: RunnableConfig):
    """Get the appropriate search tool based on configuration"""
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)

    if search_api.lower() == "tavily":
        return tavily_search
    elif search_api.lower() == "duckduckgo":
        return duckduckgo_search
    else:
        raise NotImplementedError(
            f"The search API '{search_api}' is not yet supported in the multi-agent implementation. "
            f"Currently, only Tavily is supported. Please use the graph-based implementation in "
            f"src/open_deep_research/graph.py for other search APIs, or set search_api to 'tavily'."
        )

def get_supervisor_tools(config: RunnableConfig):
    """Get supervisor tools based on configuration"""
    logger.info("get_supervisor_tools called")
    
    # Extract Configuration object
    configurable = Configuration.from_runnable_config(config)
    
    # Get search tool
    search_tool = get_search_tool(config)
    
    # Standard tools
    tool_list = [search_tool, Sections, Introduction, Conclusion]
    
    # MODIFIED: Get MCP tools from context var
    try:
        mcp_tools = _mcp_tools.get()
        if mcp_tools:
            logger.info(f"Retrieved {len(mcp_tools)} tools from context")
            tool_list.extend(mcp_tools)
    except LookupError:
        logger.warning("No MCP tools found in context")
    
    # Create and return the tool dictionary
    tool_dict = {tool.name: tool for tool in tool_list if hasattr(tool, 'name')}
    logger.info(f"Returning {len(tool_dict)} named tools")
    
    return tool_list, tool_dict

def get_research_tools(config: RunnableConfig):
    """Get research tools based on configuration"""
    logger.info("get_research_tools called")
    
    # Extract Configuration object
    configurable = Configuration.from_runnable_config(config)
    
    # Get search tool
    search_tool = get_search_tool(config)
    
    # Standard tools
    tool_list = [search_tool, Section]
    
    # Get MCP tools from context var
    try:
        mcp_tools = _mcp_tools.get()
        if mcp_tools:
            logger.info(f"Retrieved {len(mcp_tools)} tools from context")
            tool_list.extend(mcp_tools)
    except LookupError:
        logger.warning("No MCP tools found in context")
    
    # Create and return the tool dictionary
    tool_dict = {tool.name: tool for tool in tool_list if hasattr(tool, 'name')}
    logger.info(f"Returning {len(tool_dict)} named tools")
    
    return tool_list, tool_dict

async def supervisor(state: ReportState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    # Messages
    messages = state["messages"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    supervisor_model = get_config_value(configurable.supervisor_model)
    
    # Initialize the model
    llm = init_chat_model(model=supervisor_model)
    
    # If sections have been completed, but we don't yet have the final report, then we need to initiate writing the introduction and conclusion
    if state.get("completed_sections") and not state.get("final_report"):
        research_complete_message = {"role": "user", "content": "Research is complete. Now write the introduction and conclusion for the report. Here are the completed main body sections: \n\n" + "\n\n".join([s.content for s in state["completed_sections"]])}
        messages = messages + [research_complete_message]

    # Get tools based on configuration
    supervisor_tool_list, _ = get_supervisor_tools(config)
    
    # Invoke
    return {
        "messages": [
            await llm.bind_tools(supervisor_tool_list, parallel_tool_calls=False).ainvoke(
                [
                    {"role": "system",
                     "content": SUPERVISOR_INSTRUCTIONS,
                    }
                ]
                + messages
            )
        ]
    }

async def supervisor_tools(state: ReportState, config: RunnableConfig) -> Command[Literal["supervisor", "research_team", END]]:
    """Performs the tool call and sends to the research agent"""
    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None

    # Get tools based on configuration
    _, supervisor_tools_by_name = get_supervisor_tools(config)
    
    # Get the last message
    last_message = state["messages"][-1]
    
    # Get tool_calls safely
    tool_calls = getattr(last_message, "tool_calls", [])
    if not tool_calls:
        # No tool calls, just continue
        return Command(goto="supervisor", update={"messages": result})
    
    # First process all tool calls to ensure we respond to each one (required for OpenAI)
    for tool_call in tool_calls:
        # Get the tool
        tool_name = tool_call.get("name")
        if tool_name not in supervisor_tools_by_name:
            logger.warning(f"Tool '{tool_name}' not found in available tools")
            result.append({
                "role": "tool", 
                "content": f"Tool '{tool_name}' not found", 
                "tool_call_id": tool_call.get("id", "unknown"),
                "name": tool_name
            })
            continue
            
        tool = supervisor_tools_by_name[tool_name]
        
        try:
            # Perform the tool call - use ainvoke for async tools
            if hasattr(tool, 'ainvoke'):
                observation = await tool.ainvoke(tool_call.get("args", {}))
            else:
                observation = tool.invoke(tool_call.get("args", {}))

            # Append to messages
            result.append({
                "role": "tool", 
                "content": observation,
                "name": tool_name, 
                "tool_call_id": tool_call.get("id", "unknown")
            })
            
            # Store special tool results for processing after all tools have been called
            if tool_name == "Sections":
                sections_list = observation.sections
            elif tool_name == "Introduction":
                # Format introduction with proper H1 heading if not already formatted
                if not observation.content.startswith("# "):
                    intro_content = f"# {observation.name}\n\n{observation.content}"
                else:
                    intro_content = observation.content
            elif tool_name == "Conclusion":
                # Format conclusion with proper H2 heading if not already formatted
                if not observation.content.startswith("## "):
                    conclusion_content = f"## {observation.name}\n\n{observation.content}"
                else:
                    conclusion_content = observation.content
        except Exception as e:
            logger.error(f"Error executing supervisor tool '{tool_name}': {e}", exc_info=True)
            result.append({
                "role": "tool", 
                "content": f"Error executing tool: {str(e)}", 
                "name": tool_name, 
                "tool_call_id": tool_call.get("id", "unknown")
            })
    
    # After processing all tool calls, decide what to do next
    if sections_list:
        # Send the sections to the research agents
        return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], 
                      update={"messages": result})
    elif intro_content:
        # Store introduction while waiting for conclusion
        # Append to messages to guide the LLM to write conclusion next
        result.append({"role": "user", "content": "Introduction written. Now write a conclusion section."})
        return Command(goto="supervisor", update={"final_report": intro_content, "messages": result})
    elif conclusion_content:
        # Get all sections and combine in proper order: Introduction, Body Sections, Conclusion
        intro = state.get("final_report", "")
        body_sections = "\n\n".join([s.content for s in state["completed_sections"]])

        # Assemble final report in correct order
        complete_report = f"{intro}\n\n{body_sections}\n\n{conclusion_content}"

        # Append to messages to indicate completion
        result.append({"role": "user", "content": "Report is now complete with introduction, body sections, and conclusion."})
        return Command(goto="supervisor", update={"final_report": complete_report, "messages": result})
    else:
        return Command(goto="supervisor", update={"messages": result})
    
async def supervisor_should_continue(state: ReportState) -> Literal["supervisor_tools", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]    
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "supervisor_tools"
    
    # Else end because the supervisor asked a question or is finished
    else:
        return END

async def research_agent(state: SectionState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    researcher_model = get_config_value(configurable.researcher_model)
    
    # Initialize the model
    llm = init_chat_model(model=researcher_model)

    # Get tools based on configuration
    research_tool_list, _ = get_research_tools(config)
    
    return {
        "messages": [
            # Enforce tool calling to either perform more search or call the Section tool to write the section
            await llm.bind_tools(research_tool_list).ainvoke(
                [
                    {"role": "system",
                     "content": RESEARCH_INSTRUCTIONS.format(section_description=state["section"])
                    }
                ]
                + state["messages"]
            )
        ]
    }

async def research_agent_tools(state: SectionState, config: RunnableConfig):
    """Performs the tool call and route to supervisor or continue the research loop"""
    result = []
    completed_section = None
    
    # Get tools based on configuration
    _, research_tools_by_name = get_research_tools(config)
    
    # Get the last message
    last_message = state["messages"][-1]
    
    # Get tool_calls safely
    tool_calls = getattr(last_message, "tool_calls", [])
    if not tool_calls:
        # No tool calls, just continue
        return {"messages": result}
    
    # Process all tool calls first (required for OpenAI)
    for tool_call in tool_calls:
        try:
            # Get the tool
            tool_name = tool_call.get("name")
            if tool_name not in research_tools_by_name:
                logger.warning(f"Tool '{tool_name}' not found in available tools")
                result.append({
                    "role": "tool", 
                    "content": f"Tool '{tool_name}' not found", 
                    "tool_call_id": tool_call.get("id", "unknown"),
                    "name": tool_name
                })
                continue
                
            tool = research_tools_by_name[tool_name]
            
            # Perform the tool call - use ainvoke for async tools
            if hasattr(tool, 'ainvoke'):
                observation = await tool.ainvoke(tool_call.get("args", {}))
            else:
                observation = tool.invoke(tool_call.get("args", {}))
                
            # Pass observation directly without transformation
            result.append({
                "role": "tool", 
                "content": observation,
                "name": tool_name, 
                "tool_call_id": tool_call.get("id", "unknown")
            })
            
            # Store the section observation if Section tool was called
            if tool_name == "Section":
                completed_section = observation
                
        except Exception as e:
            logger.error(f"Error executing research tool '{tool_call.get('name', 'unknown')}': {e}", exc_info=True)
            result.append({
                "role": "tool", 
                "content": f"Error executing tool: {str(e)}", 
                "name": tool_call.get("name", "unknown"), 
                "tool_call_id": tool_call.get("id", "unknown")
            })
    
    # After processing all tools, decide what to do next
    if completed_section:
        # Write the completed section to state and return to the supervisor
        return {"messages": result, "completed_sections": [completed_section]}
    else:
        # Continue the research loop for search tools, etc.
        return {"messages": result}

async def research_agent_should_continue(state: SectionState) -> Literal["research_agent_tools", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    if not messages:
        return END
        
    last_message = messages[-1]

    # Check if the message has tool_calls attribute and it's not empty
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "research_agent_tools"
    else:
        return END
    
# Research agent workflow
research_builder = StateGraph(SectionState, output=SectionOutputState, config_schema=Configuration)
research_builder.add_node("research_agent", research_agent)
research_builder.add_node("research_agent_tools", research_agent_tools)
research_builder.add_edge(START, "research_agent") 
research_builder.add_conditional_edges(
    "research_agent",
    research_agent_should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "research_agent_tools": "research_agent_tools",
        END: END,
    },
)
research_builder.add_edge("research_agent_tools", "research_agent")

# Supervisor workflow
supervisor_builder = StateGraph(ReportState, input=MessagesState, output=ReportStateOutput, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("research_team", research_builder.compile())
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_conditional_edges(
    "supervisor",
    supervisor_should_continue,
    {
        "supervisor_tools": "supervisor_tools",
        END: END,
    },
)
supervisor_builder.add_edge("research_team", "supervisor")

graph = supervisor_builder.compile()

# Context manager for working with MCP directly
from contextlib import asynccontextmanager
from langgraph.checkpoint.memory import MemorySaver

@asynccontextmanager
async def create_mcp_research_graph(config: RunnableConfig):
    """Create a research graph with MCP integration following the documentation pattern."""
    # Extract MCP servers from config
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = getattr(configurable, "mcp_servers", None)
    
    logger.info(f"Starting create_mcp_research_graph with servers: {mcp_servers}")

    checkpointer = MemorySaver()
    workflow = supervisor_builder.compile(name="research_team", checkpointer=checkpointer)
            
    if mcp_servers:
        async with MultiServerMCPClient(mcp_servers) as client:
            # Set mcp tools in context variable
            _mcp_tools.set(client.get_tools())
            logger.info(f"Set {len(client.get_tools())} MCP tools in context")
            # yield the compiled graph
            yield workflow
    else:
        # No MCP servers configured, use standard graph
        logger.warning("No MCP servers configured, using standard graph")
        yield workflow
