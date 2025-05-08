from typing import List, Annotated, TypedDict, operator, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
import asyncio

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from langgraph.types import Command, Send
from langgraph.graph import START, END, StateGraph

from open_deep_research.configuration import Configuration
from open_deep_research.utils import get_config_value, tavily_search, duckduckgo_search
from open_deep_research.prompts import SUPERVISOR_INSTRUCTIONS, RESEARCH_INSTRUCTIONS
from open_deep_research.mcp_integration import create_mcp_manager

import logging
logger = logging.getLogger(__name__)

class MCPState:
    """Stores MCP state for the graph execution."""
    def __init__(self):
        self.manager = None

# Create state in graph context
_mcp_state = MCPState()

## Tools factory - will be initialized based on configuration
def get_search_tool(config: RunnableConfig):
    """Get the appropriate search tool based on configuration"""
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)

    # TODO: Configure other search functions as tools
    if search_api.lower() == "tavily":
        # Use Tavily search tool
        return tavily_search
    elif search_api.lower() == "duckduckgo":
        # Use the DuckDuckGo search tool
        return duckduckgo_search
    else:
        # Raise NotImplementedError for search APIs other than Tavily
        raise NotImplementedError(
            f"The search API '{search_api}' is not yet supported in the multi-agent implementation. "
            f"Currently, only Tavily is supported. Please use the graph-based implementation in "
            f"src/open_deep_research/graph.py for other search APIs, or set search_api to 'tavily'."
        )

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

# Initialize MCP tools at the start
async def initialize_mcp(state: ReportState, config: RunnableConfig):
    """Initialize MCP and continue to supervisor"""
    global _mcp_state
    
    # Get MCP configuration
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = getattr(configurable, "mcp_servers", None)
    
    if mcp_servers:
        logger.info(f"Initializing MCP manager with {len(mcp_servers)} servers")
        try:
            _mcp_state.manager = await create_mcp_manager(mcp_servers)
            logger.info(f"MCP manager initialized: {_mcp_state.manager is not None}")
        except Exception as e:
            logger.error(f"Error initializing MCP: {e}", exc_info=True)
    
    return state

async def cleanup_mcp(state: ReportState, config: RunnableConfig):
    """Clean up MCP resources before exiting"""
    global _mcp_state
    
    if _mcp_state.manager:
        logger.info("Cleaning up MCP resources")
        try:
            await _mcp_state.manager.cleanup()
            logger.info("MCP resources cleaned up successfully")
        except RuntimeError as e:
            if "cancel scope in a different task" in str(e):
                # This is actually expected in LangGraph's execution model
                logger.warning("Task context error during cleanup - this is expected in the StateGraph context")
            else:
                logger.error(f"Error during MCP cleanup: {e}", exc_info=True)
        
        # Set manager to None regardless of success
        _mcp_state.manager = None
    
    return {"final_report": state.get("final_report", "")}

# Tool lists will be built dynamically based on configuration
def get_supervisor_tools(config: RunnableConfig):
    """Get supervisor tools based on configuration"""
    global _mcp_state
    
    search_tool = get_search_tool(config)
    tool_list = [search_tool, Sections, Introduction, Conclusion]
    
    # Add MCP tools if available
    if _mcp_state.manager:
        mcp_tools = _mcp_state.manager.get_tools()
        if mcp_tools:
            logger.info(f"Adding {len(mcp_tools)} MCP tools to supervisor tools")
            tool_list.extend(mcp_tools)
    
    return tool_list, {tool.name: tool for tool in tool_list}

def get_research_tools(config: RunnableConfig):
    """Get research tools based on configuration"""
    global _mcp_state
    
    search_tool = get_search_tool(config)
    tool_list = [search_tool, Section]
    
    # Safely add MCP tools if available
    if _mcp_state.manager is not None:
        try:
            mcp_tools = _mcp_state.manager.get_tools()
            if mcp_tools:
                logger.info(f"Adding {len(mcp_tools)} MCP tools to research tools")
                tool_list.extend(mcp_tools)
        except Exception as e:
            logger.warning(f"Error getting MCP tools for researcher: {e}")
    
    return tool_list, {tool.name: tool for tool in tool_list}

async def supervisor(state: ReportState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""

    # Messages
    messages = state["messages"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    supervisor_model = get_config_value(configurable.supervisor_model)
    
    # Initialize the model
    llm = init_chat_model(model=supervisor_model)
    
    # If sections have been completed, but we don't yet have the final report, 
    # then we need to initiate writing the introduction and conclusion
    # CHANGE: Remove the section count check to match original flow
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

async def supervisor_tools(state: ReportState, config: RunnableConfig) -> Command[Literal["supervisor", "research_team", "cleanup_mcp"]]:
    """Performs the tool call and sends to the research agent"""

    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None

    # Get tools based on configuration
    _, supervisor_tools_by_name = get_supervisor_tools(config)
    
    # Simplify tool handling to match old implementation
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool
        tool = supervisor_tools_by_name[tool_call["name"]]
        
        # Perform the tool call - use ainvoke for async tools
        if hasattr(tool, 'ainvoke'):
            observation = await tool.ainvoke(tool_call["args"])
        else:
            observation = tool.invoke(tool_call["args"])

        # Append to messages - Keep format exactly like old implementation
        result.append({
            "role": "tool", 
            "content": observation,  # Don't convert to string/JSON 
            "name": tool_call["name"], 
            "tool_call_id": tool_call["id"]
        })
        
        # Store special tool results exactly as in old implementation
        if tool_call["name"] == "Sections":
            sections_list = observation.sections
        elif tool_call["name"] == "Introduction":
            if not observation.content.startswith("# "):
                intro_content = f"# {observation.name}\n\n{observation.content}"
            else:
                intro_content = observation.content
        elif tool_call["name"] == "Conclusion":
            if not observation.content.startswith("## "):
                conclusion_content = f"## {observation.name}\n\n{observation.content}"
            else:
                conclusion_content = observation.content
    
    # Match old implementation's decision flow exactly
    if sections_list:
        return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], 
                      update={"messages": result})
    elif intro_content:
        result.append({"role": "user", "content": "Introduction written. Now write a conclusion section."})
        return Command(goto="supervisor", update={"final_report": intro_content, "messages": result})
    elif conclusion_content:
        intro = state.get("final_report", "")
        body_sections = "\n\n".join([s.content for s in state["completed_sections"]])
        complete_report = f"{intro}\n\n{body_sections}\n\n{conclusion_content}"
        result.append({"role": "user", "content": "Report is now complete with introduction, body sections, and conclusion."})
        return Command(goto="supervisor", update={"final_report": complete_report, "messages": result})
    else:
        return Command(goto="supervisor", update={"messages": result})
    
async def supervisor_should_continue(state: ReportState) -> Literal["supervisor_tools", "cleanup_mcp"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "supervisor_tools"
    
    # Else end because the supervisor asked a question or is finished
    # Match old implementation by returning END directly
    else:
        return "cleanup_mcp"

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
    
    # Process all tool calls first (required for OpenAI)
    for tool_call in state["messages"][-1].tool_calls:
        try:
            # Get the tool
            tool_name = tool_call["name"]
            if tool_name not in research_tools_by_name:
                logger.warning(f"Tool '{tool_name}' not found in available tools")
                result.append({
                    "role": "tool", 
                    "content": f"Tool '{tool_name}' not found", 
                    "tool_call_id": tool_call["id"],
                    "name": tool_name
                })
                continue
                
            tool = research_tools_by_name[tool_name]
            
            # Perform the tool call - use ainvoke for async tools
            if hasattr(tool, 'ainvoke'):
                observation = await tool.ainvoke(tool_call["args"])
            else:
                observation = tool.invoke(tool_call["args"])
                
            # Special handling for MCP tools vs standard tools
            if tool_name == "Section":
                # For standard Section tool - pass observation directly
                tool_content = observation
                completed_section = observation
            else:
                # For MCP tools - convert to JSON string to ensure consistency
                import json
                if isinstance(observation, (dict, list)):
                    tool_content = json.dumps(observation)
                else:
                    # If already a string or other type, use as is
                    tool_content = str(observation)
            
            # Only append ONCE with properly formatted content
            result.append({
                "role": "tool", 
                "content": tool_content,
                "name": tool_name, 
                "tool_call_id": tool_call["id"]
            })
                
        except Exception as e:
            logger.error(f"Error executing research tool '{tool_call['name']}': {e}", exc_info=True)
            result.append({
                "role": "tool", 
                "content": f"Error executing tool: {str(e)}", 
                "name": tool_call["name"], 
                "tool_call_id": tool_call["id"]
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
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "research_agent_tools"

    else:
        return END
    
"""Build the multi-agent workflow"""

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
supervisor_builder.add_node("initialize_mcp", initialize_mcp)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("research_team", research_builder.compile())
supervisor_builder.add_node("cleanup_mcp", cleanup_mcp)

# Flow of the supervisor agent with MCP lifecycle
supervisor_builder.add_edge(START, "initialize_mcp")
supervisor_builder.add_edge("initialize_mcp", "supervisor")
supervisor_builder.add_conditional_edges(
    "supervisor",
    supervisor_should_continue,
    {
        "supervisor_tools": "supervisor_tools",
        "cleanup_mcp": "cleanup_mcp",
    },
)
supervisor_builder.add_edge("research_team", "supervisor")
supervisor_builder.add_edge("cleanup_mcp", END)

# Keep the original graph entry point for LangGraph Studio
graph = supervisor_builder.compile()

# Create an async factory function that matches the expected interface
async def create_graph(config=None):
    """Create a graph with the given configuration"""
    return graph.with_config(config) if config else graph