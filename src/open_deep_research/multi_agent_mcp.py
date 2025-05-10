# src/open_deep_research/multi_agent_mcp_v2.py
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

# This registry stores our mcp tools without being serialized
_TOOL_REGISTRY = {}

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
    
    # Get MCP tools from registry if available
    mcp_tools = []
    registry_key = None
    
    if hasattr(configurable, "mcp_tool_registry_key"):
        registry_key = configurable.mcp_tool_registry_key
    elif isinstance(config, dict) and "configurable" in config and "mcp_tool_registry_key" in config["configurable"]:
        registry_key = config["configurable"]["mcp_tool_registry_key"]
    
    if registry_key and registry_key in _TOOL_REGISTRY:
        mcp_tools = _TOOL_REGISTRY[registry_key]
        logger.info(f"Retrieved {len(mcp_tools)} tools from registry with key {registry_key}")
    else:
        logger.warning(f"No MCP tools found in registry with key {registry_key}")
    
    if mcp_tools:
        logger.info(f"Using {len(mcp_tools)} tools: {[getattr(t, 'name', str(t)) for t in mcp_tools]}")
        tool_list.extend(mcp_tools)
    
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
    
    # Get MCP tools from registry if available
    mcp_tools = []
    registry_key = None
    
    if hasattr(configurable, "mcp_tool_registry_key"):
        registry_key = configurable.mcp_tool_registry_key
    elif isinstance(config, dict) and "configurable" in config and "mcp_tool_registry_key" in config["configurable"]:
        registry_key = config["configurable"]["mcp_tool_registry_key"]
    
    if registry_key and registry_key in _TOOL_REGISTRY:
        mcp_tools = _TOOL_REGISTRY[registry_key]
        logger.info(f"Retrieved {len(mcp_tools)} tools from registry with key {registry_key}")
    else:
        logger.warning(f"No MCP tools found in registry with key {registry_key}")
    
    if mcp_tools:
        logger.info(f"Using {len(mcp_tools)} tools: {[getattr(t, 'name', str(t)) for t in mcp_tools]}")
        tool_list.extend(mcp_tools)
    
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
    
    # If sections have been completed, but we don't yet have the final report, 
    # then we need to initiate writing the introduction and conclusion
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
    
    # Process all tool calls
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

            # Append to messages - Keep format exactly like original implementation
            result.append({
                "role": "tool", 
                "content": observation,
                "name": tool_name, 
                "tool_call_id": tool_call.get("id", "unknown")
            })
            
            # Store special tool results exactly as in original implementation
            if tool_name == "Sections":
                sections_list = observation.sections
            elif tool_name == "Introduction":
                if not observation.content.startswith("# "):
                    intro_content = f"# {observation.name}\n\n{observation.content}"
                else:
                    intro_content = observation.content
            elif tool_name == "Conclusion":
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
    
    # Match original implementation's decision flow exactly
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
    
async def supervisor_should_continue(state: ReportState) -> Literal["supervisor_tools", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    if not messages:
        return END
        
    last_message = messages[-1]

    # Check if the message has tool_calls attribute and it's not empty
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
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

# Then modify the create_mcp_research_graph function
@asynccontextmanager
async def create_mcp_research_graph(config: RunnableConfig):
    """Create a research graph with MCP integration using direct tools approach."""
    # Extract MCP servers from config
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = getattr(configurable, "mcp_servers", None)
    
    logger.info(f"Starting create_mcp_research_graph with servers: {mcp_servers}")
    
    # Ensure we have a dictionary for modifying config
    config_dict = dict(config) if config else {}
    
    # Ensure thread_id
    if "thread_id" not in config_dict:
        config_dict["thread_id"] = str(uuid.uuid4())
    
    # Set up MCP client if servers are configured
    client = None
    if mcp_servers:
        try:
            # Create and start the client
            logger.info(f"Creating MultiServerMCPClient with servers: {mcp_servers}")
            client = MultiServerMCPClient(mcp_servers)
            logger.info("Initializing MCP client")
            await client.__aenter__()  # Call the async enter method explicitly
            logger.info("MCP client initialized successfully")
            
            # Get tools directly and use them without proxies
            mcp_tools = client.get_tools()
            logger.info(f"Retrieved {len(mcp_tools)} tools directly from MCP client: {[t.name for t in mcp_tools if hasattr(t, 'name')]}")
            
            # CHANGE: Store tools in global registry instead of config
            tool_registry_key = str(uuid.uuid4())
            _TOOL_REGISTRY[tool_registry_key] = mcp_tools
            
            # Only store the registry key in config
            if "configurable" not in config_dict:
                config_dict["configurable"] = {}
            config_dict["configurable"]["mcp_tool_registry_key"] = tool_registry_key
            
            # Configure graph with the key reference
            logger.info("Configuring graph with MCP tool registry key")
            configured_graph = graph.with_config(config_dict)
            
            # Yield the configured graph
            logger.info("Yielding configured graph with tool registry reference")
            yield configured_graph
            
            # Clean up registry after use
            if tool_registry_key in _TOOL_REGISTRY:
                del _TOOL_REGISTRY[tool_registry_key]
            
        except Exception as e:
            logger.error(f"Error in create_mcp_research_graph: {e}", exc_info=True)
            raise
        finally:
            # Clean up client
            logger.info("Context exiting, cleaning up client")
            if client:
                logger.info("Closing MCP client")
                await client.__aexit__(None, None, None)
                logger.info("MCP client closed")
    else:
        logger.warning("No MCP servers configured, using standard graph")
        yield graph.with_config(config_dict)

# Utility function for using the context manager
async def run_with_mcp(query, config: RunnableConfig):
    """
    Run a research query using the multi-agent system with direct MCP integration.
    
    Args:
        query: The research query
        config: RunnableConfig containing MCP configuration
        
    Returns:
        The research results
    """
    async with create_mcp_research_graph(config) as research_graph:
        # Create input message
        messages = [{"role": "user", "content": query}]
        
        # Run the graph
        response = await research_graph.ainvoke({"messages": messages})
        
        # Log the final report
        if "final_report" in response:
            logger.info(f"Generated report with {len(response['final_report'])} characters")
        
        return response