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

# Create a serializable proxy for MCP tools
# Global registry with initial logging
_MCP_CLIENT_REGISTRY = {}
logger.info(f"Initialized empty MCP registry: {_MCP_CLIENT_REGISTRY}")

class SerializableMCPToolProxy(BaseTool):
    """A serializable proxy for MCP tools that doesn't maintain references to async resources."""
    
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool")
    args_schema: Dict[str, Any] = Field(default_factory=dict, description="Schema for tool arguments")
    server_name: str = Field(..., description="Server identifier for this tool")
    
    def _run(self, **kwargs) -> Any:
        """Sync version just raises an error - MCP tools should be async."""
        raise NotImplementedError("MCP tools must be used asynchronously")
    
    async def _arun(self, **kwargs) -> Any:
        """Dynamically get the MCP client from our local registry and locate the tool."""
        logger.info(f"Tool '{self.name}' invoked with args: {kwargs}")
        logger.info(f"Looking for client with server_name '{self.server_name}' in registry")
        
        # Get client from registry
        client = _MCP_CLIENT_REGISTRY.get(self.server_name)
        if not client:
            logger.error(f"No MCP client found for server {self.server_name}")
            raise ValueError(f"No MCP client found for server {self.server_name}")
        
        # Get all tools from the client
        logger.info(f"Found client, getting tools")
        all_tools = client.get_tools()
        
        # Find the specific tool by name
        logger.info(f"Looking for tool '{self.name}' among {len(all_tools)} tools")
        for tool in all_tools:
            if tool.name == self.name:
                logger.info(f"Found tool '{self.name}', invoking with {kwargs}")
                # Use the tool's ainvoke method directly
                return await tool.ainvoke(kwargs)
        
        # Tool not found
        logger.error(f"Tool '{self.name}' not found in client tools")
        raise ValueError(f"Tool '{self.name}' not found in client tools")
    
def get_mcp_client_registry():
    """Get the global MCP client registry."""
    return _MCP_CLIENT_REGISTRY

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
    
    # Extract Configuration object properly
    configurable = Configuration.from_runnable_config(config)
    
    # Get search tool
    search_tool = get_search_tool(config)
    
    # Standard tools
    tool_list = [search_tool, Sections, Introduction, Conclusion]
    
    # Check for direct MCP client access via session_id
    session_id = None
    if hasattr(configurable, "mcp_session_id"):
        session_id = configurable.mcp_session_id
        logger.info(f"Found session_id in configurable: {session_id}")
    elif isinstance(config, dict):
        session_id = config.get("mcp_session_id")
        logger.info(f"Found session_id in config dict: {session_id}")
    else:
        logger.warning("No session_id found in config")
    
    # Log registry state
    logger.info(f"Current registry state: {list(_MCP_CLIENT_REGISTRY.keys())}")
    
    if session_id and session_id in _MCP_CLIENT_REGISTRY:
        # Get tools directly from client in registry
        logger.info(f"Found live client for session '{session_id}' in registry")
        client = _MCP_CLIENT_REGISTRY[session_id]
        mcp_tools = client.get_tools()
        logger.info(f"Retrieved {len(mcp_tools)} tools from live client: {[getattr(t, 'name', str(t)) for t in mcp_tools]}")
        tool_list.extend(mcp_tools)
    
    # Fall back to tools stored in configurable
    else:
        if session_id:
            logger.warning(f"Session '{session_id}' not found in registry, falling back to configurable")
        
        # Properly check for mcp_tools in the configurable
        mcp_tools = []
        if hasattr(configurable, "mcp_tools"):
            mcp_tools = configurable.mcp_tools
            logger.info(f"Found {len(mcp_tools)} tools in configurable.mcp_tools")
        elif isinstance(config, dict) and "configurable" in config and "mcp_tools" in config["configurable"]:
            mcp_tools = config["configurable"]["mcp_tools"]
            logger.info(f"Found {len(mcp_tools)} tools in config dict configurable.mcp_tools")
        
        if mcp_tools:
            logger.info(f"Using {len(mcp_tools)} tools from configurable: {[getattr(t, 'name', str(t)) for t in mcp_tools]}")
            tool_list.extend(mcp_tools)
        else:
            logger.warning("No MCP tools found in config")
    
    # Log the final combined tool list
    logger.info(f"Final tool list: {[getattr(t, 'name', str(t)) for t in tool_list]}")
    
    # Create and return the tool dictionary
    tool_dict = {tool.name: tool for tool in tool_list if hasattr(tool, 'name')}
    logger.info(f"Returning {len(tool_dict)} named tools")
    
    return tool_list, tool_dict

def get_research_tools(config: RunnableConfig):
    """Get research tools based on configuration"""
    # Extract Configuration object properly
    configurable = Configuration.from_runnable_config(config)
    
    # Get search tool
    search_tool = get_search_tool(config)
    
    # Standard tools
    tool_list = [search_tool, Section]
    
    # Check for direct MCP client access via session_id
    session_id = None
    if hasattr(configurable, "mcp_session_id"):
        session_id = configurable.mcp_session_id
    elif isinstance(config, dict):
        session_id = config.get("mcp_session_id")
    
    if session_id and session_id in _MCP_CLIENT_REGISTRY:  # USE THE SINGLE REGISTRY
        # Get tools directly from client in registry
        client = _MCP_CLIENT_REGISTRY[session_id]
        mcp_tools = client.get_tools()
        logger.info(f"Using live MCP client with session ID {session_id} for research tools")
        tool_list.extend(mcp_tools)
    
    # Fall back to tools stored in configurable
    else:
        # Properly check for mcp_tools in the configurable
        mcp_tools = []
        if hasattr(configurable, "mcp_tools"):
            mcp_tools = configurable.mcp_tools
        
        if mcp_tools:
            logger.info(f"MCP tools found in configurable for research: {[getattr(t, 'name', str(t)) for t in mcp_tools]}")
            tool_list.extend(mcp_tools)
        else:
            logger.warning("No MCP tools found in research config")
    
    # Return tools and their name-keyed dictionary
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

# Option 1 using the registry

# TODO: Implement direct tool approach (Option 2)
# -----------------------------------------------
# The current implementation (Option 1) uses a proxy approach with a registry
# to maintain client references across serialization boundaries. This works but
# isn't the most elegant solution.
#
# Will work on Option 2 which aligns closer with langchain-mcp-adapters demo 
# documentation this weekend. Option 2 would use the tools directly without a
# tool registry. I had some issues with the langgraph pickle/asyncio context
# manager when trying to implement this approach initially.
# 
# Current implementation considerations:
# 1. We use SerializableMCPToolProxy to avoid references to async resources
# 2. The global _MCP_CLIENT_REGISTRY maintains client instances by session ID
# 3. Each proxy looks up its client at execution time using server_name
# 4. The proxy's _arun method finds the correct tool by name from all client tools

@asynccontextmanager
async def create_mcp_research_graph(config: RunnableConfig):
    """Create a research graph with MCP integration using proxy tools."""
    # Extract MCP servers from config
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = getattr(configurable, "mcp_servers", None)
    
    logger.info(f"Starting create_mcp_research_graph_with_proxy with servers: {mcp_servers}")
    
    # Ensure we have a dictionary for modifying config
    config_dict = dict(config) if config else {}
    
    # Ensure thread_id
    if "thread_id" not in config_dict:
        config_dict["thread_id"] = str(uuid.uuid4())
    
    # Generate a unique session ID for this graph instance
    session_id = str(uuid.uuid4())
    logger.info(f"Generated session ID: {session_id}")
    config_dict["mcp_session_id"] = session_id
    
    # Set up MCP client if servers are configured
    if mcp_servers:
        try:
            # Create and start the client
            logger.info(f"Creating MultiServerMCPClient with servers: {mcp_servers}")
            client = MultiServerMCPClient(mcp_servers)
            logger.info("Initializing MCP client")
            await client.__aenter__()  # Call the async enter method explicitly
            logger.info("MCP client initialized successfully")
            
            # Store in global registry for later access by tools
            logger.info(f"Storing client in registry with key '{session_id}'")
            _MCP_CLIENT_REGISTRY[session_id] = client
            logger.info(f"Registry state after storing client: {list(_MCP_CLIENT_REGISTRY.keys())}")
            
            # Get tools from the client to create proxies
            actual_tools = client.get_tools()
            logger.info(f"Retrieved {len(actual_tools)} tools from MCP client: {[t.name for t in actual_tools if hasattr(t, 'name')]}")
            
            # Create proxy tools for serialization safety
            logger.info("Creating proxy tools")
            mcp_proxies = []
            for tool in actual_tools:
                try:
                    tool_name = getattr(tool, 'name', f"unknown-{id(tool)}")
                    proxy = SerializableMCPToolProxy(
                        name=tool_name,
                        description=getattr(tool, 'description', ''),
                        args_schema=getattr(tool, "args_schema", {}),
                        server_name=session_id
                    )
                    mcp_proxies.append(proxy)
                    logger.info(f"Created proxy for tool: {tool_name}")
                except Exception as e:
                    logger.error(f"Error creating proxy for tool {getattr(tool, 'name', 'unknown')}: {e}", exc_info=True)
            
            # Store proxies in config
            logger.info(f"Storing {len(mcp_proxies)} proxy tools in config")
            if "configurable" not in config_dict:
                config_dict["configurable"] = {}
            config_dict["configurable"]["mcp_tools"] = mcp_proxies
            
            # Configure graph with session ID and proxies
            logger.info("Configuring graph with MCP proxy tools")
            configured_graph = graph.with_config(config_dict)
            
            # Yield the configured graph
            logger.info("Yielding configured graph with proxy tools")
            yield configured_graph
            
        except Exception as e:
            logger.error(f"Error in create_mcp_research_graph_with_proxy: {e}", exc_info=True)
            raise
        finally:
            # Clean up client when context exits
            logger.info(f"Context exiting, cleaning up client for session '{session_id}'")
            client = _MCP_CLIENT_REGISTRY.pop(session_id, None)
            if client:
                logger.info("Closing MCP client")
                await client.__aexit__(None, None, None)
                logger.info("MCP client closed")
            else:
                logger.warning(f"No client found for session '{session_id}' during cleanup")
    else:
        logger.warning("No MCP servers configured, using standard graph")
        yield graph.with_config(config_dict)
