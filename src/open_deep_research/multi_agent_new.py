# multi_agent_new.py 
# LangGraph workflow + optional MCP tools  (v3)

from __future__ import annotations

import json, textwrap, operator, asyncio, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Annotated, TypedDict

from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig, Runnable
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command, Send

from open_deep_research.configuration import Configuration
from open_deep_research.utils import (
    get_config_value,
    tavily_search,
    duckduckgo_search,
)
from open_deep_research.prompts import SUPERVISOR_INSTRUCTIONS, RESEARCH_INSTRUCTIONS
from open_deep_research.mcp_integration import create_mcp_manager

logger = logging.getLogger(__name__)


# Helper – stringify any tool observation safely

def _to_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    try:
        return textwrap.dedent(json.dumps(x, indent=2))
    except TypeError:
        return str(x)


# MCP helpers

def _flatten(tools: List[BaseTool] | Any) -> List[BaseTool]:
    """Expand any StructuredTool collections returned by MCP."""
    flat: List[BaseTool] = []
    for t in tools:
        flat.extend(getattr(t, "tools", [t]))
    return flat


def _search_tool(cfg: Configuration):
    api = get_config_value(cfg.search_api).lower()
    return tavily_search if api == "tavily" else duckduckgo_search


# Shared tool‑lists

def _supervisor_tools(r: RunnableConfig):
    cfg = Configuration.from_runnable_config(r)
    base = [_search_tool(cfg), Sections, Introduction, Conclusion]
    
    # Get MCP tools from both possible locations with improved logging
    mcp_tools = []
    if "_mcp_tools" in r:
        mcp_tools = r["_mcp_tools"]
        logger.info(f"Found {len(mcp_tools)} MCP tools in _mcp_tools")
    elif "configurable" in r and "mcp_tools" in r["configurable"]:
        mcp_tools = r["configurable"]["mcp_tools"]
        logger.info(f"Found {len(mcp_tools)} MCP tools in configurable.mcp_tools")
    else:
        logger.warning("No MCP tools found in configuration")
    
    extra = _flatten(mcp_tools)
    tools = base + extra
    
    # Log full details of available tools
    tools_map = {t.name: t for t in tools}
    logger.info(f"Supervisor tools available: {list(tools_map.keys())}")
    
    return tools, tools_map


def _research_tools(r: RunnableConfig):
    cfg = Configuration.from_runnable_config(r)
    base = [_search_tool(cfg), Section]
    
    # Get MCP tools from both possible locations with improved logging
    mcp_tools = []
    if "_mcp_tools" in r:
        mcp_tools = r["_mcp_tools"]
        logger.info(f"Found {len(mcp_tools)} MCP tools in _mcp_tools")
    elif "configurable" in r and "mcp_tools" in r["configurable"]:
        mcp_tools = r["configurable"]["mcp_tools"]
        logger.info(f"Found {len(mcp_tools)} MCP tools in configurable.mcp_tools")
    else:
        logger.warning("No MCP tools found in configuration")
    
    extra = _flatten(mcp_tools)
    tools = base + extra
    
    # Log full details of available tools
    tools_map = {t.name: t for t in tools}
    logger.info(f"Research tools available: {list(tools_map.keys())}")
    
    return tools, tools_map


# Pydantic tool schemas

@tool
class Section(BaseModel):
    name: str
    description: str
    content: str


@tool
class Sections(BaseModel):
    sections: List[str]


@tool
class Introduction(BaseModel):
    name: str
    content: str


@tool
class Conclusion(BaseModel):
    name: str
    content: str


# States

class ReportState(MessagesState):
    sections: list[str]
    completed_sections: Annotated[list, operator.add]
    final_report: str


class ReportOutput(TypedDict):
    final_report: str


class SectionState(MessagesState):
    section: str
    completed_sections: list[Section]


class SectionOutput(TypedDict):
    completed_sections: list[Section]


# Nodes – supervisor / researcher

async def supervisor(state: ReportState, config: RunnableConfig):
    msgs = state["messages"]
    cfg = Configuration.from_runnable_config(config)
    llm = init_chat_model(model=get_config_value(cfg.supervisor_model))

    # Check for tool calls without responses in the message history
    tool_call_ids_with_responses = set()
    tool_call_ids_made = set()
    
    # Scan message history to find tool calls and their responses
    for msg in msgs:
        # Track tool calls made by assistant
        if hasattr(msg, "role") and msg.role == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict) and "id" in tool_call:
                    tool_call_ids_made.add(tool_call["id"])
        # Handle dictionary-style messages
        elif isinstance(msg, dict):
            if msg.get("role") == "ai" and "tool_calls" in msg:
                for tool_call in msg["tool_calls"]:
                    if "id" in tool_call:
                        tool_call_ids_made.add(tool_call["id"])
            elif msg.get("role") == "tool" and "tool_call_id" in msg:
                tool_call_ids_with_responses.add(msg["tool_call_id"])
    
    # Add empty responses for any missing tool calls
    new_msgs = list(msgs)  # Create a copy to avoid modifying during iteration
    for tool_call_id in tool_call_ids_made:
        if tool_call_id not in tool_call_ids_with_responses:
            logger.warning(f"Adding missing tool response for tool_call_id: {tool_call_id}")
            new_msgs.append({
                "role": "tool",
                "name": "unknown_tool",  # Use a generic name since we don't know which tool it was
                "content": "No response was recorded for this tool call.",
                "tool_call_id": tool_call_id
            })
    msgs = new_msgs

    if state.get("completed_sections") and not state.get("final_report"):
        msgs = msgs + [
            {
                "role": "user",
                "content": "Research complete – please draft intro & conclusion:\n\n"
                + "\n\n".join(s.content for s in state["completed_sections"]),
            }
        ]

    tools, _ = _supervisor_tools(config)
    
    try:
        ai_msg = await llm.bind_tools(tools).ainvoke(
            [{"role": "system", "content": SUPERVISOR_INSTRUCTIONS}] + msgs
        )
        return {"messages": [ai_msg]}
    except Exception as e:
        # If we get an error about missing tool responses, log it and return a generic message
        if "tool_call_id" in str(e):
            logger.error(f"Tool call error: {str(e)}")
            return {"messages": [{
                "role": "ai", 
                "content": "I encountered an issue with processing tool calls. Let's continue with the report creation."
            }]}
        else:
            # Re-raise other exceptions
            raise

async def supervisor_tools(state: ReportState, config: RunnableConfig) -> Command[Literal["supervisor", "research_team", "__end__"]]:
    """Performs the tool call and sends to the research agent with on-demand MCP tools."""
    
    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None
    mcp_manager = None
    
    try:
        # Initialize MCP manager on-demand for this task only
        mcp_servers = None
        if "configurable" in config and "_mcp_servers_config" in config["configurable"]:
            mcp_servers = config["configurable"]["_mcp_servers_config"]
            
        # Get base tools (without MCP)
        cfg = Configuration.from_runnable_config(config)
        base_tools = [_search_tool(cfg), Sections, Introduction, Conclusion]
        tools_by_name = {t.name: t for t in base_tools}
        
        # Add MCP tools if available
        if mcp_servers:
            # Create a new MCP manager just for this function call
            from open_deep_research.mcp_integration import create_mcp_manager
            mcp_manager = await create_mcp_manager(mcp_servers)
            logger.info(f"Created on-demand MCP manager for this request")
            
            if mcp_manager and mcp_manager.get_tools():
                mcp_tools = mcp_manager.get_tools()
                for tool in mcp_tools:
                    tools_by_name[tool.name] = tool
                logger.info(f"Added MCP tools: {[t for t in tools_by_name if t not in [x.name for x in base_tools]]}")
        
        # Process tool calls
        logger.info(f"Processing tool calls: {[call['name'] for call in state['messages'][-1].tool_calls]}")
        logger.info(f"Available tools: {list(tools_by_name.keys())}")
        
        # Process all tool calls to ensure we respond to each one (required for OpenAI)
        for tool_call in state["messages"][-1].tool_calls:
            # Get the tool name
            tool_name = tool_call["name"]
            
            # Add error handling for missing tools
            if tool_name not in tools_by_name:
                error_msg = f"Tool '{tool_name}' not found in available tools. Available tools: {list(tools_by_name.keys())}"
                logger.error(error_msg)
                
                # Return error message instead of crashing
                result.append({
                    "role": "tool", 
                    "content": f"Error: {error_msg}", 
                    "name": tool_name, 
                    "tool_call_id": tool_call["id"]
                })
                continue
            
            try:
                # Get the tool
                tool = tools_by_name[tool_name]
                
                # Log tool execution for debugging
                logger.info(f"Executing tool '{tool_name}' with args: {tool_call['args']}")
                
                # Perform the tool call - use ainvoke for async tools
                if hasattr(tool, 'ainvoke'):
                    observation = await tool.ainvoke(tool_call["args"])
                else:
                    observation = tool.invoke(tool_call["args"])

                # Append to messages 
                result.append({
                    "role": "tool", 
                    "content": _to_str(observation), 
                    "name": tool_name, 
                    "tool_call_id": tool_call["id"]
                })
                
                # Store special tool results for processing after all tools have been called
                if tool_name == "Sections":
                    sections_list = observation.sections
                    logger.info(f"Sections defined: {sections_list}")
                elif tool_name == "Introduction":
                    # Format introduction with proper H1 heading if not already formatted
                    intro_content = f"# {observation.name}\n\n{observation.content}" if not _to_str(observation).startswith("# ") else _to_str(observation)
                    logger.info("Introduction created")
                elif tool_name == "Conclusion":
                    # Format conclusion with proper H2 heading if not already formatted
                    conclusion_content = f"## {observation.name}\n\n{observation.content}" if not _to_str(observation).startswith("## ") else _to_str(observation)
                    logger.info("Conclusion created")
                    
            except Exception as e:
                # Add better error handling with full traceback
                error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                result.append({
                    "role": "tool", 
                    "content": f"Error: {error_msg}", 
                    "name": tool_name, 
                    "tool_call_id": tool_call["id"]
                })
    
    finally:
        # Always clean up MCP manager in the same task that created it
        if mcp_manager:
            await mcp_manager.cleanup()
            logger.info("Cleaned up on-demand MCP manager")
    
    # After processing all tool calls, decide what to do next
    if sections_list:
        # Send the sections to the research agents
        logger.info(f"Sending {len(sections_list)} sections to research team")
        return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], update={"messages": result})
    elif intro_content:
        # Store introduction while waiting for conclusion
        # Append to messages to guide the LLM to write conclusion next
        logger.info("Introduction complete, prompting for conclusion")
        result.append({"role": "user", "content": "Introduction written. Now write a conclusion section."})
        return Command(goto="supervisor", update={"final_report": intro_content, "messages": result})
    elif conclusion_content:
        # Get all sections and combine in proper order: Introduction, Body Sections, Conclusion
        logger.info("Conclusion complete, generating final report")
        intro = state.get("final_report", "")
        body_sections = "\n\n".join([s.content for s in state["completed_sections"]])
        
        # Assemble final report in correct order
        complete_report = f"{intro}\n\n{body_sections}\n\n{conclusion_content}"
        
        # Append to messages to indicate completion
        result.append({"role": "user", "content": "Report is now complete with introduction, body sections, and conclusion."})
        return Command(goto="supervisor", update={"final_report": complete_report, "messages": result})
    else:
        # Default case (for search tools, etc.)
        return Command(goto="supervisor", update={"messages": result})

async def supervisor_should_continue(state: ReportState) -> Literal["supervisor_tools", END]:
    """Decide if supervisor should continue based on tool calls"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Simple check for tool_calls attribute/property
    has_tool_calls = False
    if hasattr(last_message, "tool_calls"):
        has_tool_calls = bool(last_message.tool_calls)
    elif isinstance(last_message, dict) and "tool_calls" in last_message:
        has_tool_calls = bool(last_message["tool_calls"])
    
    return "supervisor_tools" if has_tool_calls else END


async def researcher(state: SectionState, config: RunnableConfig):
    """LLM decides whether to call a tool or not, with improved tool call tracking"""
    
    # Check for tool calls without responses in the message history
    msgs = state["messages"]
    tool_call_ids_with_responses = set()
    tool_call_ids_made = set()
    
    # Scan message history to find tool calls and their responses
    for msg in msgs:
        # Track tool calls made by assistant
        if hasattr(msg, "role") and msg.role == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict) and "id" in tool_call:
                    tool_call_ids_made.add(tool_call["id"])
        # Handle dictionary-style messages
        elif isinstance(msg, dict):
            if msg.get("role") == "ai" and "tool_calls" in msg:
                for tool_call in msg["tool_calls"]:
                    if "id" in tool_call:
                        tool_call_ids_made.add(tool_call["id"])
            elif msg.get("role") == "tool" and "tool_call_id" in msg:
                tool_call_ids_with_responses.add(msg["tool_call_id"])
    
    # Add empty responses for any missing tool calls
    new_msgs = list(msgs)  # Create a copy to avoid modifying during iteration
    for tool_call_id in tool_call_ids_made:
        if tool_call_id not in tool_call_ids_with_responses:
            logger.warning(f"Adding missing tool response for tool_call_id: {tool_call_id}")
            new_msgs.append({
                "role": "tool",
                "name": "unknown_tool",  # Use a generic name since we don't know which tool it was
                "content": "No response was recorded for this tool call.",
                "tool_call_id": tool_call_id
            })
    msgs = new_msgs
    
    # Add defensive check for section key
    if "section" not in state:
        logger.warning("Researcher called without section data, using default")
        section_description = "General research"
    else:
        # Handle different section formats
        section = state["section"]
        if isinstance(section, str):
            section_description = section
        elif hasattr(section, "description"):
            section_description = section.description
        else:
            # Try to convert to string or use default
            try:
                section_description = str(section)
            except:
                section_description = "Unknown section"
    
    # Log for debugging
    logger.info(f"Researcher processing section: {section_description}")
    
    # Get configuration
    cfg = Configuration.from_runnable_config(config)
    llm = init_chat_model(model=get_config_value(cfg.researcher_model))
    
    # Get tools
    tools, _ = _research_tools(config)
    
    # Create formatted instructions
    formatted_instructions = RESEARCH_INSTRUCTIONS.format(section_description=section_description)
    
    try:
        # Invoke the model
        ai_msg = await llm.bind_tools(tools).ainvoke(
            [{"role": "system", "content": formatted_instructions}] + msgs
        )
        return {"messages": [ai_msg]}
    except Exception as e:
        # If we get an error about missing tool responses, log it and return a generic message
        if "tool_call_id" in str(e):
            logger.error(f"Tool call error in researcher: {str(e)}")
            return {"messages": [{
                "role": "ai", 
                "content": "I encountered an issue with processing tool calls. Let me focus on completing this section with the information I have."
            }]}
        else:
            # Re-raise other exceptions
            raise


async def researcher_tools(state: SectionState, config: RunnableConfig):
    """Performs the tool call with on-demand MCP tools and routes to supervisor or continues the research loop."""

    result = []
    completed_section = None
    mcp_manager = None
    
    try:
        # Initialize MCP manager on-demand for this task only
        mcp_servers = None
        if "configurable" in config and "_mcp_servers_config" in config["configurable"]:
            mcp_servers = config["configurable"]["_mcp_servers_config"]
            
        # Get base tools (without MCP)
        cfg = Configuration.from_runnable_config(config)
        base_tools = [_search_tool(cfg), Section]
        tools_by_name = {t.name: t for t in base_tools}
        
        # Add MCP tools if available
        if mcp_servers:
            # Create a new MCP manager just for this function call
            from open_deep_research.mcp_integration import create_mcp_manager
            mcp_manager = await create_mcp_manager(mcp_servers)
            logger.info(f"Created on-demand MCP manager for researcher request")
            
            if mcp_manager and mcp_manager.get_tools():
                mcp_tools = mcp_manager.get_tools()
                for tool in mcp_tools:
                    tools_by_name[tool.name] = tool
                logger.info(f"Added MCP tools to researcher: {[t for t in tools_by_name if t not in [x.name for x in base_tools]]}")
    
        # Add debugging to show available tools and tool calls
        logger.info(f"Research: Processing tool calls: {[call['name'] for call in state['messages'][-1].tool_calls]}")
        logger.info(f"Research: Available tools: {list(tools_by_name.keys())}")
        
        # Process all tool calls first (required for OpenAI)
        for tool_call in state["messages"][-1].tool_calls:
            # Get the tool name
            tool_name = tool_call["name"]
            
            # Add error handling for missing tools
            if tool_name not in tools_by_name:
                error_msg = f"Tool '{tool_name}' not found in available tools. Available tools: {list(tools_by_name.keys())}"
                logger.error(error_msg)
                
                # Return error message instead of crashing
                result.append({
                    "role": "tool", 
                    "content": f"Error: {error_msg}", 
                    "name": tool_name, 
                    "tool_call_id": tool_call["id"]
                })
                continue
            
            try:
                # Get the tool
                tool = tools_by_name[tool_name]
                
                # Log tool execution for debugging
                logger.info(f"Executing tool '{tool_name}' with args: {tool_call['args']}")
                
                # Perform the tool call - use ainvoke for async tools
                if hasattr(tool, 'ainvoke'):
                    observation = await tool.ainvoke(tool_call["args"])
                else:
                    observation = tool.invoke(tool_call["args"])
                    
                # Append to messages 
                result.append({
                    "role": "tool", 
                    "content": _to_str(observation), 
                    "name": tool_name, 
                    "tool_call_id": tool_call["id"]
                })
                
                # Store the section observation if a Section tool was called
                if tool_name == "Section":
                    completed_section = observation
                    logger.info(f"Section completed: {observation.name}")
                    
            except Exception as e:
                # Add better error handling with full traceback
                error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                result.append({
                    "role": "tool", 
                    "content": f"Error: {error_msg}", 
                    "name": tool_name, 
                    "tool_call_id": tool_call["id"]
                })
    
    finally:
        # Always clean up MCP manager in the same task that created it
        if mcp_manager:
            await mcp_manager.cleanup()
            logger.info("Cleaned up on-demand MCP manager for researcher")
    
    # After processing all tools, decide what to do next
    if completed_section:
        # Write the completed section to state and return to the supervisor
        logger.info("Section completed, returning to supervisor")
        return {"messages": result, "completed_sections": [completed_section]}
    else:
        # Continue the research loop for search tools, etc.
        return {"messages": result}


async def researcher_should_continue(state: SectionState) -> Literal["researcher_tools", END]:
    """Decide if researcher should continue based on tool calls"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Simple check for tool_calls attribute/property
    has_tool_calls = False
    if hasattr(last_message, "tool_calls"):
        has_tool_calls = bool(last_message.tool_calls)
    elif isinstance(last_message, dict) and "tool_calls" in last_message:
        has_tool_calls = bool(last_message["tool_calls"])
    
    return "researcher_tools" if has_tool_calls else END


# Research graph construction 
research = StateGraph(SectionState, output=SectionOutput, config_schema=Configuration)
research.add_node("researcher", researcher)
research.add_node("researcher_tools", researcher_tools)
research.add_edge(START, "researcher")
research.add_conditional_edges(
    "researcher",
    researcher_should_continue,
    {
        "researcher_tools": "researcher_tools",
        END: END,
    }
)
research.add_edge("researcher_tools", "researcher")

# Supervisor graph construction - exactly like original
supervisor_g = StateGraph(ReportState, input=MessagesState, output=ReportOutput, config_schema=Configuration)
supervisor_g.add_node("supervisor", supervisor)
supervisor_g.add_node("supervisor_tools", supervisor_tools)
supervisor_g.add_node("research_team", research.compile())
supervisor_g.add_edge(START, "supervisor")
supervisor_g.add_conditional_edges(
    "supervisor",
    supervisor_should_continue,
    {
        "supervisor_tools": "supervisor_tools",
        END: END,
    }
)
supervisor_g.add_edge("supervisor_tools", "supervisor")
supervisor_g.add_edge("research_team", "supervisor")

# Factory helpers

async def build_graph_with_mcp(cfg: Optional[Dict[str, Any]] = None):
    """Create a graph with MCP configuration but without long-lived MCP connections."""
    cfg = cfg or {}
    
    # Get MCP configuration but DON'T create the manager yet
    configurable = Configuration.from_runnable_config(cfg)
    mcp_servers = configurable.mcp_servers
    
    # Log configuration info
    logger.info(f"Building graph with config keys: {list(cfg.keys() if cfg else [])}")
    if mcp_servers:
        logger.info(f"MCP servers configured: {list(mcp_servers.keys())}")
        
        # Store MCP servers config in a special key
        if "configurable" not in cfg:
            cfg["configurable"] = {}
        cfg["configurable"]["_mcp_servers_config"] = mcp_servers
    
    # Create graph with config only, no MCP manager
    g = supervisor_g.compile().with_config(cfg)
    return g, None, cfg


# Async factory expected by LangGraph Studio 
async def graph(config: dict | None = None):
    logger.info(f"Creating graph with config: {config}")
    g, _, _ = await build_graph_with_mcp(config)
    return g

# Sync alias for non‑async import flows 
def graph_sync(config: dict | None = None):
    return asyncio.run(graph(config))

# run `python multi_agent_new.py` to smoke‑test

if __name__ == "__main__":
    async def _demo():
        """Launch an example MCP server and ask a question about P001."""
        # Use default configuration
        # Build the graph
        graph, _, rc = await build_graph_with_mcp()
        
        try:
            logger.info("Invoking graph with patient query...")
            
            # Create initial state with our test query
            initial_state = {
                "messages": [{"role": "user", "content": "What do we know about patient with id P001?"}]
            }
            
            # First, run the supervisor directly to get AI response
            supervisor_state = await supervisor(initial_state, rc)
            logger.info("Supervisor completed initial processing")
            
            # Check if we have tool calls
            has_tool_calls = False
            if "messages" in supervisor_state and supervisor_state["messages"]:
                last_message = supervisor_state["messages"][-1]
                has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
            
            if not has_tool_calls:
                print("\n=== DEMO RESULT ===\n")
                print("No tool calls were made by the supervisor.")
                return supervisor_state
                
            # Update the state with supervisor results
            combined_state = {**initial_state, **supervisor_state}
            
            # Now run the tools function directly with our query
            logger.info("Running supervisor_tools with MCP calls...")
            tools_result = await supervisor_tools(combined_state, rc)
            
            # Print the MCP tool results
            print("\n=== PATIENT DATA FROM MCP TOOLS ===\n")
            
            # Extract tool responses from the Command update
            tool_responses = []
            if isinstance(tools_result, Command) and hasattr(tools_result, "update"):
                for msg in tools_result.update.get("messages", []):
                    if isinstance(msg, dict) and msg.get("role") == "tool":
                        tool_responses.append(msg)
            
            # Display each tool response with nice formatting
            for resp in tool_responses:
                tool_name = resp.get("name", "Unknown Tool")
                content = resp.get("content", "No data")
                
                print(f"\n## {tool_name}:")
                
                # Try to parse and pretty-print JSON content
                try:
                    if isinstance(content, str) and (content.startswith("{") or content.startswith("[")):
                        parsed = json.loads(content)
                        formatted = json.dumps(parsed, indent=2)
                        print(formatted)
                    else:
                        print(content)
                except:
                    # If not valid JSON, just print as is
                    print(content)
            
            # Now run one more step to get the AI's interpretation
            print("\n=== AI INTERPRETATION ===\n")
            
            # Update state with both supervisor and tool results
            if isinstance(tools_result, Command) and hasattr(tools_result, "update"):
                final_state = {
                    "messages": initial_state["messages"] + supervisor_state["messages"] + tools_result.update.get("messages", [])
                }
                
                # Run supervisor again to get interpretation
                final_response = await supervisor(final_state, rc)
                
                # Display the AI's interpretation
                if "messages" in final_response and final_response["messages"]:
                    ai_msg = final_response["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        print(ai_msg.content)
                    elif isinstance(ai_msg, dict) and "content" in ai_msg:
                        print(ai_msg["content"])
                    else:
                        print("No AI interpretation available")
                else:
                    print("No AI interpretation available")
            
            return {
                "supervisor": supervisor_state,
                "tools": tools_result,
                "interpretation": final_response if 'final_response' in locals() else None
            }
                
        except Exception as e:
            logger.error(f"Error in demo execution: {str(e)}", exc_info=True)
            return None

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demo
    asyncio.run(_demo())