import os
import logging
from typing import List, Dict, Any, Annotated, TypedDict, operator, Literal, Union
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.types import Command, Send
from langgraph.graph import START, END, StateGraph

from open_deep_research.configuration import Configuration
from open_deep_research.utils import get_config_value, tavily_search, duckduckgo_search
from open_deep_research.prompts import SUPERVISOR_INSTRUCTIONS, RESEARCH_INSTRUCTIONS
from open_deep_research.logging_config import configure_logging
from open_deep_research.validators import validate_search_api_config

# Get module-level logger
logger = logging.getLogger(__name__)




# Configure logging if this module is run directly
if __name__ == "__main__":
    configure_logging()

## Tools factory - will be initialized based on configuration
def get_search_tool(config: RunnableConfig):
    """Get the appropriate search tool based on configuration
    
    Args:
        config: The RunnableConfig containing configuration parameters
        
    Returns:
        The appropriate search tool function based on the configuration
        
    Raises:
        NotImplementedError: If the specified search API is not supported
        ValueError: If the configuration is invalid
    """
    try:
        configurable = Configuration.from_runnable_config(config)
        search_api = get_config_value(configurable.search_api)
        
        logger.info(f"Initializing search tool with API: {search_api}")
        
        # Validate search API configuration if present
        if configurable.search_api_config:
            validated_config = validate_search_api_config(
                search_api, configurable.search_api_config
            )
            logger.debug(f"Validated search API config: {validated_config}")
        
        # Return the appropriate search tool based on the search API
        supported_apis = {
            "tavily": tavily_search,
            "duckduckgo": duckduckgo_search
        }
        
        if search_api.lower() in supported_apis:
            return supported_apis[search_api.lower()]
        else:
            # Raise NotImplementedError for search APIs other than supported ones
            error_msg = (
                f"The search API '{search_api}' is not yet supported in the multi-agent implementation. "
                f"Currently, only {', '.join(supported_apis.keys())} are supported. "
                f"Please use the graph-based implementation in src/open_deep_research/graph.py "
                f"for other search APIs, or set search_api to one of the supported options."
            )
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
    except Exception as e:
        logger.error(f"Error in get_search_tool: {e}")
        raise

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

# Tool lists will be built dynamically based on configuration
def get_supervisor_tools(config: RunnableConfig):
    """Get supervisor tools based on configuration"""
    try:
        search_tool = get_search_tool(config)
        tool_list = [search_tool, Sections, Introduction, Conclusion]
        logger.info(f"Initialized supervisor tools: {[tool.__name__ if hasattr(tool, '__name__') else tool.__class__.__name__ for tool in tool_list]}")
        return tool_list, {tool.name: tool for tool in tool_list}
    except Exception as e:
        logger.error(f"Error initializing supervisor tools: {e}")
        raise

def get_research_tools(config: RunnableConfig):
    """Get research tools based on configuration"""
    try:
        search_tool = get_search_tool(config)
        tool_list = [search_tool, Section]
        logger.info(f"Initialized research tools: {[tool.__name__ if hasattr(tool, '__name__') else tool.__class__.__name__ for tool in tool_list]}")
        return tool_list, {tool.name: tool for tool in tool_list}
    except Exception as e:
        logger.error(f"Error initializing research tools: {e}")
        raise

async def supervisor(state: ReportState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""

    # Messages
    messages = state["messages"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    supervisor_model = get_config_value(configurable.supervisor_model)
    
    # Check if OPENAI_API_KEY is set and not empty
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize the model
    try:
        # Try using OpenAI with explicit API key
        if openai_api_key and len(openai_api_key) > 20:
            logger.info("Using OpenAI model for supervisor agent")
            # Use OpenAI models when OpenAI API key is available
            llm = init_chat_model(
                model="gpt-4o", 
                model_provider="openai",
                model_kwargs={"openai_api_key": openai_api_key}
            )
        elif ':' in supervisor_model:
            provider, model_name = supervisor_model.split(':', 1)
            logger.info(f"Using {provider} model {model_name} for supervisor agent")
            llm = init_chat_model(model=model_name, model_provider=provider)
        else:
            # Default to Groq if no provider specified
            logger.info(f"Using Groq model {supervisor_model} for supervisor agent")
            llm = init_chat_model(model=supervisor_model, model_provider="groq")
    except Exception as e:
        # Fallback to Groq if there's an error with OpenAI
        logger.error(f"Error initializing model: {e}. Falling back to Groq.")
        try:
            llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")
            logger.info("Successfully initialized fallback Groq model")
        except Exception as fallback_error:
            logger.critical(f"Critical error: Failed to initialize fallback model: {fallback_error}")
            raise RuntimeError(f"Failed to initialize any LLM model: {fallback_error}") from e
    
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

async def supervisor_tools(state: ReportState, config: RunnableConfig)  -> Command[Literal["supervisor", "research_team", "__end__"]]:
    """Performs the tool call and sends to the research agent"""

    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None

    logger.info("Processing supervisor tool calls")
    
    # Get tools based on configuration
    try:
        _, supervisor_tools_by_name = get_supervisor_tools(config)
    except Exception as e:
        logger.error(f"Error getting supervisor tools: {e}")
        raise
    
    # First process all tool calls to ensure we respond to each one (required for OpenAI)
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool
        tool = supervisor_tools_by_name[tool_call["name"]]
        # Perform the tool call - use ainvoke for async tools
        if hasattr(tool, 'ainvoke'):
            observation = await tool.ainvoke(tool_call["args"])
        else:
            observation = tool.invoke(tool_call["args"])

        # Append to messages 
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_call["name"], 
                       "tool_call_id": tool_call["id"]})
        
        # Store special tool results for processing after all tools have been called
        if tool_call["name"] == "Sections":
            sections_list = observation.sections
        elif tool_call["name"] == "Introduction":
            # Format introduction with proper H1 heading if not already formatted
            if not observation.content.startswith("# "):
                intro_content = f"# {observation.name}\n\n{observation.content}"
            else:
                intro_content = observation.content
        elif tool_call["name"] == "Conclusion":
            # Format conclusion with proper H2 heading if not already formatted
            if not observation.content.startswith("## "):
                conclusion_content = f"## {observation.name}\n\n{observation.content}"
            else:
                conclusion_content = observation.content
    
    # After processing all tool calls, decide what to do next
    if sections_list:
        # Send the sections to the research agents
        return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], update={"messages": result})
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
        # Default case (for search tools, etc.)
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

    # Messages
    messages = state["messages"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    researcher_model = get_config_value(configurable.researcher_model)
    
    # Check if OPENAI_API_KEY is set and not empty
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize the model
    try:
        # Try using OpenAI with explicit API key
        if openai_api_key and len(openai_api_key) > 20:
            logger.info("Using OpenAI model for research agent")
            # Use OpenAI models when OpenAI API key is available
            llm = init_chat_model(
                model="gpt-4o", 
                model_provider="openai",
                model_kwargs={"openai_api_key": openai_api_key}
            )
        elif ':' in researcher_model:
            provider, model_name = researcher_model.split(':', 1)
            logger.info(f"Using {provider} model {model_name} for research agent")
            llm = init_chat_model(model=model_name, model_provider=provider)
        else:
            # Default to Groq if no provider specified
            logger.info(f"Using Groq model {researcher_model} for research agent")
            llm = init_chat_model(model=researcher_model, model_provider="groq")
    except Exception as e:
        # Fallback to Groq if there's an error with OpenAI
        logger.error(f"Error initializing model for research agent: {e}. Falling back to Groq.")
        try:
            llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")
            logger.info("Successfully initialized fallback Groq model for research agent")
        except Exception as fallback_error:
            logger.critical(f"Critical error: Failed to initialize fallback model for research agent: {fallback_error}")
            raise RuntimeError(f"Failed to initialize any LLM model for research agent: {fallback_error}") from e

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

async def research_agent_tools(state: SectionState, config: RunnableConfig) -> Dict[str, Any]:
    """Performs the tool call and route to supervisor or continue the research loop"""
    try:
        logger.info("Running research agent tools")
        
        # Get the last message
        last_message = state["messages"][-1]
        
        # If there's no tool call, just return the current state
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            logger.info("No tool calls found, continuing research loop")
            return {"messages": state["messages"]}
        
        # Get the tool call
        tool_call = last_message.tool_calls[0]
        
        # Get the tool name and arguments
        tool_name = tool_call.name
        tool_args = tool_call.args
        
        logger.info(f"Tool call: {tool_name} with args: {tool_args}")
        
        # Handle the tool call
        if tool_name == "search":
            # Get the search tool
            search_tool = get_search_tool(config)
            
            # Run the search
            search_results = await search_tool(tool_args["query"])
            
            # Format the search results
            search_results_str = "\n\n".join([f"Source: {result.get('url', 'Unknown')}\n{result.get('content', '')}" for result in search_results])
            
            # Add the search results to the messages
            new_messages = state["messages"] + [ToolMessage(content=search_results_str, tool_call_id=tool_call.id)]
            
            return {"messages": new_messages}
        elif tool_name == "submit_section":
            # Get the section content
            section_content = tool_args["content"]
            
            # Create a section object
            section = Section(
                name=state["section"],
                description="",  # We don't have a description in this context
                content=section_content
            )
            
            # Add the section to the completed sections
            completed_sections = state["completed_sections"] + [section]
            
            # Return the completed sections to end this agent's work
            return {"completed_sections": completed_sections, "messages": state["messages"]}
        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return {"messages": state["messages"]}
    except Exception as e:
        logger.error(f"Error in research agent tools: {e}")
        return {"messages": state["messages"]}

async def research_agent_should_continue(state: SectionState) -> Literal["research_agent_tools", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    # Check if we have completed sections (meaning we're done)
    if "completed_sections" in state and state["completed_sections"]:
        logger.info("Section completed, ending research loop")
        return END
        
    # Get the last message
    last_message = state["messages"][-1]
    
    # If there's a tool call, continue to the tools node
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("Tool call found, continuing to research_agent_tools")
        return "research_agent_tools"
    
    # Otherwise, end the loop
    logger.info("No tool call found, ending research loop")
    return END

"""Build the multi-agent workflow"""

# Research agent workflow
research_builder = StateGraph(SectionState, output=SectionOutputState, config_schema=Configuration)

async def run_research(query: str, config: dict = None):
    """
    Run the research workflow with the given query and configuration.
    
    Args:
        query: The research query to process
        config: Optional configuration dictionary
        
    Returns:
        The final research report
    """
    logger.info(f"Starting research for query: {query}")
    
    # Initialize configuration
    if config is None:
        config = {}
    
    # Create the workflow graph
    workflow = build_workflow()
    
    # Run the workflow
    result = await workflow.ainvoke(
        {
            "sections": [query],
            "completed_sections": [],
            "final_report": ""
        },
        config
    )
    
    logger.info("Research completed successfully")
    return result.get("final_report", "")


def build_workflow():
    """
    Build and return the multi-agent workflow graph
    """
    # Create a new research builder
    research_graph = StateGraph(SectionState, output=SectionOutputState, config_schema=Configuration)
    research_graph.add_node("research_agent", research_agent)
    research_graph.add_node("research_agent_tools", research_agent_tools)
    research_graph.add_edge(START, "research_agent")
    research_graph.add_edge("research_agent", "research_agent_tools")
    research_graph.add_conditional_edges(
        "research_agent_tools",
        research_agent_should_continue,
        {
            "research_agent": "research_agent",
            END: END
        }
    )
    research_workflow = research_graph.compile()
    
    # Create a new supervisor builder
    supervisor_graph = StateGraph(ReportState, output=ReportStateOutput, config_schema=Configuration)
    supervisor_graph.add_node("supervisor", supervisor)
    supervisor_graph.add_node("supervisor_tools", supervisor_tools)
    supervisor_graph.add_node("research_team", research_workflow)
    supervisor_graph.add_edge(START, "supervisor")
    supervisor_graph.add_edge("supervisor", "supervisor_tools")
    supervisor_graph.add_edge("supervisor_tools", "supervisor")
    supervisor_graph.add_edge("research_team", "supervisor")
    supervisor_graph.add_conditional_edges(
        "supervisor",
        supervisor_should_continue,
        {
            "supervisor_tools": "supervisor_tools",
            END: END
        }
    )
    
    # Return the compiled workflow
    return supervisor_graph.compile()


# Create the graph for LangGraph Studio
graph = build_workflow()