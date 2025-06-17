import uuid
import asyncio
# ────────────────────────────────────────────────────────────────
#  LOGGING SET-UP (runs once when the module is imported)
# ────────────────────────────────────────────────────────────────
import logging
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver

import hashlib
import requests
from datetime import datetime
import os

from typing import List
from pydantic import Field, HttpUrl, StrictStr, BaseModel

import re
from typing import Set
from pydantic.functional_validators import model_validator

timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
# create logs directory if it does not exist
if not os.path.exists("logs"):
    os.makedirs("logs")
log_filename = f"logs/{timestamp}.log"

logging.basicConfig(
    level=logging.DEBUG,                                    # global default
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()                           # keep terminal output
    ],
)

logger = logging.getLogger(__name__)
logger.info("Logging initialised → %s", log_filename)


from typing import List, Annotated, TypedDict, operator, Literal
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from langgraph.types import Command, Send
from langgraph.graph import START, END, StateGraph

from open_deep_research.configuration import Configuration
from open_deep_research.utils import get_config_value, azureaisearch_search
from modules.prompts import SUPERVISOR_INSTRUCTIONS, RESEARCH_INSTRUCTIONS

#logging.getLogger("langgraph").setLevel(logging.DEBUG)
#logging.getLogger("langchain").setLevel(logging.DEBUG)
#logging.getLogger("open_deep_research").setLevel(logging.DEBUG)

#logger.propagate = True


import hashlib
import requests
from datetime import datetime
import os

def save_mermaid_png(graph, xray: int = 1, max_retries: int = 5, output_dir: str = ".") -> str:
    """
    Erstellt und speichert ein Mermaid-PNG aus einem Graphenobjekt.
    Unterstützt Rückgabe als Bytes oder URL.

    Gibt den Pfad zur gespeicherten Datei zurück.
    """
    result = graph.get_graph(xray=xray).draw_mermaid_png(max_retries=max_retries)

    # Prüfe den Rückgabetyp
    if isinstance(result, bytes):
        png_bytes = result
    elif isinstance(result, str) and result.startswith("http"):
        # Als URL behandeln und versuchen herunterzuladen
        try:
            response = requests.get(result)
            response.raise_for_status()
            png_bytes = response.content
        except Exception as e:
            raise RuntimeError(f"Fehler beim Herunterladen der PNG von {result}: {e}")
    else:
        raise TypeError(f"Unerwarteter Rückgabewert von draw_mermaid_png: {type(result)}")

    # Datei benennen und speichern
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    hashcode = hashlib.sha1(png_bytes).hexdigest()[:4]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{timestamp}-{hashcode}.png"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        f.write(png_bytes)

    print(f"PNG gespeichert unter: {output_path}")
    return output_path


# Setze explizit Logging-Level für relevante Bibliotheken
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.DEBUG)



## Tools factory - will be initialized based on configuration
def get_search_tool(config: RunnableConfig):
    """Get the appropriate search tool based on configuration"""
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    logger.debug("Selecting search API: %s", search_api)

    return azureaisearch_search


# @tool
# class Section(BaseModel):
#     name: str = Field(
#         description="Name for this section of the report.",
#     )
#     description: str = Field(
#         description="Research scope for this section of the report.",
#     )
#     content: str = Field(
#         description="The content of the section."
#     )


from datetime import date
from pydantic import BaseModel, Field

from typing import Dict

#@tool
class Reference(BaseModel):
    anchor: StrictStr = Field(
        ...,
        pattern=r"[A-Za-z][A-Za-z0-9_-]*",
        description="unique indetifier (i.e. Doe2024)."
    )
    author: StrictStr
    title: StrictStr
    quote: StrictStr
    url: HttpUrl
    published: date
    tags: Dict[StrictStr, StrictStr] = Field(
        default_factory=dict,
        description="additional key-value tags"
    )

@tool
class Section(BaseModel):
    name: str = Field(..., description="Name for this section of the report.")
    description: str = Field(..., description="Brief overview of the main topics and concepts to be covered in this section.")
    research: bool = Field(..., description="Whether to perform web research for this section of the report.")
    content: str = Field(..., description="The content of the section with reference anchors")
    references: List[Reference] = Field(
        default_factory=list,
        description="List of anchored references"
    )

    @model_validator(mode='after')
    def check_reference_anchors(self):
        content = self.content
        anchors: Set[str] = {ref.anchor for ref in self.references}
        found = set(re.findall(r"\[@([A-Za-z][A-Za-z0-9_-]*)\]", content))
        missing_def = found - anchors
        missing_use = anchors - found
        if missing_def:
            raise ValueError(f"undefined anchors: {missing_def}")
        if missing_use:
            raise ValueError(f"unused references: {missing_use}")
        return self




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
    search_tool = get_search_tool(config)
    tool_list = [search_tool, Sections, Introduction, Conclusion]
    return tool_list, {tool.name: tool for tool in tool_list}

def get_research_tools(config: RunnableConfig):
    """Get research tools based on configuration"""
    logger.debug("Getting research tools based on configuration")
    search_tool = get_search_tool(config)
    tool_list = [search_tool, Section]
    return tool_list, {tool.name: tool for tool in tool_list}

async def supervisor(state: ReportState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    logger.debug("Supervisor agent invoked with state: %s", state)
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

async def supervisor_tools(state: ReportState, config: RunnableConfig)  -> Command[Literal["supervisor", "research_team", "__end__"]]:
    """Performs the tool call and sends to the research agent"""

    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None

    logger.debug("Supervisor tools invoked with state: %s", state)

    # Get tools based on configuration
    _, supervisor_tools_by_name = get_supervisor_tools(config)
    
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

    logger.debug("Checking if supervisor should continue with state: %s", state)
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
    logger.debug("Research agent tools invoked with state: %s", state)
    
    # Get tools based on configuration
    _, research_tools_by_name = get_research_tools(config)
    
    # Process all tool calls first (required for OpenAI)
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool
        tool = research_tools_by_name[tool_call["name"]]
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
        
        # Store the section observation if a Section tool was called
        if tool_call["name"] == "Section":
            completed_section = observation
    
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
    

# Run the multi-agent workflow with the specified configuration
async def main():
    """Main function to run the multi-agent workflow"""

    """Build the multi-agent workflow"""
    
    logger.info("Building multi-agent workflow...")


    # Research agent workflow
    research_builder = StateGraph(SectionState, output=SectionOutputState, config_schema=Configuration)
    logger.debug("Adding nodes to research builder")
    research_builder.add_node("research_agent", research_agent)
    logger.debug("Adding research agent tools node")
    research_builder.add_node("research_agent_tools", research_agent_tools)
    logger.debug("Adding edges to research builder")
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
    logger.debug("Adding edge from research_agent_tools to research_agent")
    research_builder.add_edge("research_agent_tools", "research_agent")

    # Supervisor workflow
    logger.debug("Building supervisor workflow")
    supervisor_builder = StateGraph(ReportState, input=MessagesState, output=ReportStateOutput, config_schema=Configuration)
    supervisor_builder.add_node("supervisor", supervisor)
    logger.debug("Adding supervisor tools node")
    supervisor_builder.add_node("supervisor_tools", supervisor_tools)
    logger.debug("Adding research team node")
    supervisor_builder.add_node("research_team", research_builder.compile())

    # Flow of the supervisor agent
    logger.debug("Adding edges to supervisor builder")
    supervisor_builder.add_edge(START, "supervisor")
    logger.debug("Adding edge from supervisor to supervisor_tools")
    supervisor_builder.add_conditional_edges(
        "supervisor",
        supervisor_should_continue,
        {
            # Name returned by should_continue : Name of next node to visit
            "supervisor_tools": "supervisor_tools",
            END: END,
        },
    )

    logger.debug("Adding edge from supervisor_tools to research_team")
    supervisor_builder.add_edge("research_team", "supervisor")
    logger.debug("now compile..")
    graph = supervisor_builder.compile(checkpointer=MemorySaver())

    #graph.get_graph(xray=1).draw_mermaid_png(max_retries=5)
    save_mermaid_png(graph, xray=1, max_retries=5, output_dir="graphs")

    config = {
        "thread_id": str(uuid.uuid4()),
        "search_api": "azureaisearch",  # Use Azure AI Search as the default search API
        "supervisor_model": "azure_openai:gpt-4.1-2025-04-14",  # Use a specific model for the supervisor
        "researcher_model": "azure_openai:gpt-4.1-2025-04-14",  # Use a specific model for the research agents
        "number_of_queries": 6,  # Number of search queries to generate per iteration
        "max_search_depth": 4,  # Maximum number of reflection + search iterations
        }

    # Set up thread configuration with the specified parameters
    thread_config = {"configurable": config}

    # Define the research topic as a user message
    msg = [{"role": "user", "content": "What is model context protocol?"}]


    agent = graph.bind()
    
    
    # Start the agent with the initial message
    response = await agent.ainvoke({"messages": msg}, config=thread_config)
    
    print("\n")
    print("\n")
    print("############### RESPONSE FROM AGENT ##############")
    
    # print("\n")
    #for m in agent.get_state(thread_config).values['messages']:
    #
    #    print("----")
    #    m.pretty_print()
    #    print("----")
    print("##################################################")
    #print(agent.get_state(thread_config)).values['final_report']
    print("\n")
    print("##################################################")

    
    messages = agent.get_state(thread_config).values['messages']
    
    print(messages)
    print("##################################################")
    print("##################################################")
    print("##################################################")
    print("##################################################")
    messages[-1].pretty_print()


# run main
if __name__ == "__main__":
    
    asyncio.run(main())
    logger.info("Multi-agent workflow completed.")
    
