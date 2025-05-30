from typing import Literal
import builtins  # For explicit access to built-in functions

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from open_deep_research.pitch_state import (
    PitchStateInput,
    PitchStateOutput,
    Slides,
    Tagline,
    DiscordPost,
    PitchState,
    SlideState,
    SlideOutputState,
    Queries,
    Feedback
)

from open_deep_research.prompts import (
    pitch_planner_instructions,
    tagline_instructions,
    slide_writer_instructions,
    slide_writer_inputs,
    slide_grader_instructions,
    visual_suggestion_instructions,
    discord_post_instructions
)

from open_deep_research.configuration import Configuration
from open_deep_research.utils import (
    get_config_value, 
    get_search_params, 
    select_and_execute_search
)

## Nodes -- 

async def generate_pitch_plan(state: PitchState, config: RunnableConfig):
    """Generate the initial pitch deck plan with slides.
    
    This node:
    1. Gets configuration for the slide structure
    2. Uses an LLM to generate a structured pitch deck plan with 5-7 slides
    
    Args:
        state: Current graph state containing the pitch topic
        config: Configuration for models
        
    Returns:
        Dict containing the generated slides
    """

    # Inputs
    topic = state["topic"]
    feedback = state.get("feedback", None)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    
    # Get pitch organization from config
    pitch_organization = config.get("configurable", {}).get("pitch_organization", "Standard startup pitch format with Problem, Solution, Market, Business Model, Team, and Call to Action slides")

    # Set writer model (model used for pitch planning)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Slides)

    # Format system instructions
    system_instructions = pitch_planner_instructions.format(
        topic=topic, 
        pitch_organization=pitch_organization, 
        context="", 
        feedback=feedback or ""
    )

    # Generate slides plan
    results = await structured_llm.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate a 5-7 slide structure for a persuasive pitch deck.")
    ])

    # Return generated slides
    return {"slides": results.slides}

async def generate_tagline(state: PitchState, config: RunnableConfig):
    """Generate a catchy tagline for the pitch.
    
    This node:
    1. Uses the topic to generate a bold, memorable tagline
    2. Returns the tagline for use in the pitch deck
    
    Args:
        state: Current graph state with the pitch topic
        config: Configuration for models
        
    Returns:
        Dict containing the generated tagline
    """

    # Inputs
    topic = state["topic"]
    slides = state["slides"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    
    # Set writer model for tagline generation
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs)
    structured_llm = writer_model.with_structured_output(Tagline)
    
    # Format system instructions
    system_instructions = tagline_instructions.format(topic=topic, slides=slides)
    
    # Generate tagline
    result = await structured_llm.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate a catchy, bold tagline for this pitch.")
    ])
    
    # Return generated tagline
    return {"tagline": result.content}

def human_feedback(state: PitchState, config: RunnableConfig) -> Command[Literal["generate_pitch_plan", "build_slide_with_web_research"]]:
    """Get human feedback on the pitch plan and route to next steps.
    
    This node:
    1. Formats the current pitch plan for human review
    2. Gets feedback via an interrupt
    3. Routes to either:
       - Slide creation if plan is approved
       - Plan regeneration if feedback is provided
    
    Args:
        state: Current graph state with slides to review
        config: Configuration for the workflow
        
    Returns:
        Command to either regenerate plan or start slide creation
    """

    # Get slides and tagline
    topic = state["topic"]
    slides = state['slides']
    tagline = state.get('tagline', 'No tagline generated yet')
    
    slides_str = "\n\n".join(
        f"Slide: {slide.name}\n"
        f"Description: {slide.description}\n"
        f"Research needed: {'Yes' if slide.research else 'No'}\n"
        for slide in slides
    )

    # Get feedback on the pitch plan from interrupt
    interrupt_message = f"""Please review this pitch deck plan for: {topic}
                        
                        Tagline: {tagline}
                        
                        {slides_str}
                        
                        Does this pitch plan meet your needs?
                        Pass 'true' to approve the pitch plan.
                        Or, provide feedback to regenerate the pitch plan:"""
    
    feedback = interrupt(interrupt_message)

    # If the user approves the pitch plan, kick off slide creation
    if isinstance(feedback, bool) and feedback is True:
        # Treat this as approve and kick off slide writing
        return Command(goto=[
            Send("build_slide_with_web_research", {"topic": topic, "slide": s, "search_iterations": 0}) 
            for s in slides 
            if s.research
        ])
    
    # If the user provides feedback, regenerate the pitch plan 
    elif isinstance(feedback, str):
        # Treat this as feedback
        return Command(goto="generate_pitch_plan", 
                       update={"feedback": feedback})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")

async def generate_queries(state: SlideState, config: RunnableConfig):
    """Generate search queries for researching a specific slide.
    
    This node uses an LLM to generate targeted search queries based on the 
    slide topic and description.
    
    Args:
        state: Current state containing slide details
        config: Configuration including number of queries to generate
        
    Returns:
        Dict containing the generated search queries
    """

    # Get state 
    topic = state["topic"]
    slide = state["slide"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs)
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions - reusing query_writer_instructions but with slide context
    system_instructions = """You are an expert at creating targeted web search queries for pitch deck content.

    <Pitch topic>
    {topic}
    </Pitch topic>

    <Slide topic>
    {slide_description}
    </Slide topic>

    <Task>
    Generate {number_of_queries} search queries that will help gather compelling information for this slide.
    Focus on data, market insights, or evidence that will make this pitch persuasive to investors.
    </Task>

    <Format>
    Call the Queries tool 
    </Format>
    """.format(topic=topic, slide_description=slide.description, number_of_queries=number_of_queries)

    # Generate queries  
    queries = await structured_llm.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"Generate {number_of_queries} targeted search queries for the '{slide.name}' slide.")
    ])

    return {"search_queries": queries.queries}

async def search_web(state: SlideState, config: RunnableConfig):
    """Execute web searches for the slide queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

async def write_slide(state: SlideState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a slide of the pitch deck and evaluate if more research is needed.
    
    This node:
    1. Writes slide content using search results
    2. Evaluates the quality of the slide
    3. Either:
       - Completes the slide if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results and slide info
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete slide or do more research
    """

    # Get state 
    topic = state["topic"]
    slide = state["slide"]
    source_str = state["source_str"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Format inputs for slide writer
    slide_writer_inputs_formatted = slide_writer_inputs.format(
        topic=topic, 
        slide_name=slide.name, 
        slide_description=slide.description, 
        context=source_str, 
        slide_content=slide.content
    )

    # Generate slide content
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs)

    slide_content = await writer_model.ainvoke([
        SystemMessage(content=slide_writer_instructions),
        HumanMessage(content=slide_writer_inputs_formatted)
    ])
    
    # Write content to the slide object
    slide.content = slide_content.content

    # Grade the slide
    slide_grader_message = ("Grade the slide and consider follow-up questions for missing information. "
                          "If the grade is 'pass', return empty strings for all follow-up queries. "
                          "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    slide_grader_instructions_formatted = slide_grader_instructions.format(
        topic=topic, 
        slide_description=slide.description,
        slide_content=slide.content, 
        number_of_follow_up_queries=configurable.number_of_queries
    )

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})

    reflection_model = init_chat_model(
        model=planner_model, 
        model_provider=planner_provider, 
        model_kwargs=planner_model_kwargs
    ).with_structured_output(Feedback)
    
    # Generate feedback
    feedback = await reflection_model.ainvoke([
        SystemMessage(content=slide_grader_instructions_formatted),
        HumanMessage(content=slide_grader_message)
    ])

    # If the slide is passing or the max search depth is reached, publish the slide
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Update the current slide for the visual suggestion step
        return Command(
            update={"current_slide": slide},
            goto="suggest_visuals"
        )

    # Update the existing slide with new content and update search queries
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "slide": slide},
            goto="search_web"
        )

async def suggest_visuals(state: SlideState, config: RunnableConfig):
    """Suggest visuals for a slide to enhance its impact.
    
    This node:
    1. Takes a completed slide
    2. Generates visual suggestions to enhance the slide
    3. Adds the suggestion to the slide
    
    Args:
        state: Current state with the completed slide
        config: Configuration for models
        
    Returns:
        Dict with the slide updated with visual suggestions
    """
    
    # Get the current slide
    slide = state["current_slide"]
    topic = state["topic"]
    
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    
    # Set model for visual suggestions
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs)
    
    # Format visual suggestion instructions
    system_instructions = visual_suggestion_instructions.format(
        topic=topic,
        slide_name=slide.name,
        slide_content=slide.content
    )
    
    # Generate visual suggestion
    result = await writer_model.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"Suggest a compelling visual for the '{slide.name}' slide.")
    ])
    
    # Add visual suggestion to the slide
    slide.visual_suggestion = result.content
    
    # Add completed slide to output
    return {"completed_slides": [slide]}

def create_discord_post(state: PitchState, config: RunnableConfig):
    """Format pitch content for sharing on Discord.
    
    This node:
    1. Takes the tagline and key slides (problem and solution)
    2. Formats them into an engaging Discord post
    
    Args:
        state: Current state with tagline and completed slides
        config: Configuration for models
        
    Returns:
        Dict with formatted Discord post
    """
    
    # Get tagline and slides
    tagline = state["tagline"]
    slides = state["completed_slides"]
    
    # Find problem and solution slides
    problem_slide = next((s for s in slides if "Problem" in s.name), slides[0] if slides else None)
    solution_slide = next((s for s in slides if "Solution" in s.name), slides[1] if len(slides) > 1 else None)
    
    if not problem_slide or not solution_slide:
        return {"discord_post": {
            "title": tagline,
            "message": "Check out my pitch deck!",
            "picture_suggestion": "Project logo or mockup"
        }}
    
    # Create Discord post
    discord_post = {
        "title": tagline,
        "message": f"{problem_slide.description}\n\n{solution_slide.description}\n\nLooking for feedback on my pitch!",
        "picture_suggestion": solution_slide.visual_suggestion or "Project mockup or diagram"
    }
    
    return {"discord_post": discord_post}

def compile_final_pitch(state: PitchState):
    """Compile all slides into the final pitch deck.
    
    This node:
    1. Gets all completed slides
    2. Orders them according to original plan
    3. Combines them into the final pitch deck
    
    Args:
        state: Current state with all completed slides
        
    Returns:
        Dict containing the complete pitch deck
    """

    # Get slides and tagline
    slides = state["slides"]
    completed_slides = {s.name: s for s in state["completed_slides"]}
    tagline = state["tagline"]
    
    # Create title slide with tagline
    title_content = f"# {tagline}\n\n"
    
    # Update slides with completed content while maintaining original order
    pitch_content = []
    for slide in slides:
        if slide.name in completed_slides:
            completed_slide = completed_slides[slide.name]
            slide_text = completed_slide.content
            
            # Add visual suggestion if available
            if completed_slide.visual_suggestion:
                slide_text += f"\n\n*Visual: {completed_slide.visual_suggestion}*"
                
            pitch_content.append(slide_text)
    
    # Compile final pitch deck
    all_slides = title_content + "\n\n".join(pitch_content)

    return {
        "final_pitch": all_slides,
    }

# Slide builder sub-graph -- 

# Add nodes 
slide_builder = StateGraph(SlideState, output=SlideOutputState)
slide_builder.add_node("generate_queries", generate_queries)
slide_builder.add_node("search_web", search_web)
slide_builder.add_node("write_slide", write_slide)
slide_builder.add_node("suggest_visuals", suggest_visuals)

# Add edges
slide_builder.add_edge(START, "generate_queries")
slide_builder.add_edge("generate_queries", "search_web")
slide_builder.add_edge("search_web", "write_slide")
slide_builder.add_edge("suggest_visuals", END)

# Outer graph for pitch deck generation

# Add nodes
builder = StateGraph(PitchState, input=PitchStateInput, output=PitchStateOutput, config_schema=Configuration)
builder.add_node("generate_pitch_plan", generate_pitch_plan)
builder.add_node("generate_tagline", generate_tagline)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_slide_with_web_research", slide_builder.compile())
builder.add_node("create_discord_post", create_discord_post)
builder.add_node("compile_final_pitch", compile_final_pitch)

# Add edges
builder.add_edge(START, "generate_pitch_plan")
builder.add_edge("generate_pitch_plan", "generate_tagline")
builder.add_edge("generate_tagline", "human_feedback")
builder.add_edge("build_slide_with_web_research", "create_discord_post")
builder.add_edge("create_discord_post", "compile_final_pitch")
builder.add_edge("compile_final_pitch", END)

# Build non-research slides
def include_non_research_slides(state: PitchState):
    """Create completed slides for those that don't require research"""
    # Get config and model
    topic = state["topic"]
    slides = state["slides"]
    tagline = state["tagline"]
    
    # Filter slides that don't need research
    non_research_slides = [s for s in slides if not s.research]
    
    if not non_research_slides:
        return {}
    
    # Return completed non-research slides to be added to the state
    return {"completed_slides": non_research_slides}

# Add nodes
builder.add_node("include_non_research_slides", include_non_research_slides)

# Add edges
builder.add_edge("human_feedback", "include_non_research_slides")
builder.add_conditional_edges(
    "include_non_research_slides",
    lambda state: [
        Send("build_slide_with_web_research", {"topic": state["topic"], "slide": s, "search_iterations": 0}) 
        for s in state["slides"] 
        if s.research
    ],
    ["build_slide_with_web_research"]
)

graph = builder.compile()
