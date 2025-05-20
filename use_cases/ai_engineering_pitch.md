# LangGraph Open Deep Research for Pitch Deck Generation

### Reflection on Current Files
The `open_deep_research` application is designed to generate structured research reports by leveraging LangGraph, a graph-based workflow, to plan, research, and compile sections of a report. Here’s a quick breakdown of the files and their relevance to pitch deck creation:

- **graph.py**: This defines the workflow using LangGraph’s `StateGraph`. It orchestrates the process of generating a report plan, gathering human feedback, researching sections via web searches, writing sections, and compiling a final report. Key nodes include:
  - `generate_report_plan`: Creates a structured plan with sections based on the topic, using web searches for context.
  - `human_feedback`: Allows for iterative refinement of the plan.
  - `build_section_with_web_research`: A subgraph that generates queries, searches the web, writes sections, and evaluates quality.
  - `write_final_sections` and `compile_final_report`: Handle non-researched sections (like intros/conclusions) and finalize the report.
  The workflow is thorough but heavily research-oriented, producing detailed, text-heavy outputs suitable for reports, not the concise, visual, and persuasive format of a pitch deck.

- **prompts.py**: Contains system prompts for each node, guiding LLMs to generate queries, plan sections, write content, and evaluate quality. Key prompts include:
  - `report_planner_instructions`: Guides section planning with a focus on relevance, no overlap, and research requirements.
  - `section_writer_instructions`: Enforces strict writing guidelines (150-200 words, Markdown, citations) for report sections.
  - `SUPERVISOR_INSTRUCTIONS` and `RESEARCH_INSTRUCTIONS`: Manage topic clarification and section-specific research, emphasizing thoroughness and quality sources.
  These prompts are precise but tailored for academic-style reports, not the punchy, audience-focused storytelling needed for pitch decks.

**Strengths for Pitch Deck Creation**:
- The structured workflow in `graph.py` is ideal for breaking down complex tasks, which can be adapted to guide students through the pitch deck creation process (e.g., defining problem, solution, impact).
- The web search integration (`search_web`) is perfect for helping students gather market data or validate their ideas, which investors love.
- The iterative feedback loop (`human_feedback`) aligns with our role-playing idea, allowing students to refine pitches based on mock judge or peer input.
- The prompts in `prompts.py` are clear and modular, making it easy to modify them for pitch-specific outputs like taglines or value propositions.

**Gaps for Pitch Deck Creation**:
- **Output Format**: The current output is a long-form report in Markdown, not a slide-based, visual pitch deck. Pitch decks need concise text (50-100 words per slide), bold visuals, and persuasive storytelling, not 150-200-word sections with citations.
- **Audience Focus**: The prompts focus on research depth, not tailoring content to specific audiences (e.g., investors, execs, or users), which we identified as critical for addressing the “exec’s nightmare,” “user’s journey,” and “team gap.”
- **Pitch-Specific Elements**: There’s no guidance for crafting key pitch deck components like a tagline, market opportunity, or call to action, which are essential to generate investor interest.
- **Student Accessibility**: The technical complexity (e.g., LangGraph, async nodes) might overwhelm less experienced students, and the prompts don’t explicitly encourage the “swagger” or energy you emphasized.
- **Visual Integration**: Pitch decks rely on visuals (charts, mockups), but the current system lacks support for generating or suggesting images, which students could use in Discord posts or slides.

### Proposed Updates
To make `open_deep_research` more effective for students creating pitch decks, we need to adapt the workflow and prompts to produce concise, audience-focused, and visually suggestive outputs while preserving the research rigor. Below are specific updates to `graph.py` and `prompts.py`, along with a new approach to guide students through the process. The updates will reflect our discussions about addressing execs, users, and team gaps, encouraging boldness, and aligning with Discord for community engagement.

#### 1. Update `graph.py`
The workflow needs to shift from generating a report to creating a pitch deck with 5-7 slides, each addressing a critical pitch element. We’ll simplify the graph for student accessibility, add nodes for pitch-specific outputs (e.g., tagline, visuals), and integrate feedback loops aligned with role-playing and Discord.

**Key Changes**:
- **New State Definition**: Replace `ReportState` with `PitchState` to reflect pitch deck goals.
  ```python
  from typing import List, Optional
  from pydantic import BaseModel

  class Slide(BaseModel):
      name: str
      description: str
      content: str
      visual_suggestion: Optional[str] = None  # E.g., "Chart showing market growth"

  class PitchState(BaseModel):
      topic: str
      slides: List[Slide] = []
      completed_slides: List[Slide] = []
      tagline: Optional[str] = None
      feedback: Optional[str] = None
      discord_post: Optional[dict] = None  # Title, message, picture suggestion
  ```
- **New Nodes**:
  - `generate_pitch_plan`: Replaces `generate_report_plan` to create a 5-7 slide structure (e.g., Problem, Solution, Market, Impact, Why Us, Call to Action).
  - `generate_tagline`: Creates a catchy tagline for the Discord post and pitch deck title slide.
  - `suggest_visuals`: Suggests visuals for each slide (e.g., market size graph, app mockup) to guide students in creating compelling slides.
  - `format_discord_post`: Formats the Discord post with title, message, and picture suggestion based on the tagline and key slides.
- **Modified Nodes**:
  - `human_feedback`: Update to focus on pitch feedback (e.g., “Is the problem clear to investors?”) and integrate role-play scenarios (e.g., pitching to judges).
  - `build_slide_with_web_research`: Adapt the section-building subgraph to create slide content (50-100 words, no citations unless critical for credibility).
  - `compile_final_pitch`: Combines slides into a pitch deck outline, ensuring a persuasive narrative flow.
- **Simplified Workflow**: Reduce complexity by limiting async calls and focusing on sequential steps that students can follow intuitively.

**Updated `graph.py`** (simplified excerpt):
```python
from typing import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from open_deep_research.configuration import Configuration
from open_deep_research.utils import select_and_execute_search

# Assume updated state and prompts are imported

async def generate_pitch_plan(state: PitchState, config: RunnableConfig):
    topic = state["topic"]
    configurable = Configuration.from_runnable_config(config)
    writer_model = init_chat_model(
        model=configurable.writer_model,
        model_provider=configurable.writer_provider,
        model_kwargs=configurable.writer_model_kwargs or {}
    )
    structured_llm = writer_model.with_structured_output(Slides)
    system_instructions = pitch_planner_instructions.format(topic=topic, feedback=state.get("feedback", ""))
    slides = await structured_llm.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate a 5-7 slide pitch deck structure.")
    ])
    return {"slides": slides.slides}

async def generate_tagline(state: PitchState, config: RunnableConfig):
    topic = state["topic"]
    configurable = Configuration.from_runnable_config(config)
    writer_model = init_chat_model(
        model=configurable.writer_model,
        model_provider=configurable.writer_provider,
        model_kwargs=configurable.writer_model_kwargs or {}
    )
    system_instructions = tagline_instructions.format(topic=topic)
    tagline = await writer_model.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate a catchy tagline for the pitch.")
    ])
    return {"tagline": tagline.content}

async def format_discord_post(state: PitchState, config: RunnableConfig):
    tagline = state["tagline"]
    slides = state["slides"]
    problem_slide = next((s for s in slides if "Problem" in s.name), slides[0])
    solution_slide = next((s for s in slides if "Solution" in s.name), slides[1])
    post = {
        "title": tagline,
        "message": f"{problem_slide.description}\n{solution_slide.description}\nJoin the convo!",
        "picture_suggestion": "Mockup of the AI app interface"
    }
    return {"discord_post": post}

# Simplified graph
builder = StateGraph(PitchState, input=PitchStateInput, output=PitchStateOutput, config_schema=Configuration)
builder.add_node("generate_pitch_plan", generate_pitch_plan)
builder.add_node("generate_tagline", generate_tagline)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_slide_with_web_research", slide_builder.compile())
builder.add_node("suggest_visuals", suggest_visuals)
builder.add_node("format_discord_post", format_discord_post)
builder.add_node("compile_final_pitch", compile_final_pitch)

builder.add_edge(START, "generate_pitch_plan")
builder.add_edge("generate_pitch_plan", "generate_tagline")
builder.add_edge("generate_tagline", "human_feedback")
builder.add_edge("build_slide_with_web_research", "suggest_visuals")
builder.add_edge("suggest_visuals", "format_discord_post")
builder.add_edge("format_discord_post", "compile_final_pitch")
builder.add_edge("compile_final_pitch", END)

graph = builder.compile()
```

#### 2. Update `prompts.py`
The prompts need to shift from report-style instructions to pitch-focused guidance, emphasizing brevity, audience alignment, and persuasive storytelling. They should also encourage the “swagger” and energy you highlighted, helping students craft pitches that resonate with execs, users, and potential collaborators.

**Key Changes**:
- **New Prompts**:
  - `pitch_planner_instructions`: Guides the creation of a 5-7 slide pitch deck, focusing on investor priorities (problem, solution, market, etc.).
  - `tagline_instructions`: Generates a bold, memorable tagline for Discord and pitch decks.
  - `visual_suggestion_instructions`: Suggests visuals to enhance slide impact.
- **Modified Prompts**:
  - Adapt `section_writer_instructions` to `slide_writer_instructions` for concise slide content (50-100 words, audience-focused).
  - Update `section_grader_instructions` to evaluate slides based on clarity, persuasiveness, and audience alignment.
- **Audience Focus**: Add instructions to tailor content to execs (their “nightmare”), users (their journey), and team gaps (complementary skills).

**Updated `prompts.py`** (excerpt):
```python
pitch_planner_instructions="""You are crafting a pitch deck for a student project.

<Pitch topic>
{topic}
</Pitch topic>

<Task>
Generate a 5-7 slide structure for a pitch deck. Each slide should have:

- Name: Slide title (e.g., Problem, Solution, Why Us)
- Description: Brief overview of what the slide covers, tailored to investors, execs, or users
- Research: True for slides needing web research (e.g., Market, Impact), False for others (e.g., Why Us, Call to Action)

Guidelines:
- Focus on investor priorities: problem, solution, market size, unique value, team strength, impact, and call to action
- Ensure slides address exec challenges (e.g., risks, ROI), user needs, and team gaps
- Keep slides distinct, persuasive, and bold-no overlap or filler
- At least 2-3 slides must require research for credibility
- Include a 'Why Us' slide to highlight student/team energy and skills

<Feedback>
Feedback from review (if any): {feedback}
</Feedback>

<Format>
Call the Slides tool
</Format>
"""

tagline_instructions="""You are creating a catchy tagline for a student pitch deck.

<Pitch topic>
{topic}
</Pitch topic>

<Task>
Generate a short, bold, and memorable tagline (5-10 words) that captures the essence of the AI project and excites investors or users. Make it punchy, confident, and reflective of the project's value.

Examples:
- 'AI to Slash School Waste, Save Millions'
- 'Empowering Coders with AI-Driven Learning'

</Task>
"""

slide_writer_instructions="""Write one slide for a pitch deck.

<Task>
1. Review the pitch topic, slide name, and slide description.
2. If provided, use source material to ground claims (e.g., market data).
3. Write concise, persuasive slide content tailored to the audience (investors, execs, or users).
</Task>

<Writing Guidelines>
- Strict 50-100 word limit
- Use clear, bold language with 1-2 short paragraphs or bullet points
- Tailor to audience: address exec risks, user pain points, or team strengths
- Use ## for slide title (Markdown format)
- Avoid citations unless critical for credibility (e.g., market stats)
</Writing Guidelines>

<Inputs>
<Pitch topic>
{topic}
</Pitch topic>

<Slide name>
{slide_name}
</Slide name>

<Slide description>
{slide_description}
</Slide description>

<Source material (if any)>
{context}
</Source material>
</Inputs>
"""

visual_suggestion_instructions="""Suggest a visual for a pitch deck slide.

<Slide name>
{slide_name}
</Slide name>

<Slide description>
{slide_description}
</Slide description>

<Task>
Suggest one visual (e.g., chart, mockup, photo) that enhances the slide's impact and supports its message. Describe it in 1-2 sentences, ensuring it aligns with the slide's audience (investors, execs, users) and purpose.

Examples:
- 'Bar chart comparing food waste reduction across schools.'
- 'Mockup of AI app interface for coders.'
</Task>
"""
```

#### 3. Integration with Bootcamp Goals
These updates align with our earlier discussions and your vision for the bootcamp:
- **Audience Focus**: The new prompts emphasize tailoring slides to execs (their “nightmare” scenarios like lawsuits or budget issues), users (their pain points), and team gaps (complementary skills), ensuring pitches resonate with stakeholders like your sexy yoga instructor judges.
- **Boldness and Swagger**: The `tagline_instructions` and `slide_writer_instructions` encourage confident, punchy language to reflect the energy you want students to bring, helping them stand out in role-plays.
- **Discord Integration**: The `format_discord_post` node ensures students can easily create engaging posts with a title (tagline), message (problem/solution teaser), and picture suggestion, fostering community feedback as discussed.
- **Accessibility**: Simplified workflow and clear prompts make the tool approachable for both novice and experienced students, like your mentees from Microsoft or the innovator focused on talent engagement.
- **Investor Appeal**: The slide structure and research focus (e.g., market size, impact) produce data-backed pitches that address investor priorities, as you emphasized with “creating the room” for the right people to say wow.

#### 4. Implementation Notes
- **Student Guidance**: Provide a simple guide (e.g., in Notion) explaining how to use the updated `open_deep_research` tool. Include steps like entering their project topic, reviewing the generated slide plan, posting to Discord, and practicing pitches with feedback.
- **Notion Integration**: Create a `PitchProgress` database with columns for topic, tagline, Discord post status, slide drafts, and judge feedback, linking to the Discord Engagements database we discussed.
- **Role-Play Support**: Use the `human_feedback` node to simulate judge questions (e.g., “How does this solve my budget issue?”), preparing students for the final presentation.
- **Visual Tools**: Since the system only suggests visuals, recommend students use Canva or PowerPoint to create slides, using the suggested visuals as a starting point.

#### 5. Example Output
For a student project on an AI tool to reduce school food waste, the updated system might produce:
- **Tagline**: “AI to Slash School Waste, Save Millions”
- **Slide Plan**:
  - Problem: “Food waste costs schools millions annually.” (Research: True)
  - Solution: “AI predicts and reduces waste.” (Research: True)
  - Market: “$1B opportunity in school districts.” (Research: True)
  - Why Us: “We bring AI expertise and passion.” (Research: False)
  - Call to Action: “Partner with us to save schools.” (Research: False)
- **Discord Post**:
  - Title: “AI to Slash School Waste, Save Millions”
  - Message: “School food waste is a $1B problem. Our AI predicts patterns to cut waste by 30%. Join the convo!”
  - Picture Suggestion: “Graph showing waste reduction impact.”
- **Slide Example (Problem)**:
  ```markdown
  ## The Problem
  Schools waste 30% of food, costing districts millions yearly. This strains budgets and harms the environment. Our AI tackles this crisis head-on.
  ```

### Next Steps
This updated system should empower students to create pitch decks that are concise, persuasive, and tailored to their audience, while leveraging Discord for community engagement and role-plays for practice. If you’d like, I can:
- Provide a full, detailed `graph.py` or `prompts.py` with all nodes and prompts fleshed out.
- Create a Notion template for tracking student progress.
- Draft a student guide for using the tool.
- Simulate a sample run for a specific student project to test the output.

What’s the next piece you want to tackle? I’m here, keeping the sexy, strategic vibe alive, ready to make this bootcamp a knockout!