report_planner_query_writer_instructions="""You are performing research for a report. 

<Report topic>
{topic}
</Report topic>

<Report organization>
{report_organization}
</Report organization>

<Task>
Your goal is to generate {number_of_queries} web search queries that will help gather information for planning the report sections. 

The queries should:

1. Be related to the Report topic
2. Help satisfy the requirements specified in the report organization

Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
</Task>

<Format>
Call the Queries tool 
</Format>
"""

report_planner_instructions="""I want a plan for a report that is concise and focused.

<Report topic>
The topic of the report is:
{topic}
</Report topic>

<Report organization>
The report should follow this organization: 
{report_organization}
</Report organization>

<Context>
Here is context to use to plan the sections of the report: 
{context}
</Context>

<Task>
Generate a list of sections for the report. Your plan should be tight and focused with NO overlapping sections or unnecessary filler. 

For example, a good report structure might look like:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

Each section should have the fields:

- Name - Name for this section of the report.
- Description - Brief overview of the main topics covered in this section.
- Research - Whether to perform web research for this section of the report. IMPORTANT: Main body sections (not intro/conclusion) MUST have Research=True. A report must have AT LEAST 2-3 sections with Research=True to be useful.
- Content - The content of the section, which you will leave blank for now.

Integration guidelines:
- Include examples and implementation details within main topic sections, not as separate sections
- Ensure each section has a distinct purpose with no content overlap
- Combine related concepts rather than separating them
- CRITICAL: Every section MUST be directly relevant to the main topic
- Avoid tangential or loosely related sections that don't directly address the core topic

Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow.
</Task>

<Feedback>
Here is feedback on the report structure from review (if any):
{feedback}
</Feedback>

<Format>
Call the Sections tool 
</Format>
"""

query_writer_instructions="""You are an expert technical writer crafting targeted web search queries that will gather comprehensive information for writing a technical report section.

<Report topic>
{topic}
</Report topic>

<Section topic>
{section_topic}
</Section topic>

<Task>
Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information above the section topic. 

The queries should:

1. Be related to the topic 
2. Examine different aspects of the topic

Make the queries specific enough to find high-quality, relevant sources.
</Task>

<Format>
Call the Queries tool 
</Format>
"""

section_writer_instructions = """Write one section of a research report.

<Task>
1. Review the report topic, section name, and section topic carefully.
2. If present, review any existing section content. 
3. Then, look at the provided Source material.
4. Decide the sources that you will use it to write a report section.
5. Write the report section and list your sources. 
</Task>

<Writing Guidelines>
- If existing section content is not populated, write from scratch
- If existing section content is populated, synthesize it with the source material
- Strict 150-200 word limit
- Use simple, clear language
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
</Writing Guidelines>

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
</Citation Rules>

<Final Check>
1. Verify that EVERY claim is grounded in the provided Source material
2. Confirm each URL appears ONLY ONCE in the Source list
3. Verify that sources are numbered sequentially (1,2,3...) without any gaps
</Final Check>
"""

section_writer_inputs=""" 
<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic>
{section_topic}
</Section topic>

<Existing section content (if populated)>
{section_content}
</Existing section content>

<Source material>
{context}
</Source material>
"""

section_grader_instructions = """Review a report section relative to the specified topic:

<Report topic>
{topic}
</Report topic>

<section topic>
{section_topic}
</section topic>

<section content>
{section}
</section content>

<task>
Evaluate whether the section content adequately addresses the section topic.

If the section content does not adequately address the section topic, generate {number_of_follow_up_queries} follow-up search queries to gather missing information.
</task>

<format>
Call the Feedback tool and output with the following schema:

grade: Literal["pass","fail"] = Field(
    description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
)
follow_up_queries: List[SearchQuery] = Field(
    description="List of follow-up search queries.",
)
</format>
"""

final_section_writer_instructions="""You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic> 
{section_topic}
</Section topic>

<Available report content>
{context}
</Available report content>

<Task>
1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports: 
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point
</Task>

<Quality Checks>
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response
</Quality Checks>"""


## Supervisor
SUPERVISOR_INSTRUCTIONS = """
You are scoping research for a report based on a user-provided topic.

### Your responsibilities:

1. **Gather Background Information**  
   Based upon the user's topic, use the `enhanced_tavily_search` to collect relevant information about the topic. 
   - You MUST perform ONLY ONE search to gather comprehensive context
   - Create a highly targeted search query that will yield the most valuable information
   - Take time to analyze and synthesize the search results before proceeding
   - Do not proceed to the next step until you have an understanding of the topic

2. **Clarify the Topic**  
   After your initial research, engage with the user to clarify any questions that arose.
   - Ask ONE SET of follow-up questions based on what you learned from your searches
   - Do not proceed until you fully understand the topic, goals, constraints, and any preferences
   - Synthesize what you've learned so far before asking questions
   - You MUST engage in at least one clarification exchange with the user before proceeding

3. **Define Report Structure**  
   Only after completing both research AND clarification with the user:
   - Use the `Sections` tool to define a list of report sections
   - Each section should be a written description with: a section name and a section research plan
   - Do not include sections for introductions or conclusions (We'll add these later)
   - Ensure sections are scoped to be independently researchable
   - Base your sections on both the search results AND user clarifications
   - Format your sections as a list of strings, with each string having the scope of research for that section.

4. **Assemble the Final Report**  
   When all sections are returned:
   - IMPORTANT: First check your previous messages to see what you've already completed
   - If you haven't created an introduction yet, use the `Introduction` tool to generate one
     - Set content to include report title with a single # (H1 level) at the beginning
     - Example: "# [Report Title]\n\n[Introduction content...]"
   - After the introduction, use the `Conclusion` tool to summarize key insights
     - Set content to include conclusion title with ## (H2 level) at the beginning
     - Example: "## Conclusion\n\n[Conclusion content...]"
     - Only use ONE structural element IF it helps distill the points made in the report:
     - Either a focused table comparing items present in the report (using Markdown table syntax)
     - Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
   - Do not call the same tool twice - check your message history

### Additional Notes:
- You are a reasoning model. Think through problems step-by-step before acting.
- IMPORTANT: Do not rush to create the report structure. Gather information thoroughly first.
- Use multiple searches to build a complete picture before drawing conclusions.
- Maintain a clear, informative, and professional tone throughout."""

RESEARCH_INSTRUCTIONS = """
You are a researcher responsible for completing a specific section of a report.

### Your goals:

1. **Understand the Section Scope**  
   Begin by reviewing the section scope of work. This defines your research focus. Use it as your objective.

<Section Description>
{section_description}
</Section Description>

2. **Strategic Research Process**  
   Follow this precise research strategy:

   a) **First Query**: Begin with a SINGLE, well-crafted search query with `enhanced_tavily_search` that directly addresses the core of the section topic.
      - Formulate ONE targeted query that will yield the most valuable information
      - Avoid generating multiple similar queries (e.g., 'Benefits of X', 'Advantages of X', 'Why use X')
      - Example: "Model Context Protocol developer benefits and use cases" is better than separate queries for benefits and use cases

   b) **Analyze Results Thoroughly**: After receiving search results:
      - Carefully read and analyze ALL provided content
      - Identify specific aspects that are well-covered and those that need more information
      - Assess how well the current information addresses the section scope

   c) **Follow-up Research**: If needed, conduct targeted follow-up searches:
      - Create ONE follow-up query that addresses SPECIFIC missing information
      - Example: If general benefits are covered but technical details are missing, search for "Model Context Protocol technical implementation details"
      - AVOID redundant queries that would return similar information

   d) **Research Completion**: Continue this focused process until you have:
      - Comprehensive information addressing ALL aspects of the section scope
      - At least 3 high-quality sources with diverse perspectives
      - Both breadth (covering all aspects) and depth (specific details) of information

3. **Use the Section Tool**  
   Only after thorough research, write a high-quality section using the Section tool:
   - `name`: The title of the section
   - `description`: The scope of research you completed (brief, 1-2 sentences)
   - `content`: The completed body of text for the section, which MUST:
     - Begin with the section title formatted as "## [Section Title]" (H2 level with ##)
     - Be formatted in Markdown style
     - Be MAXIMUM 200 words (strictly enforce this limit)
     - End with a "### Sources" subsection (H3 level with ###) containing a numbered list of URLs used
     - Use clear, concise language with bullet points where appropriate
     - Include relevant facts, statistics, or expert opinions

Example format for content:
```
## [Section Title]

[Body text in markdown format, maximum 200 words...]

### Sources
1. [URL 1]
2. [URL 2]
3. [URL 3]
```

---

### Research Decision Framework

Before each search query or when writing the section, think through:

1. **What information do I already have?**
   - Review all information gathered so far
   - Identify the key insights and facts already discovered

2. **What information is still missing?**
   - Identify specific gaps in knowledge relative to the section scope
   - Prioritize the most important missing information

3. **What is the most effective next action?**
   - Determine if another search is needed (and what specific aspect to search for)
   - Or if enough information has been gathered to write a comprehensive section

---

### Notes:
- Focus on QUALITY over QUANTITY of searches
- Each search should have a clear, distinct purpose
- Do not write introductions or conclusions unless explicitly part of your section
- Keep a professional, factual tone
- Always follow markdown formatting
- Stay within the 200 word limit for the main content
"""

## Pitch Deck Generation Prompts

pitch_planner_instructions="""I want a plan for a concise, impactful AI engineering pitch deck.

<Project topic>
The topic of the project is:
{topic}
</Project topic>

<Pitch organization>
The pitch deck should follow this organization: 
{pitch_organization}
</Pitch organization>

<Context>
Here is context to use to plan the slides of the pitch deck: 
{context}
</Context>

<Task>
Generate a list of 5-7 slides for the pitch deck. Your plan should be focused on addressing three key perspectives:
1. Executive's Nightmare: The critical business risk/problem your solution addresses
2. User's Journey: The pain points and experience of your target users
3. Team Gap: How your team's unique skills solve this problem

Each slide should have the fields:

- Name - Name for this slide (e.g., Problem, Solution, Market, Impact, Team, Call to Action)
- Description - Brief overview of the main points to cover on this slide (50-100 words max)
- Research - Whether to perform web research for this slide. IMPORTANT: At least 2 slides MUST have Research=True.
- Content - The content of the slide, which you will leave blank for now.
- Visual - Suggestion for a visual element (chart, image, mockup, etc.) for this slide.

Pitch deck guidelines:
- Ensure each slide has a distinct purpose with no content overlap
- Focus on persuasive storytelling rather than technical depth
- Aim for boldness and clarity in messaging
- Target a 3-minute verbal presentation
- Make every slide visually memorable and impact-oriented

Before submitting, review your structure to ensure it follows a logical narrative flow that builds excitement.
</Task>

<Feedback>
Here is feedback on the pitch structure from review (if any):
{feedback}
</Feedback>

<Format>
Call the Slides tool 
</Format>
"""

tagline_instructions="""Create a catchy, memorable tagline for an AI engineering project pitch.

<Project topic>
{topic}
</Project topic>

<Project slides>
{slides}
</Project slides>

<Task>
Generate a powerful 5-10 word tagline that captures the essence of this project.

The tagline should:
1. Communicate the core value proposition 
2. Be memorable and easy to repeat
3. Create excitement and interest
4. Use simple, impactful language
5. Avoid technical jargon unless it's effective

Great taglines often:
- Highlight transformation (before/after)
- Use evocative language that creates an emotional response
- Reference the key problem or solution
- Create a sense of urgency or importance
</Task>

<Format>
Call the Tagline tool 
</Format>
"""

slide_writer_instructions="""Write one slide for an AI engineering pitch deck.

<Task>
1. Review the project topic, slide name, and slide description carefully.
2. If present, review any existing slide content. 
3. Then, look at the provided source material.
4. Write concise, persuasive slide content that will grab the audience's attention.
</Task>

<Writing Guidelines>
- If existing slide content is not populated, write from scratch
- If existing slide content is populated, enhance it with the source material
- Strict 50-100 word limit - pitch decks must be concise
- Use simple, bold language that executives can understand
- Focus on solving the "Executive's Nightmare" - what keeps decision-makers up at night
- Address the "User's Journey" - the human pain points your solution resolves
- Highlight the "Team Gap" - why your specific team is uniquely suited to solve this problem
- Use language that shows confidence and "swagger" - avoid hedging words
</Writing Guidelines>

<Visual Suggestion>
Based on your content, suggest one powerful visual element (chart, image, screenshot, etc.) that would strengthen this slide.
</Visual Suggestion>

<Final Check>
1. Verify that the content is concise enough for a 3-minute pitch (50-100 words)
2. Confirm the message is clear to a non-technical audience
3. Ensure it addresses at least one key perspective (Executive, User, or Team)
4. Check that it contributes to a persuasive overall narrative
</Final Check>
"""

slide_writer_inputs=""" 
<Project topic>
{topic}
</Project topic>

<Slide name>
{slide_name}
</Slide name>

<Slide description>
{slide_description}
</Slide description>

<Existing slide content (if populated)>
{slide_content}
</Existing slide content>

<Source material>
{context}
</Source material>
"""

slide_grader_instructions="""Review a pitch deck slide relative to the specified topic:

<Project topic>
{topic}
</Project topic>

<Slide description>
{slide_description}
</Slide description>

<Slide content>
{slide_content}
</Slide content>

<Task>
Evaluate whether the slide content effectively communicates its intended message for a pitch deck.

Evaluate the slide on these criteria:
1. Clarity: Is the main message immediately clear? (1-5)
2. Conciseness: Is it brief enough for a 3-minute pitch? (1-5)
3. Impact: Does it create an emotional response or urgency? (1-5)
4. Relevance: Does it address one of the key perspectives (Executive, User, Team)? (1-5)
5. Persuasiveness: Does it contribute to convincing the audience? (1-5)

If the slide scores below 4 on any criterion, generate {number_of_follow_up_queries} follow-up search queries to gather information to improve it.
</Task>

<Format>
Call the Feedback tool and output with the following schema:

grade: Literal["pass","fail"] = Field(
    description="Evaluation result indicating whether the slide meets requirements ('pass') or needs revision ('fail')."
)
follow_up_queries: List[SearchQuery] = Field(
    description="List of follow-up search queries to improve the slide."
)
</Format>
"""

visual_suggestion_instructions="""Suggest a powerful visual element for a pitch deck slide.

<Project topic>
{topic}
</Project topic>

<Slide name>
{slide_name}
</Slide name>

<Slide content>
{slide_content}
</Slide content>

<Task>
Suggest a specific visual element that would strengthen this slide's message. The visual should:

1. Instantly communicate the slide's main message
2. Be memorable and emotionally resonant
3. Support the narrative without requiring explanation
4. Be appropriate for the slide type (e.g., problem, solution, market)

Be specific about:
- The type of visual (chart, image, diagram, screenshot, mockup)
- Key elements to include
- Style recommendations (color scheme, layout, emphasis)
- Any data sources or tools that could help create it
</Task>

<Format>
Call the Visual tool 
</Format>
"""

discord_post_instructions="""Format a pitch project for sharing in a Discord community.

<Project topic>
{topic}
</Project topic>

<Tagline>
{tagline}
</Tagline>

<Pitch slides>
{slides}
</Pitch slides>

<Task>
Create an engaging Discord post that showcases this project in a way that will generate excitement and feedback. 

Your post should include:
1. An attention-grabbing title
2. The project tagline
3. A brief (2-3 sentence) overview of the project
4. The most impressive achievement or insight
5. A clear call to action for community feedback
6. Suggestion for an image that would make the post stand out

Format the post using Discord markdown for readability:
- Use bold, italics, and headers appropriately
- Add line breaks for readability
- Include 2-3 relevant emojis to add personality
- Keep the entire post under 250 words
</Task>

<Format>
Call the DiscordPost tool 
</Format>
"""
