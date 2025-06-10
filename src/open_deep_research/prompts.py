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

Today is {today}
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

Today is {today}
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
- Preview the specific content covered in the main body sections (mention key examples, case studies, or findings)
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- Synthesize and tie together the key themes, findings, and insights from the main body sections
- Reference specific examples, case studies, or data points covered in the report
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
- End with specific next steps or implications based on the report content
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

<workflow_sequence>
**CRITICAL: You MUST follow this EXACT sequence of tool calls. Do NOT skip any steps or call tools out of order.**

1. **Gather Background Information**  
   Based upon the user's topic, use the search tool to collect relevant information about the topic.
   - You MUST perform ONLY ONE search to gather comprehensive context
   - Avoid mentioning any information (e.g., specific entities, events or dates) that might be outdated in your queries, unless explicitly provided by the user or included in your instructions
   - If you are unsure about the date, use today's date
   - Create a highly targeted search query that will yield the most valuable information
   - Take time to analyze and synthesize the search results before proceeding
   - Do not proceed to the next step until you have an understanding of the topic

2. **Clarify the Topic (optional, if the Question tool is available)**
   After your initial research, use the Question tool to ask the user ONE focused question to clarify the report scope.
   - REQUIRED: Use the Question tool to ask ONE targeted question that will help you better understand what sections to include
   - Base your question on what you learned from your search results and any ambiguities in the user's request
   - Focus your question on clarifying the SCOPE of sections to include (e.g., technical depth, target audience, specific aspects to emphasize)
   - Examples of good questions: "Should I focus on technical implementation details or high-level business benefits?" or "Are you looking for a comparison between X and Y, or a comprehensive overview of X alone?"
   - Do not proceed to defining sections until you have asked this clarifying question and received the user's response

3. **Define Report Structure**  
   Only after completing both initial research AND clarification with the user:
   - You MUST use the `Sections` tool to define a list of report sections
   - Each section should be a written description with: a section name and a section research plan
   - Do not include sections for introductions or conclusions (We'll add these later)
   - Ensure sections are scoped to be independently researchable
   - Base your sections on both the search results AND user clarifications
   - Format your sections as a list of strings, with each string having the scope of research for that section.

4. **Assemble the Final Report**  
   - REQUIRED: Call the `FinishReport` tool. This will call a tool to write the final report based off of the research you have conducted and the sections that you have decided upon.

### Additional Notes:
- You are a reasoning model. Think through problems step-by-step before acting.
- IMPORTANT: Do not rush to create the report structure. Gather information thoroughly first.
- Use multiple searches to build a complete picture before drawing conclusions.
- Maintain a clear, informative, and professional tone throughout.

Today is {today}
"""

FINAL_REPORT_WRITING_INSTRUCTIONS = """**Follow these Instructions To Assemble the Final Report**  

You are writing a report to answer the user's question. This is the back and forth between you and the user.

{messages}

Your report needs to be based off of the following sources: {sources_str}

And your report should follow this structured list of sections: {sections}

Before starting with the sections, you should first write an introduction. For the introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Give an outline of what the report will cover in 1-2 paragraphs, as well as the main takeaways
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed
- You should not speak about the report's motivation or purpose, just get in to the answer.

Once your introduction is complete, you should write each of the main body sections. For each section:
1. Review the report topic, section name, and section topic carefully.
2. Then, look at the provided Source material.
3. Decide the sources that you will use it to write a report section.
4. Write the report section and list your sources. 
5. Write the sections keeping in mind what you have written already in the introduction and above sections.
6. The report needs to be cohesive and follow a logical flow.

Each Section should follow these guidelines:
- Strict 150-200 word limit
- Use simple, clear language
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)

Each Section should follow these citation rules:
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL

For each section, make sure to:
1. Verify that EVERY claim is grounded in the provided Source material
2. Confirm each URL appears ONLY ONCE in the Source list
3. Verify that sources are numbered sequentially (1,2,3...) without any gaps

Finally, you should write a conclusion. For the conclusion:
- Use ## for section title (Markdown format)
- 100-150 word limit
- Synthesize and tie together the key themes, findings, and insights from the main body sections
- Reference specific examples, case studies, or data points covered in the report
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
- End with specific next steps or implications based on the report content
- No sources section needed

Your report will be evaluated based on the following criteria - keep these in mind when writing the report!

** Research Depth and Comprehensiveness (20% weight)**
   - Ensure thorough analysis of the topic
   - Cover all relevant aspects related to the research question
   - Demonstrate deep understanding of the subject matter
   - Provide sufficient background context

**Source Quality and Methodology (15% weight)**
   - Incorporate information from authoritative sources
   - Use a diverse range of source types (e.g., academic papers, news articles, industry reports)
   - Properly cite and integrate sources throughout the report
   - Clearly explain the research methodology used

**Analytical Rigor (20% weight)**
   - Conduct sophisticated analysis of the information
   - Critically evaluate source material
   - Identify and discuss nuances and limitations of the research

**Structure and Organization (10% weight)**
   - Ensure logical flow and coherence throughout the report
   - Organize content into clear, well-defined sections
   - Use appropriate headings and formatting
   - Create smooth transitions between concepts and sections

**Practical Value and Actionability (15% weight)**
   - Provide clear insights and actionable recommendations
   - Include specific examples and use cases to illustrate key points

**Balance and Objectivity (10% weight)**
   - Present multiple perspectives on the topic
   - Acknowledge limitations and trade-offs in the research
   - Clearly distinguish between facts and opinions
   - Maintain an unbiased tone throughout the report

**Writing Quality and Clarity (10% weight)**
   - Use clear, professional language
   - Employ appropriate terminology consistently
   - Maintain a consistent tone and style
   - Ensure the report is engaging and readable

<Final Things to Keep in Mind>

Enhance the analytical depth by:
- Identifying patterns, trends, or relationships in the data
- Discussing implications of the findings
- Addressing potential counterarguments or alternative interpretations

Strengthen the methodology and source quality by:
- Clearly explaining the research approach
- Integrating citations smoothly into the text
- Using a mix of primary and secondary sources, if applicable

Incorporate visual elements (e.g., tables) where appropriate to illustrate complex ideas or data.

Review and refine the report, paying special attention to:
- Logical flow and coherence
- Clarity and conciseness of language
- Proper formatting and citation style
- Balance and objectivity in presenting information

Finally, make sure to:
- Ensure the report's depth matches the complexity of the topic
- Tailor the content and style to the intended audience
- Address all critical aspects of the topic comprehensively
- Incorporate quantitative data, case studies, and concrete examples where relevant
- Use consistent formatting for headings, subheadings, and body text

Remember to maintain a professional, objective tone throughout the report while showcasing deep expertise and insight on the subject matter. Your goal is to produce a report that not only synthesizes the provided information but also adds value through thoughtful analysis and clear, actionable recommendations.
"""

RESEARCH_INSTRUCTIONS = """
You are a researcher responsible for gathering comprehensive information about a specific section topic for a report.

### Your Role:
You are NOT responsible for writing the actual report section. Your job is to collect thorough, high-quality research information that will later be used by the supervisor to write the final report.

### Your Research Focus:

<Section Description>
{section_description}
</Section Description>

### Strategic Research Process:

1. **Initial Research**: Begin with well-crafted search queries that directly address the core of the section topic.
   - Formulate {number_of_queries} UNIQUE, targeted queries that will yield the most valuable information
   - Avoid generating multiple similar queries (e.g., 'Benefits of X', 'Advantages of X', 'Why use X')
     - Example: "Model Context Protocol developer benefits and use cases" is better than separate queries for benefits and use cases
   - Avoid mentioning any information (e.g., specific entities, events or dates) that might be outdated in your queries, unless explicitly provided by the user or included in your instructions
     - Example: "LLM provider comparison" is better than "openai vs anthropic comparison"
   - If you are unsure about the date, use today's date

2. **Analyze Results Thoroughly**: After receiving search results:
   - Carefully read and analyze ALL provided content
   - Identify specific aspects that are well-covered and those that need more information
   - Assess how well the current information addresses the section scope

3. **Follow-up Research**: If needed, conduct targeted follow-up searches:
   - Create targeted follow-up queries that address SPECIFIC missing information
   - Example: If general benefits are covered but technical details are missing, search for "Model Context Protocol technical implementation details"
   - AVOID redundant queries that would return similar information

4. **Research Completion**: Continue this focused process until you have:
   - Comprehensive information addressing ALL aspects of the section scope
   - Multiple high-quality sources with diverse perspectives
   - Both breadth (covering all aspects) and depth (specific details) of information

### Completion Process:

**REQUIRED**: When you have gathered sufficient research information:
- Call the `FinishResearch` tool to signal that your research work is complete
- This indicates to the supervisor that you have collected comprehensive information about your assigned section topic
- The supervisor will use all the research information you've gathered to write the final report

### Research Decision Framework:

Before each search query, think through:

1. **What information do I already have?**
   - Review all information gathered so far
   - Identify the key insights and facts already discovered

2. **What information is still missing?**
   - Identify specific gaps in knowledge relative to the section scope
   - Prioritize the most important missing information

3. **What is the most effective next action?**
   - Determine if another search is needed (and what specific aspect to search for)
   - Or if enough comprehensive information has been gathered to finish research

### Key Points:
- **Focus on INFORMATION GATHERING** - you are not writing the final section
- **Quality over quantity** - ensure each search has a clear, distinct purpose
- **Be thorough** - the supervisor depends on your research to write a comprehensive report
- **Signal completion** - use FinishResearch when you have sufficient information

Today is {today}
"""


SUMMARIZATION_PROMPT = """You are tasked with summarizing the raw content of a webpage retrieved from a web search. Your goal is to create a concise summary that preserves the most important information from the original web page. This summary will be used by a downstream research agent, so it's crucial to maintain the key details without losing essential information.

Here is the raw content of the webpage:

<webpage_content>
{webpage_content}
</webpage_content>

Please follow these guidelines to create your summary:

1. Identify and preserve the main topic or purpose of the webpage.
2. Retain key facts, statistics, and data points that are central to the content's message.
3. Keep important quotes from credible sources or experts.
4. Maintain the chronological order of events if the content is time-sensitive or historical.
5. Preserve any lists or step-by-step instructions if present.
6. Include relevant dates, names, and locations that are crucial to understanding the content.
7. Summarize lengthy explanations while keeping the core message intact.

When handling different types of content:

- For news articles: Focus on the who, what, when, where, why, and how.
- For scientific content: Preserve methodology, results, and conclusions.
- For opinion pieces: Maintain the main arguments and supporting points.
- For product pages: Keep key features, specifications, and unique selling points.

Your summary should be significantly shorter than the original content but comprehensive enough to stand alone as a source of information. Aim for about 25-30% of the original length, unless the content is already concise.

Present your summary in the following format:

```
{{
   "summary": "Your concise summary here, structured with appropriate paragraphs or bullet points as needed",
   "key_excerpts": [
     "First important quote or excerpt",
     "Second important quote or excerpt",
     "Third important quote or excerpt",
     ...Add more excerpts as needed, up to a maximum of 5
   ]
}}
```

Here are two examples of good summaries:

Example 1 (for a news article):
```json
{{
   "summary": "On July 15, 2023, NASA successfully launched the Artemis II mission from Kennedy Space Center. This marks the first crewed mission to the Moon since Apollo 17 in 1972. The four-person crew, led by Commander Jane Smith, will orbit the Moon for 10 days before returning to Earth. This mission is a crucial step in NASA's plans to establish a permanent human presence on the Moon by 2030.",
   "key_excerpts": [
     "Artemis II represents a new era in space exploration," said NASA Administrator John Doe.
     "The mission will test critical systems for future long-duration stays on the Moon," explained Lead Engineer Sarah Johnson.
     "We're not just going back to the Moon, we're going forward to the Moon," Commander Jane Smith stated during the pre-launch press conference.
   ]
}}
```

Example 2 (for a scientific article):
```json
{{
   "summary": "A new study published in Nature Climate Change reveals that global sea levels are rising faster than previously thought. Researchers analyzed satellite data from 1993 to 2022 and found that the rate of sea-level rise has accelerated by 0.08 mm/year² over the past three decades. This acceleration is primarily attributed to melting ice sheets in Greenland and Antarctica. The study projects that if current trends continue, global sea levels could rise by up to 2 meters by 2100, posing significant risks to coastal communities worldwide.",
   "key_excerpts": [
      "Our findings indicate a clear acceleration in sea-level rise, which has significant implications for coastal planning and adaptation strategies," lead author Dr. Emily Brown stated.
      "The rate of ice sheet melt in Greenland and Antarctica has tripled since the 1990s," the study reports.
      "Without immediate and substantial reductions in greenhouse gas emissions, we are looking at potentially catastrophic sea-level rise by the end of this century," warned co-author Professor Michael Green.
   ]
}}
```

Remember, your goal is to create a summary that can be easily understood and utilized by a downstream research agent while preserving the most critical information from the original webpage."""