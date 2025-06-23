clarify_with_user_instructions="""
These are the messages that have been exchanged so far from the user asking for the report:
<Messages>
{messages}
</Messages>

Return ONE targeted question to clarify report scope.
If there are acronyms, abbreviations, or unknown terms, ask the user to clarify.
Focus on: technical depth, target audience, specific aspects to emphasize
Examples: "Should I focus on technical implementation details or high-level business benefits?" 
"""

initial_upfront_model_provider_web_search_system_prompt = """You are a research assistant conducting deep research. Use the tools provided to thoroughly research what the user is asking about.

You can use general websearch, or any other tool provided to you. Select the BEST tool for the job, this may or may not be general websearch.
{mcp_prompt}

Focus on:
- Finding authoritative, comprehensive sources
- Covering multiple perspectives and aspects of the topic
- Using specific, targeted search queries
- Gathering sufficient information for a detailed report

Use multiple searches as needed to cover the topic thoroughly."""

follow_up_upfront_model_provider_web_search_system_prompt = """You are continuing deep research based on previous findings. Use the tools provided to fill knowledge gaps identified in the reflection.

You can use general websearch, or any other tool provided to you. Select the BEST tool for the job, this may or may not be general websearch.
{mcp_prompt}

Focus on areas that need deeper investigation. Use targeted searches to find missing information and perspectives.
"""

upfront_model_provider_reflection_system_prompt = """Analyze these research findings for completeness and depth with respect to the original messages asking about the topic:

Original Messages asking about the Topic: {messages}

Current Research Findings:
{findings}

Make sure to also reason about the quality of the research and the sources that were found. We want to feel confident that the sources are high quality, and only want to give the user
information that we are confident in. Keep that in mind while evaluating the current state of the research.

Please evaluate:
1. Are the findings comprehensive enough for a detailed report to answer the question(s) in the original messages?
2. What key knowledge gaps or missing perspectives exist?
3. What specific areas need deeper investigation?
4. What follow-up research queries would be most valuable?

Respond in valid JSON format with these exact keys:
"is_satisfied": boolean,
"knowledge_gaps": ["gap1", "gap2", ...],
"suggested_queries": ["query1", "query2", ...],
"reasoning": "detailed explanation of the analysis"
"""


gap_context_prompt = """Knowledge gaps to address:
{knowledge_gaps}

Suggested focus areas:
{focus_areas}

Reasoning: {reasoning}

You only need to output net new information that we have not already gathered. This will get appended to the existing information.
"""

response_structure_instructions="""You are performing deep research to comprehensively answer a question that a user asked.

These are the messages that have been exchanged so far between yourself and the user:
<Messages>
{messages}
</Messages>

We have already gathered a lot of information from web searches about the topic. Here is a all of the information that we've gathered so far:
<Information gathered already>
{context}
</Information gathered already>


<Task>
Generate a list of sections for the answer to provide to the user. Your plan should be tight and focused with NO overlapping sections or unnecessary filler.
You can generate however many sections you are necessary, as these will be computed in parallel. If a user asks for a list of 10 items, you could generate 10 sections, for example.

The structure of your sections will really depend on the question that the user asked. 
This structure is important because you will be writing each section independently, so they should not have any dependence on each other.

You can structure your sections in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your sections like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things

Or, you could choose to make each item in the list a separate section. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your sections like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your sections however you think is best, including in ways that are not listed above!

Each section should have the fields:

- Name - Name for this section of the report.
- Description - A specific overview of what the section will be about. Do not be vague or general.

Integration guidelines:
- Ensure each section has a distinct purpose with no content overlap
- Combine related concepts rather than separating them
- CRITICAL: Every section MUST be directly relevant to the main topic
- Avoid tangential or loosely related sections that don't directly address the core topic

Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow that answers the user's question in the best way possible.
</Task>

Return your outline in the following format as a list of sections!
<Format>
Call the Outline tool 
</Format>
"""

initial_section_write_instructions = """Write a first draft of this section based on the research that we have done so far.
<Task>
1. Review the messages describing the report request, the section name, and the section topic carefully.
2. Then, look at the provided information gathered from web searches. Not all of this information will be relevant to this section. Be particular about what information you use.
3. Write the report section. 
</Task>

<Writing Guidelines>
- Strict 150-200 word limit
- Use simple, clear language
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
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
4. DO NOT include any commentary from yourself in the section. Just write the section. Do not say "I'm writing a section" or "I'll revise this section" or anything like that. Just include the report itself.
</Final Check>
"""


initial_section_write_inputs=""" 
These are the messages that have been exchanged so far from the user asking for the report:
<Messages>
{messages}
</Messages>

This is the section name:
<Section name>
{section_name}
</Section name>

This is the section description:
<Section description>
{section_description}
</Section description>

This is the information that you have gathered from web searches so far:
<Information gathered from web searches>
{context}
</Information gathered from web searches>
"""

follow_up_section_write_instructions = """Revise the section based on the feedback and the information gathered from web searches.

<Task>
1. Review the messages describing the report request, section name, and section topic carefully.
2. Review the existing section, as well as the feedback gathered to improve on the section.
3. Then, look at the new provided information gathered from web searches.
4. Decide the sources that you will use it to revise the section.
5. Revise the section and list your sources. 
</Task>

<Writing Guidelines>
- Synthesize the existing section with the new information gathered from web searches, according to the feedback.
- Strict 150-200 word limit
- Use simple, clear language
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
</Writing Guidelines>

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Keep the existing sources in the section untouched, you can change the numbers if necessary.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
</Citation Rules>

<Final Check>
1. Verify that EVERY claim is grounded in the provided Source material
2. Confirm each URL appears ONLY ONCE in the Source list
3. Verify that sources are numbered sequentially (1,2,3...) without any gaps
4. DO NOT include any commentary from yourself in the section. Just write the section. Do not say "I'm writing a section" or "I'll revise this section" or anything like that. Just include the report itself.
</Final Check>
"""

follow_up_section_write_inputs=""" 
These are the messages that have been exchanged so far from the user asking for the report:
<Messages>
{messages}
</Messages>

This is the section name:
<Section name>
{section_name}
</Section name>

This is the section description:
<Section description>
{section_description}
</Section description>

This is the existing section content:
<Existing section content>
{section_content}
</Existing section content>

This is the feedback on the section:
<Feedback on the section>
{feedback}
</Feedback on the section>

This is the information that you have gathered from web searches so that you can address that feedback:
<Information gathered from web searches>
{context}
</Information gathered from web searches>
"""

section_grader_instructions = """Review a section relative to the section description and user's messages:

These are the messages that have been exchanged so far from the user asking for the report:
<Messages>
{messages}
</Messages>

This is the section description:
<section description>
{section_description}
</section description>

This is the section content:
<section content>
{section_content}
</section content>

<task>
Evaluate whether the section content adequately addresses the section description.

If the section content does not adequately address the section description, generate {number_of_follow_up_queries} follow-up search queries to gather missing information.
</task>
"""

final_report_generation_prompt = """Based on all the research conducted, create a comprehensive, well-structured answer to the question(s) in these original messages: {messages}

Here are the findings from the research that you conducted:
{findings}

These are the additional sources that you found during research (reference these as appropriate as you write your response):
{source_list}

We iterated on a section outline for the answer to the question. Here is the outline, follow this as best as you can:
{outline}

Please create a detailed answer to the question(s) in the original messages that:
1. Follows the section outline as closely as possible
2. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
3. Includes specific facts and insights from the research
4. References relevant sources using [Title](URL) format
5. Provides a balanced, thorough analysis
6. Includes a "Sources" section at the end with all referenced links

For each section, do the following:
- Strict 150-200 word limit
- Use simple, clear language
- Use short paragraphs (2-3 sentences max)
- Use ## for section title (Markdown format)
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
</Citation Rules>"""


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