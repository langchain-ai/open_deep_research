"""
Research Enhancement Layer - Improves research quality without modifying core code.

This module sits BETWEEN the user query and the base deep_researcher.
It enhances queries, optimizes configuration, and synthesizes results better.

Strategy:
1. Query Enhancement - Break complex queries into multiple focused sub-queries
2. Smart Configuration - Optimize settings based on query type
3. Result Synthesis - Combine and deduplicate results intelligently
4. Source Enrichment - Extract more value from existing sources

NO changes to open_deep_research core code required!
"""
import asyncio
import re
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage

from open_deep_research.deep_researcher import deep_researcher
from extensions.utils.llm_factory import get_extensions_llm


class ResearchEnhancementLayer:
    """Enhancement layer that makes research more comprehensive.

    Works by:
    - Breaking queries into focused sub-queries
    - Running multiple research passes
    - Combining and deduplicating results
    - Enriching source extraction

    No modifications to base researcher needed!
    """

    def __init__(self, provider: str = None, model: str = None):
        """Initialize enhancement layer.

        Args:
            provider: LLM provider override. Falls back to env vars.
            model: LLM model override. Falls back to env vars.
        """
        self.base_researcher = deep_researcher

        # LLM for query enhancement and synthesis (provider-agnostic)
        self.llm = get_extensions_llm(provider=provider, model=model, temperature=0.3)

    async def research_enhanced(self, query: str) -> Dict[str, Any]:
        """
        Enhanced research that runs multiple passes for comprehensiveness.

        Strategy:
        1. Analyze the query
        2. Generate complementary sub-queries
        3. Run research for each sub-query
        4. Combine and synthesize all results

        Args:
            query: User's research question

        Returns:
            Enhanced research results with more sources and depth
        """
        start_time = datetime.now()

        print(f"\n[EnhancedResearch] Starting...")
        print(f"Original query: {query}\n")

        # Step 1: Generate complementary sub-queries
        sub_queries = await self._generate_sub_queries(query)
        print(f"[EnhancedResearch] Generated {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"   {i}. {sq}")
        print()

        # Step 2: Run research for each sub-query
        all_results = []
        for i, sub_query in enumerate(sub_queries, 1):
            print(f"[EnhancedResearch] Researching sub-query {i}/{len(sub_queries)}: {sub_query[:60]}...")

            enhanced_sq = self._add_research_instructions(sub_query)

            config = {
                "configurable": {
                    "allow_clarification": False,
                }
            }
            result = await self.base_researcher.ainvoke(
                {"messages": [HumanMessage(content=enhanced_sq)]},
                config=config,
            )

            all_results.append({
                'query': sub_query,
                'report': result.get('final_report', ''),
                'notes': result.get('notes', []),
                'raw_notes': result.get('raw_notes', [])
            })

            print(f"   Completed ({len(result.get('final_report', ''))} chars)")

        # Step 3: Synthesize all results
        print(f"\n[EnhancedResearch] Synthesizing {len(all_results)} research results...")
        synthesized = await self._synthesize_results(query, all_results)

        # Step 4: Extract all sources
        sources = self._extract_all_sources(all_results)

        execution_time = (datetime.now() - start_time).total_seconds()

        print(f"\n[EnhancedResearch] Complete!")
        print(f"   Total sources: {len(sources)}")
        print(f"   Report length: {len(synthesized)} chars")
        print(f"   Execution time: {execution_time:.1f}s\n")

        return {
            "agent_name": "enhanced_research",
            "status": "completed",
            "output": synthesized,
            "sources": sources,
            "sub_queries_used": sub_queries,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }

    async def _generate_sub_queries(self, query: str) -> List[str]:
        """
        Generate complementary sub-queries to cover topic comprehensively.

        Args:
            query: Original user query

        Returns:
            List of 3-5 focused sub-queries
        """
        prompt = f"""Given this research query: "{query}"

Generate 4 complementary sub-queries that will provide comprehensive coverage:

1. CURRENT STATE - Focus on recent developments, latest information, current status
2. BACKGROUND - Historical context, foundational concepts, how we got here
3. ANALYSIS - Expert opinions, critical analysis, different perspectives
4. DATA - Statistics, metrics, quantitative information, evidence

Each sub-query should:
- Be specific and focused
- Complement the others (no overlap)
- Lead to actionable research results
- Be 1-2 sentences max

Format your response as:
1. [sub-query 1]
2. [sub-query 2]
3. [sub-query 3]
4. [sub-query 4]

Only output the numbered list, nothing else."""

        response = await self.llm.ainvoke(prompt)
        content = response.content

        lines = content.strip().split('\n')
        sub_queries = []

        for line in lines:
            match = re.match(r'^\d+[\.\)]\s*(.+)$', line.strip())
            if match:
                sub_queries.append(match.group(1).strip())

        if len(sub_queries) < 3:
            sub_queries = [
                query,
                f"{query} - recent developments and current status",
                f"{query} - historical background and context",
                f"{query} - expert analysis and perspectives",
                f"{query} - statistics, data, and metrics"
            ]

        return sub_queries[:5]

    def _add_research_instructions(self, query: str) -> str:
        """
        Add research quality instructions to query.

        Args:
            query: Sub-query to enhance

        Returns:
            Enhanced query with instructions
        """
        enhanced = f"""{query}

Please provide comprehensive research with:
- Multiple credible sources (aim for 5+ sources)
- Specific facts, data, and statistics when available
- Present numerical data in markdown tables where possible (e.g., | Name | Value | Unit |)
- Recent information (prioritize last 6-12 months)
- Clear citation of sources
- Well-structured analysis

Focus on quality and depth."""

        return enhanced

    async def _synthesize_results(
        self,
        original_query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize multiple research results into one comprehensive report.

        Args:
            original_query: Original user query
            results: List of research results from sub-queries

        Returns:
            Synthesized comprehensive report
        """
        all_reports = [r['report'] for r in results if r['report']]

        if not all_reports:
            return "No research results found."

        if len(all_reports) == 1:
            return all_reports[0]

        combined_text = "\n\n---\n\n".join([
            f"Research on: {results[i]['query']}\n\n{report}"
            for i, report in enumerate(all_reports)
        ])

        if len(combined_text) > 20000:
            combined_text = combined_text[:20000] + "\n\n[Content truncated for synthesis]"

        synthesis_prompt = f"""You are synthesizing multiple research results into one comprehensive report.

ORIGINAL QUERY: {original_query}

RESEARCH RESULTS FROM MULTIPLE ANGLES:
{combined_text}

Please create a comprehensive, well-structured report that:
1. Combines all unique information from the research results
2. Organizes findings logically (current state -> background -> analysis -> data)
3. Removes duplicate information
4. Preserves all source citations and URLs
5. Maintains factual accuracy
6. Provides a complete answer to the original query
7. Includes a "## Key Data & Statistics" section near the end with ALL numerical data formatted as markdown tables
8. Preserves any data tables from the individual research results -- do not convert tables back into prose

Format:
- Use clear sections/headings
- Include all relevant statistics and data
- Cite sources inline
- Present numerical data in markdown tables where possible (e.g., | Name | Value | Unit |)
- 1500-2500 words target length

Synthesized Report:"""

        synthesis_response = await self.llm.ainvoke(synthesis_prompt)
        synthesized_report = synthesis_response.content

        return synthesized_report

    def _extract_all_sources(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract all unique sources from multiple research results.

        Args:
            results: List of research results

        Returns:
            Deduplicated list of source URLs
        """
        all_sources = set()

        for result in results:
            texts = [
                result.get('report', ''),
                *result.get('notes', []),
                *result.get('raw_notes', [])
            ]

            combined_text = '\n'.join(texts)

            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, combined_text)

            for url in urls:
                cleaned = url.rstrip('.,;:!?)')
                all_sources.add(cleaned)

        return sorted(list(all_sources))

    def run(self, query: str) -> Dict[str, Any]:
        """Sync wrapper for enhanced research."""
        return asyncio.run(self.research_enhanced(query))


# ============================================================================
# Simple wrapper to integrate with existing ResearchAgent interface
# ============================================================================

class EnhancedResearchWrapper:
    """
    Drop-in replacement for ResearchAgent that uses enhancement layer.

    Usage:
        research_agent = EnhancedResearchWrapper()
        result = await research_agent.run_async(query)
    """

    def __init__(self, provider: str = None, model: str = None):
        """Initialize with enhancement layer.

        Args:
            provider: LLM provider override. Falls back to env vars.
            model: LLM model override. Falls back to env vars.
        """
        self.name = "research"
        self.enhancer = ResearchEnhancementLayer(provider=provider, model=model)

    async def run_async(self, query: str) -> Dict[str, Any]:
        """Run enhanced research (async)."""
        return await self.enhancer.research_enhanced(query)

    def run(self, query: str) -> Dict[str, Any]:
        """Run enhanced research (sync)."""
        return self.enhancer.run(query)


__all__ = ['ResearchEnhancementLayer', 'EnhancedResearchWrapper']
