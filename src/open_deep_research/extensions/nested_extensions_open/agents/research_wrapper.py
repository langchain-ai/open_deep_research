"""Research agent wrapper for open_deep_research.

This wraps the existing deep_researcher WITHOUT modifying it.
Uses Pydantic structured output for reliable source extraction.
"""
import asyncio
import os
from typing import Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from open_deep_research.deep_researcher import deep_researcher
from extensions.utils.llm_factory import get_extensions_llm


class SourceExtraction(BaseModel):
    """Structured output for source extraction."""
    sources: List[str] = Field(description="List of source URLs mentioned in the text")
    source_count: int = Field(description="Total number of sources found")


class ResearchAgent:
    """Wrapper around open_deep_research's deep_researcher.

    Provides a simple interface to the research agent without modifying
    the original code. Uses Pydantic structured output for source extraction.
    """

    def __init__(self, provider: str = None, model: str = None):
        """Initialize research agent wrapper.

        Args:
            provider: LLM provider override. Falls back to env vars.
            model: LLM model override. Falls back to env vars.
        """
        self.name = "research"
        self.researcher = deep_researcher

        # LLM for structured source extraction (provider-agnostic)
        self.llm = get_extensions_llm(provider=provider, model=model, temperature=0.0)

        # Structured LLM for source extraction
        try:
            self.structured_llm = self.llm.with_structured_output(SourceExtraction)
        except Exception:
            self.structured_llm = None

    async def run_async(self, query: str) -> Dict[str, Any]:
        """Run deep research asynchronously.

        Args:
            query: Research question/topic

        Returns:
            Dictionary with research results
        """
        start_time = datetime.now()

        try:
            # Call open_deep_research's deep_researcher
            # Disable clarification since the query is already formulated by the MasterAgent
            config = {
                "configurable": {
                    "allow_clarification": False,
                }
            }
            result = await self.researcher.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config=config,
            )

            # Extract results
            final_report = result.get('final_report', '')
            notes = result.get('notes', [])
            raw_notes = result.get('raw_notes', [])

            # Extract sources using Pydantic structured output
            sources = await self._extract_sources_structured(final_report, notes, raw_notes)

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "agent_name": self.name,
                "status": "completed",
                "output": final_report,
                "sources": sources,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "error": None
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "agent_name": self.name,
                "status": "error",
                "output": "",
                "sources": [],
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def run(self, query: str) -> Dict[str, Any]:
        """Run deep research synchronously (wrapper for async).

        Args:
            query: Research question/topic

        Returns:
            Dictionary with research results
        """
        return asyncio.run(self.run_async(query))

    async def _extract_sources_structured(
        self,
        final_report: str,
        notes: List[str],
        raw_notes: List[str]
    ) -> List[str]:
        """Extract source URLs using Pydantic structured output.

        Args:
            final_report: Final research report
            notes: Research notes
            raw_notes: Raw research notes

        Returns:
            List of unique source URLs
        """
        # Combine all text for source extraction
        all_text = final_report + "\n\n" + "\n".join(notes) + "\n\n" + "\n".join(raw_notes)

        # Truncate if too long (keep first 10000 chars)
        if len(all_text) > 10000:
            all_text = all_text[:10000]

        prompt = f"""Extract all source URLs from the following research content.
Look for URLs that start with http:// or https://.

Research Content:
{all_text}

Extract all unique source URLs."""

        try:
            if self.structured_llm:
                result = await self.structured_llm.ainvoke(prompt)
                return list(set(result.sources))
            else:
                return self._extract_sources_regex(all_text)
        except Exception as e:
            print(f"Structured extraction failed: {e}. Using regex fallback.")
            return self._extract_sources_regex(all_text)

    def _extract_sources_regex(self, text: str) -> List[str]:
        """Fallback: Extract URLs using regex.

        Args:
            text: Text to extract URLs from

        Returns:
            List of unique URLs
        """
        import re

        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)

        cleaned_urls = []
        for url in urls:
            url = url.rstrip('.,;:!?)')
            cleaned_urls.append(url)

        return list(set(cleaned_urls))


__all__ = ['ResearchAgent']
