"""Utility search and scraping tools."""

import os
from typing import List

from crawl4ai import AsyncWebCrawler
from langchain_core.documents import Document
from tavily import TavilyClient

# Instantiate Tavily client using environment variable
_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def tavily_search(query: str) -> List[Document]:
    """Search the web using Tavily API."""
    return _tavily_client.search(query, count=5)


async def crawl_url(url: str) -> str:
    """Return the markdown from crawling a URL."""
    async with AsyncWebCrawler() as crawler:
        return (await crawler.arun(url=url)).markdown


__all__ = ["tavily_search", "crawl_url"]
