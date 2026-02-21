import os
import requests
import json
import logging
from typing import Dict, Any, List, Optional
from langsmith import traceable
from tavily import TavilyClient
import urllib.parse
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
import datetime
from difflib import SequenceMatcher
import time
import re
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("general_deep_search")


def log_tavily_status(search_type, query, results):
    """Log clear Tavily health status after each search call.

    Produces one of three log lines:
      INFO  [TAVILY STATUS] OK     — Tavily returned real results
      WARN  [TAVILY STATUS] FAILED — Tavily errored (API key, rate limit, etc.)
      WARN  [TAVILY STATUS] EMPTY  — 0 results; LLM may fill from its own knowledge
    """
    result_count = len(results.get("results", [])) if isinstance(results, dict) else 0
    has_error = results.get("error") if isinstance(results, dict) else None
    short_query = query[:80] if query else "(empty)"

    if result_count > 0:
        urls = [r.get("url", "N/A")[:80] for r in results["results"][:3]]
        logger.info(
            f"[TAVILY STATUS] OK | {search_type} | {result_count} results "
            f"| query: '{short_query}' | top URLs: {urls}"
        )
    elif has_error:
        logger.warning(
            f"[TAVILY STATUS] FAILED | {search_type} | 0 results "
            f"| query: '{short_query}' | error: {has_error}"
        )
    else:
        logger.warning(
            f"[TAVILY STATUS] EMPTY | {search_type} | 0 results "
            f"| query: '{short_query}' "
            f"| Tavily returned nothing - LLM may generate from its own knowledge!"
        )


def log_general_deep_search_params(**params):
    """Log the parameters being sent to Tavily search API

    This helps with debugging and understanding how different search types
    are configured.

    Args:
        **params: Keyword arguments being passed to tavily_client.search
    """
    # Create a formatted string of the parameters
    formatted_params = json.dumps(params, indent=2)
    logger.info(f"Tavily search parameters:\n{formatted_params}")

    # Print to console as well
    print(f"\nTavily API Call Parameters:\n{formatted_params}\n")

    return params


def deduplicate_and_format_sources(
    search_response, max_tokens_per_source, include_raw_content=False
):
    """
    Takes either a single search response or list of responses from search APIs and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError(
            "Input must be either a dict with 'results' or a list of search results"
        )

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += (
            f"Most relevant content from source: {source['content']}\n===\n"
        )
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def format_sources(search_results):
    """Format search results into a bullet-point list of sources.

    Args:
        search_results (dict): Tavily search response containing results

    Returns:
        str: Formatted string with sources and their URLs
    """
    return "\n".join(
        f"* {source['title']} : {source['url']}" for source in search_results["results"]
    )


def deduplicate_sources_list(sources_list):
    """Deduplicate a list of formatted sources.

    Args:
        sources_list (list): List of formatted source strings

    Returns:
        list: Deduplicated list of source strings
    """
    # Each source is formatted as "* Title : URL"
    # Extract URLs for deduplication
    unique_sources = {}
    for source in sources_list:
        if source and ":" in source:
            # Split at the last colon to get the URL part
            url = source.split(":", 1)[1].strip()
            unique_sources[url] = source

    # Return the deduplicated list
    return list(unique_sources.values())


@traceable
@retry(
    wait=wait_exponential(
        multiplier=1, min=2, max=60
    ),  # Start with 2s, then 4s, 8s, 16s, 32s, 60s
    stop=stop_after_attempt(5),  # Try 5 times maximum
    retry=retry_if_exception_type(
        requests.exceptions.HTTPError
    ),  # Only retry on HTTPError
)
def general_deep_search(query, include_raw_content=True, top_k=3, config=None):
    """General web search using Tavily

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        top_k (int): Maximum number of results to return after filtering (default: 3)
        config (RunnableConfig, optional): Configuration object for LangSmith tracing
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available
                - score (float): Relevance score of the result

            Note: Results will be filtered to only include items with score >= threshold,
            deduplicated by title and URL, and limited to the top_k highest scoring results.
            If no results meet the criteria, an empty results list will be returned."""

    threshold = 0.5
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    # Validate query length - Tavily has a 400 character limit
    if len(query) > 400:
        print(
            f"WARNING: Query exceeds Tavily's 400 character limit. Query length: {len(query)}"
        )
        # Truncate query to 397 characters + "..."
        query = query[:397] + "..."
        print(f"Truncated query: {query}")

    # Validate query isn't empty after stripping whitespace
    query = query.strip()
    if not query:
        print("ERROR: Empty query after stripping whitespace")
        return {"results": [], "search_string": "", "response_time": 0}

    tavily_client = TavilyClient(api_key=api_key)
    try:
        # Log the parameters
        search_params = log_general_deep_search_params(
            query=query,
            max_results=20,
            include_answer="basic",
            include_raw_content=include_raw_content,
        )

        # Get the raw search results
        search_results = tavily_client.search(**search_params)

        # Filter results by score and get the top_k
        if "results" in search_results:
            # Filter out results with score < threshold
            filtered_results = [
                r for r in search_results["results"] if r.get("score", 0) >= threshold
            ]

            # Deduplicate results by title and URL
            seen_titles = set()
            seen_urls = set()
            deduplicated_results = []

            for result in filtered_results:
                title = result.get("title", "").strip()
                url = result.get("url", "").strip()

                # Skip this result if we've seen this title or URL before
                if title in seen_titles or url in seen_urls:
                    continue

                # Add to deduplicated results and mark title and URL as seen
                deduplicated_results.append(result)
                seen_titles.add(title)
                seen_urls.add(url)

            # Sort by score in descending order and take top_k
            deduplicated_results = sorted(
                deduplicated_results, key=lambda x: x.get("score", 0), reverse=True
            )[:top_k]

            # Update the results in the response
            search_results["results"] = deduplicated_results

            print(
                f"Filtered Tavily results: {len(deduplicated_results)} out of {len(search_results.get('results', []))} meet the score threshold of {threshold} and are unique by title/URL."
            )

        log_tavily_status("GENERAL", query, search_results)
        return search_results

    except requests.exceptions.HTTPError as e:
        # Handle specific error codes
        if e.response.status_code == 422:
            print(f"Tavily API validation error (422): {e.response.text}")
            print("This is likely due to invalid parameters. Returning empty results.")
            # Return a valid empty response instead of retrying
            error_response = {
                "results": [],
                "error": f"Tavily API validation error: {str(e)}",
                "search_string": query,
                "response_time": 0,
            }
            log_tavily_status("GENERAL", query, error_response)
            return error_response
        elif e.response.status_code == 401:
            print(f"Tavily API authentication error (401): Check your API key")
            # Return empty results for auth errors
            error_response = {
                "results": [],
                "error": "Tavily API authentication error: Invalid API key",
                "search_string": query,
                "response_time": 0,
            }
            log_tavily_status("GENERAL", query, error_response)
            return error_response
        else:
            print(f"Tavily API error: {e}. Retrying...")
            raise  # Re-raise to trigger the retry
    except Exception as e:
        print(f"Unexpected error in Tavily search: {str(e)}")
        # Return a valid empty response instead of failing
        error_response = {
            "results": [],
            "error": f"Unexpected error in Tavily search: {str(e)}",
            "search_string": query,
            "response_time": 0,
        }
        log_tavily_status("GENERAL", query, error_response)
        return error_response


@traceable
@retry(
    wait=wait_exponential(
        multiplier=1, min=2, max=60
    ),  # Start with 2s, then 4s, 8s, 16s, 32s, 60s
    stop=stop_after_attempt(5),  # Try 5 times maximum
    retry=retry_if_exception_type(
        requests.exceptions.HTTPError
    ),  # Only retry on HTTPError
)
def linkedin_search(query, include_raw_content=True, top_k=3, min_score=0.1):
    """Search LinkedIn profiles using Tavily with specific constraints

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        top_k (int): Maximum number of high-scoring results to return (default: 3)
        min_score (float): Minimum score threshold for results (default: 0.1)
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries matching the criteria:
                - Only returns top K results sorted by score in descending order
                - Only returns results with scores higher than min_score
                - Only the highest scoring result will contain raw_content
            - response_time (float): Time taken to complete the search
    """

    logger.info(
        f"""
==== LinkedIn Search Configuration ====
- Query: "{query}"
- Domain Restriction: linkedin.com only
- Score threshold: {min_score} (higher than default to ensure quality)
- Raw content handling: Only keeping raw_content for the highest scoring result
- Top results: Limited to {top_k} results after filtering

This function is specifically designed to search LinkedIn profiles and posts
while minimizing hallucinations by:
1. Only including linkedin.com domain in results
2. Using a higher minimum score threshold ({min_score})
3. Only keeping raw_content for the highest scoring result
4. Limiting to top {top_k} results sorted by score
5. Deduplicating results by both title and URL
"""
    )

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    # Validate query length - Tavily has a 400 character limit
    if len(query) > 400:
        print(
            f"WARNING: Query exceeds Tavily's 400 character limit. Query length: {len(query)}"
        )
        # Truncate query to 397 characters + "..."
        query = query[:397] + "..."
        print(f"Truncated query: {query}")

    # Validate query isn't empty after stripping whitespace
    query = query.strip()
    if not query:
        print("ERROR: Empty query after stripping whitespace")
        return {"results": [], "search_string": "", "response_time": 0}

    # Prepare to search specifically on LinkedIn
    include_domains = ["linkedin.com"]

    tavily_client = TavilyClient(api_key=api_key)
    try:
        # Log the parameters
        search_params = log_general_deep_search_params(
            query=query,
            max_results=20,
            include_raw_content=include_raw_content,
            include_domains=include_domains,
            search_type="keyword" if ":" in query else "search",
        )

        # Get the raw search results
        search_results = tavily_client.search(**search_params)

        # Filter results by score and get the top_k
        if "results" in search_results:
            # Filter out results with score < min_score
            filtered_results = [
                r for r in search_results["results"] if r.get("score", 0) >= min_score
            ]

            # Deduplicate results by title and URL
            seen_titles = set()
            seen_urls = set()
            deduplicated_results = []

            for result in filtered_results:
                title = result.get("title", "").strip()
                url = result.get("url", "").strip()

                # Skip this result if we've seen this title or URL before
                if title in seen_titles or url in seen_urls:
                    continue

                # Add to deduplicated results and mark title and URL as seen
                deduplicated_results.append(result)
                seen_titles.add(title)
                seen_urls.add(url)

            # Sort by score in descending order and take top_k
            deduplicated_results = sorted(
                deduplicated_results, key=lambda x: x.get("score", 0), reverse=True
            )[:top_k]

            # Special handling for raw_content - only keep it for the highest scoring result
            if deduplicated_results:
                for i in range(1, len(deduplicated_results)):
                    deduplicated_results[i]["raw_content"] = None

            # Update the results in the response
            search_results["results"] = deduplicated_results

            print(
                f"Filtered LinkedIn results: {len(deduplicated_results)} out of {len(search_results.get('results', []))} meet the score threshold of {min_score} and are unique by title/URL."
            )

        log_tavily_status("LINKEDIN", query, search_results)
        return search_results

    except requests.exceptions.HTTPError as e:
        # Handle specific error codes
        if e.response.status_code == 422:
            print(f"Tavily API validation error (422): {e.response.text}")
            print("This is likely due to invalid parameters. Returning empty results.")
            # Return a valid empty response instead of retrying
            error_response = {
                "results": [],
                "error": f"Tavily API validation error: {str(e)}",
                "search_string": query,
                "response_time": 0,
            }
            log_tavily_status("LINKEDIN", query, error_response)
            return error_response
        elif e.response.status_code == 401:
            print(f"Tavily API authentication error (401): Check your API key")
            # Return empty results for auth errors
            error_response = {
                "results": [],
                "error": "Tavily API authentication error: Invalid API key",
                "search_string": query,
                "response_time": 0,
            }
            log_tavily_status("LINKEDIN", query, error_response)
            return error_response
        else:
            print(f"Tavily API error: {e}. Retrying...")
            raise  # Re-raise to trigger the retry
    except Exception as e:
        print(f"Unexpected error in Tavily search: {str(e)}")
        # Return a valid empty response instead of failing
        error_response = {
            "results": [],
            "error": f"Unexpected error in Tavily search: {str(e)}",
            "search_string": query,
            "response_time": 0,
        }
        log_tavily_status("LINKEDIN", query, error_response)
        return error_response


@traceable
@retry(
    wait=wait_exponential(
        multiplier=1, min=2, max=60
    ),  # Start with 2s, then 4s, 8s, 16s, 32s, 60s
    stop=stop_after_attempt(5),  # Try 5 times maximum
    retry=retry_if_exception_type(
        requests.exceptions.HTTPError
    ),  # Only retry on HTTPError
)
def github_search(query, include_raw_content=True, top_k=5, min_score=0.1):
    """Search GitHub profiles and repositories using Tavily with specific constraints

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        top_k (int): Maximum number of high-scoring results to return (default: 5)
        min_score (float): Minimum score threshold for results (default: 0.6)
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries matching the criteria:
                - Only returns top K results sorted by score in descending order
                - Only returns results with scores higher than min_score
                - Only includes raw_content for results that are likely to contain valuable code or profile information
            - response_time (float): Time taken to complete the search
    """

    logger.info(
        f"""
==== GitHub Search Configuration ====
- Query: "{query}"
- Domain Restriction: github.com only
- Score threshold: {min_score} (ensures quality GitHub results)
- Score boost: 20% for code repositories (/blob/ or /tree/ in URL)
- Raw content handling: Only kept for URLs containing actual code
- Top results: Limited to {top_k} results after filtering and boosting

This function is specifically designed to find GitHub repositories and code by:
1. Only including github.com domain in results 
2. Boosting scores for actual code repositories by 20%
3. Keeping raw_content only for results with actual code content
4. Using repository owner/name for deduplication instead of just URLs
5. Prioritizing repository pages over generic GitHub pages
6. Filtering results with a score threshold of {min_score}
"""
    )

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    # Validate query length - Tavily has a 400 character limit
    if len(query) > 400:
        print(
            f"WARNING: Query exceeds Tavily's 400 character limit. Query length: {len(query)}"
        )
        # Truncate query to 397 characters + "..."
        query = query[:397] + "..."
        print(f"Truncated query: {query}")

    # Validate query isn't empty after stripping whitespace
    query = query.strip()
    if not query:
        print("ERROR: Empty query after stripping whitespace")
        return {"results": [], "search_string": "", "response_time": 0}

    # Focus on GitHub domains
    include_domains = ["github.com"]

    tavily_client = TavilyClient(api_key=api_key)
    try:
        # Log the parameters
        search_params = log_general_deep_search_params(
            query=query,
            max_results=20,
            include_answer="basic",
            include_raw_content=include_raw_content,
            include_domains=include_domains,
            search_type="keyword" if ":" in query else "search",
        )

        # Get the raw search results
        search_results = tavily_client.search(**search_params)

        # Filter results by score and get the top_k
        if "results" in search_results:
            # Filter out results with score < min_score
            filtered_results = [
                r for r in search_results["results"] if r.get("score", 0) >= min_score
            ]

            # Prioritize repository results over other GitHub pages
            for result in filtered_results:
                url = result.get("url", "").lower()
                # Boost score for actual repositories (not just profile pages or other GitHub pages)
                if "/blob/" in url or "/tree/" in url:
                    result["score"] = (
                        result.get("score", 0) * 1.2
                    )  # 20% boost for code repositories

            # Deduplicate results by repository/profile owner
            seen_repos = set()
            deduplicated_results = []

            for result in filtered_results:
                url = result.get("url", "")
                # Extract repo owner/name to avoid duplicate repositories
                parts = url.replace("https://github.com/", "").split("/")
                repo_identifier = "/".join(parts[:2]) if len(parts) >= 2 else url

                # Skip if we've already seen this repository or profile
                if repo_identifier in seen_repos:
                    continue

                deduplicated_results.append(result)
                seen_repos.add(repo_identifier)

            # Sort by score in descending order and take top_k
            deduplicated_results = sorted(
                deduplicated_results, key=lambda x: x.get("score", 0), reverse=True
            )[:top_k]

            # Keep raw_content only for repositories with actual code
            for result in deduplicated_results:
                url = result.get("url", "").lower()
                # Remove raw_content for results that are less likely to contain valuable code
                if not (
                    "/blob/" in url
                    or "/tree/" in url
                    or "/releases/" in url
                    or "/wiki/" in url
                ):
                    result["raw_content"] = None

            # Update the results in the response
            search_results["results"] = deduplicated_results

            print(
                f"Filtered GitHub results: {len(deduplicated_results)} out of {len(search_results.get('results', []))} meet the score threshold of {min_score} and are unique repositories/profiles."
            )

        log_tavily_status("GITHUB", query, search_results)
        return search_results

    except requests.exceptions.HTTPError as e:
        # Handle specific error codes
        if e.response.status_code == 422:
            print(f"Tavily API validation error (422): {e.response.text}")
            print("This is likely due to invalid parameters. Returning empty results.")
            # Return a valid empty response instead of retrying
            error_response = {
                "results": [],
                "error": f"Tavily API validation error: {str(e)}",
                "search_string": query,
                "response_time": 0,
            }
            log_tavily_status("GITHUB", query, error_response)
            return error_response
        elif e.response.status_code == 401:
            print(f"Tavily API authentication error (401): Check your API key")
            # Return empty results for auth errors
            error_response = {
                "results": [],
                "error": "Tavily API authentication error: Invalid API key",
                "search_string": query,
                "response_time": 0,
            }
            log_tavily_status("GITHUB", query, error_response)
            return error_response
        else:
            print(f"Tavily API error: {e}. Retrying...")
            raise  # Re-raise to trigger the retry
    except Exception as e:
        print(f"Unexpected error in Tavily search: {str(e)}")
        # Return a valid empty response instead of failing
        error_response = {
            "results": [],
            "error": f"Unexpected error in Tavily search: {str(e)}",
            "search_string": query,
            "response_time": 0,
        }
        log_tavily_status("GITHUB", query, error_response)
        return error_response


@traceable
@retry(
    wait=wait_exponential(
        multiplier=1, min=2, max=60
    ),  # Start with 2s, then 4s, 8s, 16s, 32s, 60s
    stop=stop_after_attempt(5),  # Try 5 times maximum
    retry=retry_if_exception_type(
        requests.exceptions.HTTPError
    ),  # Only retry on HTTPError
)
def academic_search(
    query, include_raw_content=True, top_k=5, min_score=0.2, recent_years=None
):
    """Search for academic papers and research publications using Tavily

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        top_k (int): Maximum number of high-scoring results to return (default: 5)
        min_score (float): Minimum score threshold for results (default: 0.65)
        recent_years (int, optional): If provided, prioritize papers from the last N years
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries matching the criteria:
                - Only returns top K results sorted by score in descending order
                - Only returns results with scores higher than min_score
                - Prioritizes academic sources over general web content
                - Optionally prioritizes recent publications if recent_years is specified
            - response_time (float): Time taken to complete the search
    """

    # Determine if the query is likely academic already
    is_academic_query = any(
        term in query.lower()
        for term in [
            "paper",
            "research",
            "journal",
            "publication",
            "article",
            "study",
            "conference",
        ]
    )

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    # Validate query length - Tavily has a 400 character limit
    if len(query) > 400:
        print(
            f"WARNING: Query exceeds Tavily's 400 character limit. Query length: {len(query)}"
        )
        # Truncate query to 397 characters + "..."
        query = query[:397] + "..."
        print(f"Truncated query: {query}")

    # Validate query isn't empty after stripping whitespace
    query = query.strip()
    if not query:
        print("ERROR: Empty query after stripping whitespace")
        return {"results": [], "search_string": "", "response_time": 0}

    # Focus on academic domains
    include_domains = [
        "arxiv.org",
        "scholar.google.com",
        "researchgate.net",
        "semanticscholar.org",
        "acm.org",
        "ieee.org",
        "sciencedirect.com",
        "nature.com",
        "science.org",
        "springer.com",
        "wiley.com",
        "jstor.org",
        "ncbi.nlm.nih.gov",
        "pubmed.gov",
        "academia.edu",
        "biorxiv.org",
        "crossref.org",
    ]

    # If the query already includes academic terms, use it directly
    # Otherwise, enhance the query with academic context
    academic_query = query
    if not is_academic_query:
        academic_query = f"academic research papers about {query}"

    # If recent_years is provided, prioritize recent publications
    if recent_years and isinstance(recent_years, int) and recent_years > 0:
        current_year = datetime.datetime.now().year
        start_year = current_year - recent_years
        if (
            "after:" not in academic_query.lower()
            and "before:" not in academic_query.lower()
        ):
            academic_query = f"{academic_query} after:{start_year}"

    print(f"Enhanced academic query: {academic_query}")

    tavily_client = TavilyClient(api_key=api_key)
    try:
        # Log the parameters
        search_params = log_general_deep_search_params(
            query=academic_query,
            max_results=20,
            include_answer="basic",
            include_raw_content=include_raw_content,
            include_domains=include_domains,
            search_type="keyword" if ":" in academic_query else "search",
            search_depth="advanced",  # Use advanced search depth for academic queries
        )

        # Get the raw search results
        search_results = tavily_client.search(**search_params)

        # Filter results by score and get the top_k
        if "results" in search_results:
            # Filter out results with score < min_score
            filtered_results = [
                r for r in search_results["results"] if r.get("score", 0) >= min_score
            ]

            # Deduplicate results by title and URL and handle academic-specific conditions
            seen_titles = set()
            seen_urls = set()
            deduplicated_results = []

            # Helper function to check for sufficient similarity between titles
            def similar(a, b, threshold=0.8):
                """Check if two strings are similar enough to be considered duplicates."""
                if not a or not b:
                    return False
                matcher = SequenceMatcher(None, a.lower(), b.lower())
                return matcher.ratio() > threshold

            # Boost scores of results from academic domains
            for result in filtered_results:
                url = result.get("url", "").lower()

                # Extract domain for checking
                domain = None
                try:
                    from urllib.parse import urlparse

                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc
                except Exception:
                    pass

                # Apply score boosts based on source quality
                if domain:
                    # Strong boost for primary academic sources
                    if domain in [
                        "arxiv.org",
                        "nature.com",
                        "science.org",
                        "acm.org",
                        "ieee.org",
                    ]:
                        result["score"] = result.get("score", 0) * 1.3  # 30% boost
                    # Moderate boost for other good academic sources
                    elif domain in [
                        "scholar.google.com",
                        "researchgate.net",
                        "semanticscholar.org",
                    ]:
                        result["score"] = result.get("score", 0) * 1.2  # 20% boost
                    # Small boost for other academic domains
                    elif any(d in domain for d in include_domains):
                        result["score"] = result.get("score", 0) * 1.1  # 10% boost

                # Boost score if title contains academic keywords
                title = result.get("title", "").lower()
                if any(
                    term in title
                    for term in [
                        "paper",
                        "research",
                        "journal",
                        "conference",
                        "proceedings",
                        "study",
                    ]
                ):
                    result["score"] = result.get("score", 0) * 1.1  # 10% boost

            # Sort by boosted score before deduplication
            filtered_results = sorted(
                filtered_results, key=lambda x: x.get("score", 0), reverse=True
            )

            # Handle deduplication with similarity check for academic titles
            for result in filtered_results:
                title = result.get("title", "").strip()
                url = result.get("url", "").strip()

                # Skip if this result has a duplicate title or URL
                if url in seen_urls:
                    continue

                # Check for similar titles (academic papers may have similar titles with minor differences)
                title_duplicate = False
                for seen_title in seen_titles:
                    if similar(title, seen_title):
                        title_duplicate = True
                        break

                if title_duplicate:
                    continue

                # Add to deduplicated results
                deduplicated_results.append(result)
                seen_titles.add(title)
                seen_urls.add(url)

                # Stop if we have enough results
                if len(deduplicated_results) >= top_k:
                    break

            # Update the results in the response (no need to re-sort as we sorted before deduplication)
            search_results["results"] = deduplicated_results[:top_k]

            print(
                f"Filtered Academic results: {len(deduplicated_results)} out of {len(search_results.get('results', []))} meet the score threshold of {min_score} and are unique by title/URL."
            )

        log_tavily_status("ACADEMIC", academic_query, search_results)
        return search_results

    except requests.exceptions.HTTPError as e:
        # Handle specific error codes
        if e.response.status_code == 422:
            print(f"Tavily API validation error (422): {e.response.text}")
            print("This is likely due to invalid parameters. Returning empty results.")
            # Return a valid empty response instead of retrying
            error_response = {
                "results": [],
                "error": f"Tavily API validation error: {str(e)}",
                "search_string": academic_query,
                "response_time": 0,
            }
            log_tavily_status("ACADEMIC", academic_query, error_response)
            return error_response
        elif e.response.status_code == 401:
            print(f"Tavily API authentication error (401): Check your API key")
            # Return empty results for auth errors
            error_response = {
                "results": [],
                "error": "Tavily API authentication error: Invalid API key",
                "search_string": academic_query,
                "response_time": 0,
            }
            log_tavily_status("ACADEMIC", academic_query, error_response)
            return error_response
        else:
            print(f"Tavily API error: {e}. Retrying...")
            raise  # Re-raise to trigger the retry
    except Exception as e:
        print(f"Unexpected error in Tavily search: {str(e)}")
        # Return a valid empty response instead of failing
        error_response = {
            "results": [],
            "error": f"Unexpected error in Tavily search: {str(e)}",
            "search_string": academic_query,
            "response_time": 0,
        }
        log_tavily_status("ACADEMIC", academic_query, error_response)
        return error_response


def compare_search_types(
    query, include_raw_content=True, with_original=True, config=None
):
    """Run the same query through all specialized search functions and compare results

    This function helps demonstrate how each specialized search function processes
    the same query differently, with different parameters, filtering, and boosting.

    Args:
        query (str): The query to test across all search types
        include_raw_content (bool): Whether to include raw content in the results
        with_original (bool): Whether to include the original general_deep_search in comparison
        config (RunnableConfig, optional): Configuration object for LangSmith tracing

    Returns:
        dict: Results from each search type with metadata about the differences
    """
    print(f"Comparing search types for query: '{query}'")

    results = {}
    start_time = datetime.datetime.now()

    # Run all specialized searches with the same query
    if with_original:
        print("\nRunning original general_deep_search...")
        results["original"] = general_deep_search(
            query, include_raw_content=include_raw_content, config=config
        )

    print("\nRunning linkedin_search...")
    results["linkedin"] = linkedin_search(
        query, include_raw_content=include_raw_content
    )

    print("\nRunning github_search...")
    results["github"] = github_search(query, include_raw_content=include_raw_content)

    print("\nRunning academic_search...")
    results["academic"] = academic_search(
        query, include_raw_content=include_raw_content
    )

    # Calculate total time
    total_time = (datetime.datetime.now() - start_time).total_seconds()

    # Add comparison metadata
    comparison = {
        "query": query,
        "total_time": total_time,
        "results_count": {k: len(v.get("results", [])) for k, v in results.items()},
        "avg_scores": {
            k: (
                sum(r.get("score", 0) for r in v.get("results", []))
                / len(v.get("results", []))
                if v.get("results")
                else 0
            )
            for k, v in results.items()
        },
        "domains_found": {
            k: list(set(extract_domain(r.get("url", "")) for r in v.get("results", [])))
            for k, v in results.items()
        },
    }

    # Print comparison summary
    print("\n" + "=" * 80)
    print("SEARCH COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Query: '{query}'")
    print(f"Total time: {total_time:.2f} seconds")
    print("\nResults count:")
    for k, v in comparison["results_count"].items():
        print(f"  - {k}: {v} results")

    print("\nAverage scores:")
    for k, v in comparison["avg_scores"].items():
        print(f"  - {k}: {v:.4f} average score")

    print("\nDomains found:")
    for k, domains in comparison["domains_found"].items():
        print(
            f"  - {k}: {', '.join(domains[:5])}"
            + (f" and {len(domains)-5} more..." if len(domains) > 5 else "")
        )

    # Add comparison to results
    results["comparison"] = comparison

    return results


def extract_domain(url):
    """Extract the domain from a URL.

    Args:
        url (str): The URL to extract the domain from

    Returns:
        str: The extracted domain
    """
    try:
        if not url.startswith("http"):
            url = "https://" + url

        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        # Remove www. prefix if present
        if domain.startswith("www."):
            domain = domain[4:]

        return domain
    except Exception as e:
        print(f"Error extracting domain from URL {url}: {str(e)}")
        # Fallback to simple extraction
        if "//" in url:
            return url.split("//")[1].split("/")[0]
        else:
            return url.split("/")[0]


def generate_numbered_sources(sources_list):
    """Generate numbered sources from the list of formatted sources, ensuring uniqueness by URL.

    Args:
        sources_list (list): List of formatted source strings (each formatted as "* Title : URL")

    Returns:
        tuple: (list of numbered sources, dict mapping numbers to source metadata)
    """
    # Keep track of citation numbers and their corresponding sources
    source_citations = {}
    numbered_sources = []
    seen_urls = set()  # Keep track of URLs we have already processed
    next_citation_number = 1  # Start numbering from 1

    print(
        f"DEBUG (generate_numbered_sources): Received {len(sources_list)} sources to process."
    )

    for source in sources_list:
        if source and ":" in source:
            # Split the source string to get title and URL
            parts = source.split(" : ", 1)
            if len(parts) == 2:
                title = parts[0].strip().lstrip("* ")
                url = parts[1].strip()

                # --- START FIX: Deduplication by URL ---
                if url in seen_urls:
                    # print(f"DEBUG: Skipping duplicate URL: {url}") # Optional: uncomment for verbose logging
                    continue  # Skip this source if we've already processed this URL
                # --- END FIX ---

                # Add URL to seen set
                seen_urls.add(url)

                # Use the next available citation number
                current_citation_number = next_citation_number

                # Create numbered citation
                numbered_source = f"{current_citation_number}. {title}, [{url}]"
                numbered_sources.append(numbered_source)

                # Store in our citation mapping
                source_citations[current_citation_number] = {
                    "title": title,
                    "url": url,
                    "number": current_citation_number,
                }

                # Increment the citation number for the next unique source
                next_citation_number += 1
            else:
                print(
                    f"DEBUG (generate_numbered_sources): Skipping malformed source string: {source}"
                )
        else:
            print(
                f"DEBUG (generate_numbered_sources): Skipping empty or malformed source string: {source}"
            )

    print(
        f"DEBUG (generate_numbered_sources): Produced {len(numbered_sources)} unique numbered sources."
    )
    return numbered_sources, source_citations


# New consolidated Tavily search function
def tavily_search_proper(
    query: str,
    search_depth: str = "basic",
    max_results: int = 7,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    include_answer: bool = False,
    include_raw_content: bool = False,
    include_images: bool = False,
) -> List[Dict[str, Any]]:
    """
    Performs a search using the Tavily API with specified parameters and robust error handling.

    Args:
        query: The search query string.
        search_depth: "basic" or "advanced". Advanced search gives more in-depth results.
        max_results: The maximum number of results to return.
        include_domains: A list of domains to specifically include in the search.
        exclude_domains: A list of domains to specifically exclude from the search.
        include_answer: Whether to include a direct answer to the query, if available.
        include_raw_content: Whether to include raw content of the search results.
        include_images: Whether to include image results.

    Returns:
        A list of search result dictionaries, or an empty list if an error occurs.
    """
    if not os.getenv("TAVILY_API_KEY"):
        print("TAVILY_API_KEY not set. Please set the environment variable.")
        return []

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    retries = 0

    # Clean up query to remove any problematic characters if necessary
    # (Example: removing excessive newlines or non-standard characters if they cause issues)
    # query = re.sub(r'\s+', ' ', query).strip() # Basic cleanup, adjust as needed

    while retries < 5:
        try:
            print(
                f"Attempting Tavily search for: '{query[:100]}...' (Attempt {retries + 1})"
            )
            response = client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_images=include_images,
            )
            # Successfully got a response
            # Ensure response is a list of dictionaries as expected
            if isinstance(response, dict) and "results" in response:
                # Check if 'results' is a list of dicts
                if isinstance(response["results"], list) and all(
                    isinstance(item, dict) for item in response["results"]
                ):
                    print(
                        f"Tavily search successful. Got {len(response['results'])} results."
                    )
                    return response["results"]
                else:
                    print(
                        f"Warning: Tavily response['results'] is not a list of dicts: {type(response['results'])}"
                    )
                    return []  # Or handle as appropriate
            elif isinstance(response, list) and all(
                isinstance(item, dict) for item in response
            ):
                # Sometimes Tavily might directly return a list of results
                print(
                    f"Tavily search successful (direct list). Got {len(response)} results."
                )
                return response
            else:
                print(f"Warning: Unexpected Tavily response format: {type(response)}")
                # Log the actual response for debugging if it's not too large
                print(f"Full Tavily response: {str(response)[:500]}...")
                return []  # Return empty list for unexpected format

        except Exception as e:
            error_message = str(e)
            if (
                "Max retries exceeded with url" in error_message
                or "('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))"
                in error_message
                or isinstance(e, httpx.RemoteProtocolError)
                or isinstance(e, httpx.ReadTimeout)
                or isinstance(e, httpx.ConnectError)
            ):
                # These are typically transient network issues or server-side problems
                print(f"Network/Connection error with Tavily API: {e}. Retrying...")
            elif (
                "429" in error_message or "rate_limit_exceeded" in error_message.lower()
            ):
                # Specific handling for rate limit errors
                print(f"Tavily API rate limit exceeded: {e}. Retrying after a delay...")
                time.sleep(2 * (retries + 1))  # Exponential backoff might be better
            elif "TAVILY_API_KEY" in error_message:  # Catch API key specific errors
                print(f"Tavily API key error: {e}. Please check your TAVILY_API_KEY.")
                return []  # Do not retry if API key is the issue
            else:
                print(f"Tavily API error: {e}. Retrying...")  # General Tavily API error

            retries += 1
            if retries < 5:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"Max retries reached for Tavily API. Error: {e}")
                return []  # Failed after all retries

    print(
        f"Failed to retrieve search results for query: '{query[:100]}...' after 5 retries."
    )
    return []


def extract_author_and_year_from_content(
    title: str, content: str = "", url: str = ""
) -> tuple:
    """
    Extract author and year information from search result content.

    Args:
        title: The title of the article/paper
        content: The raw content/snippet from the search result
        url: The URL of the source

    Returns:
        tuple: (first_author, year) where first_author is the first author's last name
               and year is the publication year. Returns (None, None) if not found.
    """
    first_author = None
    year = None

    # Ensure all inputs are strings to avoid NoneType errors
    title = title or ""
    content = content or ""
    url = url or ""

    # Combine title and content for analysis
    full_text = f"{title} {content}".lower()

    # Try to extract year first (look for 4-digit years, preferably recent ones)
    year_patterns = [
        r"\b(20[0-2][0-9])\b",  # 2000-2029
        r"\b(19[89][0-9])\b",  # 1980-1999
        r"\((\d{4})\)",  # Year in parentheses
        r"(\d{4})",  # Any 4-digit number
    ]

    for pattern in year_patterns:
        matches = re.findall(pattern, full_text)
        if matches:
            # Take the most recent year that's reasonable for academic content
            years = [int(match) for match in matches if 1980 <= int(match) <= 2025]
            if years:
                year = max(years)
                break

    # Try to extract author information
    author_patterns = [
        # Common academic patterns
        r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)",  # "by Author Name"
        r"([A-Z][a-z]+),?\s+[A-Z]\.(?:\s*[A-Z]\.)*",  # "Smith, J. A."
        r"([A-Z][a-z]+)\s+et\s+al\.?",  # "Smith et al."
        r"([A-Z][a-z]+)\s+and\s+[A-Z][a-z]+",  # "Smith and Jones"
        # DOI/citation patterns
        r"doi.*?([A-Z][a-z]+),?\s+[A-Z]\.",  # DOI with author
        # ArXiv patterns
        r"arxiv.*?([A-Z][a-z]+),?\s+[A-Z]\.",  # ArXiv with author
    ]

    for pattern in author_patterns:
        # Use the sanitized content string to avoid NoneType errors
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            # Take the first match as the first author
            first_author = matches[0].strip()
            # Clean up the author name and remove "et al" if present
            first_author = re.sub(r"[^\w\s-]", "", first_author).strip()
            first_author = re.sub(
                r"\s+et\s+al\.?$", "", first_author, flags=re.IGNORECASE
            ).strip()
            if first_author and len(first_author) > 1:
                break

    # If no author found in content, try to extract from URL patterns
    if not first_author and url:
        # Some academic sites have author info in URLs
        url_patterns = [
            r"author[s]?[=/-]([A-Z][a-z]+)",  # author=Smith or authors/Smith
            r"profile[s]?[=/-]([A-Z][a-z]+)",  # profile=Smith
        ]

        for pattern in url_patterns:
            matches = re.findall(pattern, url, re.IGNORECASE)
            if matches:
                first_author = matches[0].strip()
                break

    return first_author, year
