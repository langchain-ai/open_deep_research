from open_deep_research.general_researcher.state import SearchSource
from open_deep_research.general_researcher.configuration import SearchAPI, WorkflowConfiguration
from open_deep_research.utils import (
    tavily_search,
)
from langchain_core.tools import BaseTool, StructuredTool, tool, ToolException
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
import json
from langchain_core.messages import MessageLikeRepresentation, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing import Annotated, List, Literal
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langsmith import traceable
from tavily import AsyncTavilyClient
from langchain_anthropic import ChatAnthropic
from src.open_deep_research.general_researcher.prompts import SUMMARIZATION_PROMPT
from pydantic import BaseModel
import logging
import aiohttp
from typing import Dict, Optional, Any
from langgraph.config import get_store
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession, Tool, McpError
from langchain_mcp_adapters.client import MultiServerMCPClient
import warnings


class Summary(BaseModel):
    summary: str
    key_excerpts: list[str]

TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)

@traceable
async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True):
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )
    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content."""
    try:
        user_input_content = "Please summarize the article"
        if isinstance(model, ChatAnthropic):
            user_input_content = [{
                "type": "text",
                "text": user_input_content,
            }]
        summary = await asyncio.wait_for(
            model.with_structured_output(Summary).with_retry(stop_after_attempt=3).ainvoke([
                {"role": "system", "content": SUMMARIZATION_PROMPT.format(webpage_content=webpage_content)},
                {"role": "user", "content": user_input_content},
            ]),
            timeout=60.0  # 60 second timeout
        )
    except (asyncio.TimeoutError, Exception) as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content

    def format_summary(summary: Summary):
        excerpts_str = "\n".join(f'- {e}' for e in summary.key_excerpts)
        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{excerpts_str}\n</key_excerpts>"""

    return format_summary(summary)


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """
    Fetches results from Tavily search API.

    Args:
        queries (List[str]): List of search queries 
        max_results (int): Maximum number of results to return
        topic (Literal['general', 'news', 'finance']): Topic to filter results by

    Returns:
        str: A formatted string of search results
    """
    # Use tavily_search_async with include_raw_content=True to get content directly
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True
    )

    # Format the search results directly using the raw_content already provided
    formatted_output = f"Search results: \n\n"
    
    # Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    async def noop():
        return None

    configurable = WorkflowConfiguration.from_runnable_config(config)
    max_char_to_include = 30_000

    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_retries=configurable.max_structured_output_retries,
    )
    summarization_tasks = [
        noop() if not result.get("raw_content") else summarize_webpage(summarization_model, result['raw_content'][:max_char_to_include])
        for result in unique_results.values()
    ]
    summaries = await asyncio.gather(*summarization_tasks)
    unique_results = {
        url: {'title': result['title'], 'content': result['content'] if summary is None else summary}
        for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
    }
    # Format the unique results
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        if result.get('raw_content'):
            formatted_output += f"FULL CONTENT:\n{result['raw_content'][:max_char_to_include]}"  # Limit content size
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    if unique_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."

async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return [{"type": "web_search_20250305", "name": "web_search"}]
    elif search_api == SearchAPI.OPENAI:
        return [{"type": "web_search_preview"}]
    elif search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        tool_metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        search_tool.metadata = tool_metadata
        return [search_tool]
    
async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """
    Exchange a Supabase token for an MCP access token.

    Args:
        supabase_token: The Supabase token to exchange
        base_mcp_url: The base URL for the MCP server

    Returns:
        The token data as a dictionary if successful, None otherwise
    """
    try:
        # Exchange Supabase token for MCP access token
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                base_mcp_url.rstrip("/") + "/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=form_data,
            ) as token_response:
                if token_response.status == 200:
                    token_data = await token_response.json()
                    return token_data
                else:
                    response_text = await token_response.text()
                    logging.error(f"Token exchange failed: {response_text}")
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")

    return None


async def get_tokens(config: RunnableConfig):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None

    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None

    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation

    from datetime import datetime, timedelta, timezone

    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)

    if current_time > expiration_time:
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return

    await store.aput((user_id, "tokens"), "data", tokens)
    return


async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens

    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None

    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None

    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))

    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap the tool coroutine to handle `interaction_required` MCP error.

    Tried to obtain the URL from the error, which the LLM can use to render a link."""

    old_coroutine = tool.coroutine

    async def wrapped_mcp_coroutine(**kwargs):
        def _find_first_mcp_error_nested(exc: BaseException) -> McpError | None:
            if isinstance(exc, McpError):
                return exc
            if isinstance(exc, ExceptionGroup):
                for sub_exc in exc.exceptions:
                    if found := _find_first_mcp_error_nested(sub_exc):
                        return found
            return None

        try:
            return await old_coroutine(**kwargs)
        except BaseException as e_orig:
            mcp_error = _find_first_mcp_error_nested(e_orig)

            if not mcp_error:
                raise e_orig

            error_details = mcp_error.error
            is_interaction_required = getattr(error_details, "code", None) == -32003
            error_data = getattr(error_details, "data", None) or {}

            if is_interaction_required:
                message_payload = error_data.get("message", {})
                error_message_text = "Required interaction"
                if isinstance(message_payload, dict):
                    error_message_text = (
                        message_payload.get("text") or error_message_text
                    )

                if url := error_data.get("url"):
                    error_message_text = f"{error_message_text} {url}"
                raise ToolException(error_message_text) from e_orig

            raise e_orig

    tool.coroutine = wrapped_mcp_coroutine
    return tool


async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = WorkflowConfiguration.from_runnable_config(config)
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    if not (configurable.mcp_config and configurable.mcp_config.url and configurable.mcp_config.tools and (mcp_tokens or not configurable.mcp_config.auth_required)):
        return []
    tool_names_to_find = set(configurable.mcp_config.tools)
    tools = []
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    mcp_server_config = {
        "server_1":{
            "url": server_url,
            "headers": {"Authorization": f"Bearer {mcp_tokens['access_token']}"} if mcp_tokens else None,
            "transport": "streamable_http"
        }
    }
    client = MultiServerMCPClient(mcp_server_config)
    mcp_tools = await client.get_tools()
    for tool in mcp_tools:
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue
        if tool.name not in tool_names_to_find:
            continue
        tools.append(wrap_mcp_authenticate_tool(tool))
    return tools

def extract_content_from_model_websearch_response(response):
    text_blocks = []
    if hasattr(response, 'content'):
        for content_block in response.content:
            if "type" in content_block and content_block["type"] == "text":
                text_blocks.append(content_block["text"])
    return "\n\n".join(text_blocks) if text_blocks else ""
    
def call_source_extractor(response, iteration, search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return collect_sources_anthropic(response, iteration)
    elif search_api == SearchAPI.OPENAI:
        return collect_sources_openai(response, iteration)
    else:
        return []
    
def collect_sources_anthropic(response, iteration):
    collected_sources = []
    if hasattr(response, 'content'):    
        for content_block in response.content:
            if "type" in content_block:
                if content_block["type"] == 'web_search_tool_result':
                    if "content" in content_block:
                        for search_result in content_block["content"]:
                            if "type" in search_result and search_result["type"] == 'web_search_result':
                                source = SearchSource(
                                    url=search_result.get('url', ''),
                                    title=search_result.get('title', ''),
                                    content=f"Source found in iteration {iteration}",
                                    search_query=f"iteration_{iteration}",
                                    iteration=iteration
                                )
                                collected_sources.append(source)
                
                elif content_block["type"] == 'text':
                    if "citations" in content_block:
                        for citation in content_block["citations"]:
                            if "url" in citation and "title" in citation:
                                cited_content = citation.get('cited_text', '')
                                if not cited_content:
                                    cited_content = f"Citation from {citation['title']}"
                                source = SearchSource(
                                    url=citation["url"],
                                    title=citation["title"],
                                    content=cited_content[:1000],
                                    search_query=f"citation_iteration_{iteration}",
                                    iteration=iteration
                                )
                                collected_sources.append(source)
    return collected_sources

def collect_sources_openai(response, iteration):
    """Extract sources from OpenAI response format"""
    collected_sources = []
    if hasattr(response, 'content'):
        for content_block in response.content:
            if "type" in content_block and content_block["type"] == "text":
                if "annotations" in content_block:
                    for annotation in content_block["annotations"]:
                        if annotation.get("type") == "url_citation":
                            # Extract the cited text using start/end indices
                            text = content_block.get("text", "")
                            start_idx = annotation.get("start_index", 0)
                            end_idx = annotation.get("end_index", len(text))
                            cited_text = text[start_idx:end_idx] if text else ""
                            
                            source = SearchSource(
                                url=annotation.get("url", ""),
                                title=annotation.get("title", ""),
                                content=cited_text[:1000] if cited_text else f"Citation from {annotation.get('title', 'Unknown')}",
                                search_query=f"citation_iteration_{iteration}",
                                iteration=iteration
                            )
                            collected_sources.append(source)
            elif content_block.get("type") == 'web_search_tool_result':
                if "content" in content_block:
                    for search_result in content_block["content"]:
                        if "type" in search_result and search_result["type"] == 'web_search_result':
                            source = SearchSource(
                                url=search_result.get('url', ''),
                                title=search_result.get('title', ''),
                                content=f"Source found in iteration {iteration}",
                                search_query=f"iteration_{iteration}",
                                iteration=iteration
                            )
                            collected_sources.append(source)
    return collected_sources

def anthropic_websearch_called(response):
    try:
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        return web_search_requests > 0
    except (AttributeError, TypeError):
        return False

def openai_websearch_called(response):
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if tool_outputs:
        for tool_output in tool_outputs:
            if tool_output.get("type") == "web_search_call":
                return True
    return False

def handle_ai_message_response_types(message, search_api: SearchAPI, iteration: int):
    observation = ""
    collected_sources = []
    # If we provided Anthropic Search, we need to inspect the response_metadata to see if the model called a native web search tool.
    if search_api == SearchAPI.ANTHROPIC and anthropic_websearch_called(message):
        observation = extract_content_from_model_websearch_response(message)
        collected_sources = collect_sources_anthropic(message, iteration)
    # If we provided OpenAI Search, we need to inspect the additional_kwargs to see if the model called a native web search tool.
    elif search_api == SearchAPI.OPENAI and openai_websearch_called(message):
        observation = extract_content_from_model_websearch_response(message)
        collected_sources = collect_sources_openai(message, iteration)
    elif len(message.tool_calls) > 0:
        for tool_call in message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args")
            if tool_name and tool_args:
                observation += f"{tool_name}: {json.dumps(tool_args)}\n"
    else:
        observation = message.content
    # If observation is not a string, convert it to a string
    if not isinstance(observation, str):
        observation = str(observation)
    return observation, collected_sources


def extract_notes_from_research_messages(research_messages, research_iterations, search_api: SearchAPI):
    notes = f"Notes from Research Iteration {research_iterations}:\n\n"
    collected_sources = []
    for message in research_messages:
        if isinstance(message, AIMessage):
            observation, new_sources = handle_ai_message_response_types(message, search_api, research_iterations)
            notes += observation
            if new_sources:
                sources_str = "\n".join([f"- {source.title} ({source.url})" for source in new_sources])
                notes += f"\nSources:\n{sources_str}\n\n"
                collected_sources.extend(new_sources)
        elif isinstance(message, ToolMessage):
            notes += f"{message.name}: {message.content}\n\n"
    return notes, collected_sources

def get_conversation_history(messages: list[MessageLikeRepresentation]) -> str:
    return f"This is the conversation between you and the user that you are researching for:\n\n{get_buffer_string(messages)}"