import os
import aiohttp
import asyncio
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Literal, Dict, Optional, Any
from langchain_core.tools import BaseTool, StructuredTool, tool, ToolException, InjectedToolArg
from langchain_core.messages import HumanMessage, AIMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from tavily import AsyncTavilyClient
from langgraph.config import get_store
from mcp import McpError
from langchain_mcp_adapters.client import MultiServerMCPClient
from open_deep_research.state import Summary, ResearchComplete
from open_deep_research.configuration import SearchAPI, Configuration
from open_deep_research.prompts import summarize_webpage_prompt

import json
import datetime
import ssl
import certifi

import httpx
from pydantic import BaseModel
from langchain_core.tools import tool


##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """
    Fetches results from Tavily search API.

    Args
        queries (List[str]): List of search queries, you can pass in as many queries as you need.
        max_results (int): Maximum number of results to return
        topic (Literal['general', 'news', 'finance']): Topic to filter results by

    Returns:
        str: A formatted string of search results
    """
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    # Format the search results and deduplicate results by URL
    formatted_output = f"Search results: \n\n"
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = 50_000   # NOTE: This can be tuned by the developer. This character count keeps us safely under input token limits for the latest models.
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    async def noop():
        return None
    summarization_tasks = [
        noop() if not result.get("raw_content") else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include],
        )
        for result in unique_results.values()
    ]
    summaries = await asyncio.gather(*summarization_tasks)
    summarized_results = {
        url: {'title': result['title'], 'content': result['content'] if summary is None else summary}
        for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
    }
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    if summarized_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True, config: RunnableConfig = None):
    tavily_async_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
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
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    try:
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=summarize_webpage_prompt.format(webpage_content=webpage_content, date=get_today_str()))]),
            timeout=60.0
        )
        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"""
    except (asyncio.TimeoutError, Exception) as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content


##########################
# MCP Utils
##########################
async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    try:
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # Get SSL configuration from environment or use default
        verify_ssl = os.getenv("OPEN_DEEP_RESEARCH_VERIFY_SSL", "true").lower() == "true"
        
        # Use our safe HTTP request function
        response = await safe_http_request(
            url=base_mcp_url.rstrip("/") + "/oauth/token",
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=form_data,
            verify_ssl=verify_ssl,
            timeout=30,
            max_retries=3
        )
        
        if response["status"] == 200:
            return response["json"]
        else:
            logging.error(f"Token exchange failed: {response['text']}")
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
    configurable = Configuration.from_runnable_config(config)
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    if not (configurable.mcp_config and configurable.mcp_config.url and configurable.mcp_config.tools and (mcp_tokens or not configurable.mcp_config.auth_required)):
        return []
    tools = []
    # TODO: When the Multi-MCP Server support is merged in OAP, update this code.
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    mcp_server_config = {
        "server_1":{
            "url": server_url,
            "headers": {"Authorization": f"Bearer {mcp_tokens['access_token']}"} if mcp_tokens else None,
            "transport": "streamable_http"
        }
    }
    try:
        client = MultiServerMCPClient(mcp_server_config)
        mcp_tools = await client.get_tools()
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []
    for tool in mcp_tools:
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue
        if tool.name not in set(configurable.mcp_config.tools):
            continue
        tools.append(wrap_mcp_authenticate_tool(tool))
    return tools


##########################
# Tool Utils
##########################
async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    elif search_api == SearchAPI.OPENAI:
        return [{"type": "web_search_preview"}]
    elif search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        search_tool.metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        return [search_tool]
    elif search_api == SearchAPI.NONE:
        return []
    
async def get_all_tools(config: RunnableConfig):
    tools = [tool(ResearchComplete)]
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    tools.extend(await get_search_tool(search_api))
    existing_tool_names = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search") for tool in tools}
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


##########################
# Model Provider Native Websearch Utils
##########################
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


##########################
# Token Limit Exceeded Utils
##########################
def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    error_str = str(exception).lower()
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    return (_check_openai_token_limit(exception, error_str) or
            _check_anthropic_token_limit(exception, error_str) or
            _check_gemini_token_limit(exception, error_str))

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    is_openai_exception = ('openai' in exception_type.lower() or 
                          'openai' in module_name.lower())
    is_bad_request = class_name in ['BadRequestError', 'InvalidRequestError']
    if is_openai_exception and is_bad_request:
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        if (getattr(exception, 'code', '') == 'context_length_exceeded' or
            getattr(exception, 'type', '') == 'invalid_request_error'):
            return True
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    is_anthropic_exception = ('anthropic' in exception_type.lower() or 
                             'anthropic' in module_name.lower())
    is_bad_request = class_name == 'BadRequestError'
    if is_anthropic_exception and is_bad_request:
        if 'prompt is too long' in error_str:
            return True
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    is_google_exception = ('google' in exception_type.lower() or 'google' in module_name.lower())
    is_resource_exhausted = class_name in ['ResourceExhausted', 'GoogleGenerativeAIFetchError']
    if is_google_exception and is_resource_exhausted:
        return True
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
}

def get_model_token_limit(model_string):
    for key, token_limit in MODEL_TOKEN_LIMITS.items():
        if key in model_string:
            return token_limit
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]  # Return everything up to (but not including) the last AI message
    return messages

##########################
# Misc Utils
##########################
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_config_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")

##########################
# SSL/TLS Error Handling Utils
##########################

def create_ssl_context(verify_ssl: bool = True) -> ssl.SSLContext:
    """
    Create an SSL context with proper error handling.
    
    Args:
        verify_ssl (bool): Whether to verify SSL certificates. Set to False to bypass SSL verification.
        
    Returns:
        ssl.SSLContext: Configured SSL context
    """
    if verify_ssl:
        # Use certifi for certificate verification
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
    else:
        # Create context that bypasses SSL verification (for development/testing only)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        warnings.warn("SSL verification disabled. This should only be used for development/testing.")
    
    return ssl_context

def create_aiohttp_session(verify_ssl: bool = True, timeout: int = 30) -> aiohttp.ClientSession:
    """
    Create an aiohttp session with proper SSL handling.
    
    Args:
        verify_ssl (bool): Whether to verify SSL certificates
        timeout (int): Timeout in seconds
        
    Returns:
        aiohttp.ClientSession: Configured session
    """
    connector = aiohttp.TCPConnector(
        ssl=create_ssl_context(verify_ssl),
        limit=100,
        limit_per_host=30,
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    
    timeout_config = aiohttp.ClientTimeout(total=timeout)
    
    return aiohttp.ClientSession(
        connector=connector,
        timeout=timeout_config,
        headers={
            'User-Agent': 'OpenDeepResearch/1.0 (https://github.com/langchain-ai/open_deep_research)'
        }
    )

def create_httpx_client(verify_ssl: bool = True, timeout: int = 30) -> httpx.AsyncClient:
    """
    Create an httpx client with proper SSL handling.
    
    Args:
        verify_ssl (bool): Whether to verify SSL certificates
        timeout (int): Timeout in seconds
        
    Returns:
        httpx.AsyncClient: Configured client
    """
    if verify_ssl:
        # Use certifi for certificate verification
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        return httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            limits=limits,
            headers={
                'User-Agent': 'OpenDeepResearch/1.0 (https://github.com/langchain-ai/open_deep_research)'
            }
        )
    else:
        # Create client that bypasses SSL verification (for development/testing only)
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        return httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            limits=limits,
            verify=False,
            headers={
                'User-Agent': 'OpenDeepResearch/1.0 (https://github.com/langchain-ai/open_deep_research)'
            }
        )

async def safe_http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, Any]] = None,
    verify_ssl: bool = True,
    timeout: int = 30,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Make a safe HTTP request with retry logic and SSL error handling.
    
    Args:
        url (str): URL to request
        method (str): HTTP method
        headers (Optional[Dict[str, str]]): Request headers
        data (Optional[Dict[str, Any]]): Request data
        verify_ssl (bool): Whether to verify SSL certificates
        timeout (int): Timeout in seconds
        max_retries (int): Maximum number of retries
        
    Returns:
        Dict[str, Any]: Response data
        
    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            async with create_aiohttp_session(verify_ssl, timeout) as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers) as response:
                        response.raise_for_status()
                        return {
                            "status": response.status,
                            "headers": dict(response.headers),
                            "text": await response.text(),
                            "json": await response.json() if response.headers.get('content-type', '').startswith('application/json') else None
                        }
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, data=data) as response:
                        response.raise_for_status()
                        return {
                            "status": response.status,
                            "headers": dict(response.headers),
                            "text": await response.text(),
                            "json": await response.json() if response.headers.get('content-type', '').startswith('application/json') else None
                        }
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                    
        except (aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError) as e:
            if attempt == max_retries - 1:
                # Last attempt failed, try without SSL verification if it was enabled
                if verify_ssl:
                    logging.warning(f"SSL verification failed for {url}, retrying without SSL verification")
                    return await safe_http_request(url, method, headers, data, False, timeout, 1)
                else:
                    raise Exception(f"HTTP request failed after {max_retries} attempts: {str(e)}")
            else:
                # Wait before retry with exponential backoff
                wait_time = (2 ** attempt) + (0.1 * attempt)
                logging.warning(f"HTTP request attempt {attempt + 1} failed for {url}: {str(e)}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"HTTP request failed after {max_retries} attempts: {str(e)}")
            else:
                wait_time = (2 ** attempt) + (0.1 * attempt)
                logging.warning(f"HTTP request attempt {attempt + 1} failed for {url}: {str(e)}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)