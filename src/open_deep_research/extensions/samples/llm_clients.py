import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime
try:
    from langsmith import traceable
except Exception:
    # If langsmith is unavailable or fails to connect, provide a no-op decorator
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Import LangChain and provider-specific packages
from langchain_core.messages import HumanMessage, SystemMessage, ChatMessage
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


# API keys — only Azure OpenAI and Gemini are supported
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Token limits
AZURE_MAX_TOKENS = 30000
GOOGLE_MAX_OUTPUT_TOKENS = 30000

# Get the current date in various formats for the system prompt
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month
CURRENT_DAY = datetime.now().day
ONE_YEAR_AGO = datetime.now().replace(year=datetime.now().year - 1).strftime("%Y-%m-%d")
YTD_START = f"{CURRENT_YEAR}-01-01"


# Model configurations — only Azure OpenAI and Gemini
MODEL_CONFIGS = {
    "azure": {
        "available_models": [
            "gpt41",
            "gpt-4o",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-35-turbo",
        ],
        "default_model": AZURE_OPENAI_DEPLOYMENT or "gpt41",
        "requires_api_key": AZURE_OPENAI_API_KEY,
    },
    "gemini": {
        "available_models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
        ],
        "default_model": "gemini-2.5-flash",
        "requires_api_key": GEMINI_API_KEY,
    },
}

# Base system prompt template - will be formatted with current date information
SYSTEM_PROMPT_TEMPLATE = """
<intro>
You excel at the following tasks:
1. Information gathering, fact-checking, and documentation
2. Data processing, analysis, and visualization
3. Writing multi-chapter articles and in-depth research reports
4. Creating websites, applications, and tools
5. Using programming to solve various problems beyond development
6. Various tasks that can be accomplished using computers and the internet
7. IMPORTANT: The current date is {current_date}. Always use this as your reference instead of datetime.now().
</intro>

<date_information>
Current date: {current_date}
Current year: {current_year}
Current month: {current_month}
Current day: {current_day}
One year ago: {one_year_ago}
Year-to-date start: {ytd_start}
</date_information>

<requirements>
When writing code, your code MUST:
1. Start with installing necessary packages (e.g., '!pip install matplotlib pandas')
2. Include robust error handling for data retrieval and processing
3. Print sample data to validate successful retrieval
4. Properly handle data structures based on the returned format:
   - For multi-level dataframes: Use appropriate indexing like df['Close'] or df.loc[:, ('Price', 'Close')]
   - For single-level dataframes: Access columns directly
5. If asked to create visualizations with professional styling including:
   - Clear title and axis labels
   - Grid lines for readability
   - Appropriate date formatting on x-axis using matplotlib.dates
   - Legend when plotting multiple series
6. Include data validation to check for:
   - Dataset sizes and date ranges
   - Missing values (NaN) and their handling
   - Data types and any necessary conversions
7. Implement appropriate data transformations like:
   - Normalizing prices to a common baseline
   - Calculating moving averages or other indicators
   - Computing ratios or correlations between assets
8. IMPORTANT: When fetching date-sensitive data:
   - DO NOT use datetime.now() in your code
   - Instead, use these fixed dates: current="{current_date}", year_start="{ytd_start}", year_ago="{one_year_ago}"
</requirements>
"""

# Error correction prompt addition when code execution fails
ERROR_CORRECTION_PROMPT = """
<error_correction>
The previous code failed to execute properly. I'm providing the error logs below.
Please fix the code to address these issues and ensure it runs correctly:

ERROR LOGS:
{error_logs}

Common issues to check:
1. Date handling issues - Use explicit date ranges (e.g., '{ytd_start}' instead of datetime.now())
2. Data structure validation - Verify the expected structure of returned data
3. Library compatibility - Ensure all functions used are available in the imported libraries
4. Error handling - Add more robust try/except blocks
</error_correction>
"""


def get_available_providers():
    """Returns a list of available providers based on configured API keys."""
    available_providers = []
    for provider, config in MODEL_CONFIGS.items():
        if config.get("requires_api_key"):
            available_providers.append(provider)
    return available_providers


def get_llm_client(provider, model_name=None):
    """
    Get the appropriate LLM client based on provider and model name.
    Only Azure OpenAI and Gemini are supported.

    Args:
        provider: The provider name ('azure' or 'gemini')
        model_name: The model name (optional, uses default if not provided)

    Returns:
        A synchronous LangChain chat model client for the specified provider
    """
    print(f"[LLM CLIENT] get_llm_client called -> provider='{provider}', model_name='{model_name}'")

    if provider == "azure":
        if not AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY is not set in environment")
        if not AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set in environment")
        deployment = model_name or AZURE_OPENAI_DEPLOYMENT or "gpt-4o"
        print(f"Using AzureChatOpenAI for deployment: {deployment}")
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            max_tokens=AZURE_MAX_TOKENS,
        )

    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment")
        if not model_name:
            model_name = MODEL_CONFIGS["gemini"]["default_model"]
        print(f"Using ChatGoogleGenerativeAI for {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            max_output_tokens=GOOGLE_MAX_OUTPUT_TOKENS,
        )

    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Only 'azure' and 'gemini' are supported."
        )


async def get_async_llm_client(provider, model_name=None):
    """
    Get an asynchronous LLM client for the given provider.
    Only Azure OpenAI and Gemini are supported.

    Args:
        provider: The provider name ('azure' or 'gemini')
        model_name: The model name (optional, uses default if not provided)

    Returns:
        An async LangChain chat model client for the specified provider
    """
    import logging

    logger = logging.getLogger(__name__)
    print(f"[LLM CLIENT] get_async_llm_client called -> provider='{provider}', model_name='{model_name}'")
    logger.info(
        f"[get_async_llm_client] Requested provider: {provider}, model: {model_name or 'default'}"
    )

    # Get the default model from MODEL_CONFIGS if not specified
    if not model_name and provider in MODEL_CONFIGS:
        model_name = MODEL_CONFIGS[provider]["default_model"]
        logger.info(
            f"[get_async_llm_client] Using default model for {provider}: {model_name}"
        )

    if provider == "azure":
        if not AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY is not set in environment")
        if not AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set in environment")
        deployment = model_name or AZURE_OPENAI_DEPLOYMENT or "gpt-4o"
        logger.info(
            f"[get_async_llm_client] Creating async AzureChatOpenAI client with deployment {deployment}"
        )
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            max_tokens=AZURE_MAX_TOKENS,
        )

    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment")
        if not model_name:
            model_name = MODEL_CONFIGS["gemini"]["default_model"]
        logger.info(
            f"[get_async_llm_client] Creating async ChatGoogleGenerativeAI client with model {model_name}"
        )
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            max_output_tokens=GOOGLE_MAX_OUTPUT_TOKENS,
        )

    else:
        error_msg = (
            f"Unsupported provider: '{provider}'. Only 'azure' and 'gemini' are supported."
        )
        logger.error(f"[get_async_llm_client] {error_msg}")
        raise ValueError(error_msg)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_model_response(llm, system_prompt: str, user_prompt: str, config=None):
    """
    Get a response from an LLM using LangChain.

    Args:
        llm: The LangChain chat model client
        system_prompt: The system prompt to use
        user_prompt: The user prompt to use
        config: Optional config object that may contain LangSmith trace information

    Returns:
        The model's response as a string
    """
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        model_name = getattr(llm, "model_name", None)
        if model_name is None:
            model_name = getattr(llm, "model", "unknown model")

        print(f"Sending messages to {model_name}...")

        response = llm.invoke(messages, config=config)

        if isinstance(response, str):
            return response
        else:
            return response.content
    except Exception as e:
        print(f"[Model API ERROR] {str(e)}")
        raise


def get_formatted_system_prompt():
    """
    Format the system prompt template with current date information.

    Returns:
        The formatted system prompt string
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        current_month=CURRENT_MONTH,
        current_day=CURRENT_DAY,
        one_year_ago=ONE_YEAR_AGO,
        ytd_start=YTD_START,
    )
