"""
Input validation utilities for the Open Deep Research project.

This module provides functions for validating inputs to ensure they meet
the requirements of the application and to prevent potential security issues.
"""

import re
from typing import List, Optional, Dict, Any, Union
from urllib.parse import urlparse

def validate_search_query(query: str) -> bool:
    """
    Validate a search query to ensure it's safe and appropriate.
    
    Args:
        query: The search query to validate
        
    Returns:
        True if the query is valid, False otherwise
        
    Raises:
        ValueError: If the query is invalid with a description of the issue
    """
    # Check if query is empty or just whitespace
    if not query or query.isspace():
        raise ValueError("Search query cannot be empty")
    
    # Check if query is too long (most search engines have limits)
    if len(query) > 1000:
        raise ValueError("Search query is too long (max 1000 characters)")
    
    # Check for potentially harmful patterns
    harmful_patterns = [
        r'<script.*?>.*?</script>',  # Basic XSS prevention
        r'DROP\s+TABLE',  # SQL injection prevention
        r'DELETE\s+FROM',  # SQL injection prevention
        r';\s*DROP',  # SQL injection prevention
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError("Search query contains potentially harmful content")
    
    return True

def validate_url(url: str) -> bool:
    """
    Validate a URL to ensure it's properly formatted.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if the URL is valid, False otherwise
        
    Raises:
        ValueError: If the URL is invalid with a description of the issue
    """
    # Check if URL is empty
    if not url:
        raise ValueError("URL cannot be empty")
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Check if scheme and netloc are present
    if not all([parsed.scheme, parsed.netloc]):
        raise ValueError("URL must include scheme (http:// or https://) and domain")
    
    # Check if scheme is http or https
    if parsed.scheme not in ['http', 'https']:
        raise ValueError("URL scheme must be http or https")
    
    return True

def validate_api_key(api_key: str, min_length: int = 10) -> bool:
    """
    Validate an API key to ensure it meets basic requirements.
    
    Args:
        api_key: The API key to validate
        min_length: Minimum length for the API key
        
    Returns:
        True if the API key is valid, False otherwise
        
    Raises:
        ValueError: If the API key is invalid with a description of the issue
    """
    # Check if API key is empty
    if not api_key:
        raise ValueError("API key cannot be empty")
    
    # Check if API key meets minimum length requirement
    if len(api_key) < min_length:
        raise ValueError(f"API key must be at least {min_length} characters long")
    
    return True

def validate_search_api_config(search_api: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate search API configuration parameters based on the specific API.
    
    Args:
        search_api: The search API identifier
        config: The configuration dictionary for the search API
        
    Returns:
        The validated configuration dictionary
        
    Raises:
        ValueError: If the configuration is invalid with a description of the issue
    """
    if not config:
        return {}
    
    # Define validation rules for each search API
    validation_rules = {
        "exa": {
            "max_characters": lambda x: isinstance(x, int) and x > 0,
            "num_results": lambda x: isinstance(x, int) and 1 <= x <= 100,
            "include_domains": lambda x: isinstance(x, list) and all(isinstance(d, str) for d in x),
            "exclude_domains": lambda x: isinstance(x, list) and all(isinstance(d, str) for d in x),
            "subpages": lambda x: isinstance(x, int) and x >= 0
        },
        "tavily": {
            "max_results": lambda x: isinstance(x, int) and 1 <= x <= 20,
            "topic": lambda x: x in ["general", "news", "finance"]
        },
        "arxiv": {
            "load_max_docs": lambda x: isinstance(x, int) and x > 0,
            "get_full_documents": lambda x: isinstance(x, bool),
            "load_all_available_meta": lambda x: isinstance(x, bool)
        },
        "pubmed": {
            "top_k_results": lambda x: isinstance(x, int) and x > 0,
            "email": lambda x: isinstance(x, str) and "@" in x,
            "api_key": lambda x: isinstance(x, str) and len(x) > 10,
            "doc_content_chars_max": lambda x: isinstance(x, int) and x > 0
        },
        "linkup": {
            "depth": lambda x: x in ["standard", "deep"]
        },
        "googlesearch": {
            "max_results": lambda x: isinstance(x, int) and 1 <= x <= 100
        }
    }
    
    # Get validation rules for the specified search API
    api_rules = validation_rules.get(search_api.lower(), {})
    
    # Validate each parameter in the configuration
    validated_config = {}
    for param, value in config.items():
        # Check if parameter is valid for this API
        if param not in api_rules:
            continue
        
        # Check if parameter value is valid
        validation_func = api_rules[param]
        if not validation_func(value):
            raise ValueError(f"Invalid value for {param} in {search_api} configuration")
        
        validated_config[param] = value
    
    # Special validation for Exa: include_domains and exclude_domains cannot be used together
    if search_api.lower() == "exa" and "include_domains" in validated_config and "exclude_domains" in validated_config:
        raise ValueError("include_domains and exclude_domains cannot be used together in Exa configuration")
    
    return validated_config
