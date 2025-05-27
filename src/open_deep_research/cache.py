"""
Caching utilities for the Open Deep Research project.

This module provides caching mechanisms to avoid redundant API calls
and improve performance when making repeated searches.
"""

import os
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from functools import wraps

# Get module-level logger
logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".open_deep_research" / "cache"

class SearchCache:
    """
    Cache for search results to avoid redundant API calls.
    
    This class provides methods to store and retrieve search results from a
    file-based cache. It uses a simple JSON file for each cache entry, with
    the filename derived from a hash of the search query and parameters.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, max_age_hours: int = 24):
        """
        Initialize the search cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cache entries in hours
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.max_age_seconds = max_age_hours * 3600
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Initialized search cache in {self.cache_dir}")
    
    def _get_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key from a search query and parameters.
        
        Args:
            query: The search query
            params: Additional parameters for the search
            
        Returns:
            A hash string to use as the cache key
        """
        # Create a string representation of the query and parameters
        cache_str = f"{query}_{json.dumps(params, sort_keys=True)}"
        
        # Generate a hash of the string
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, query: str, params: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Get search results from the cache.
        
        Args:
            query: The search query
            params: Additional parameters for the search
            
        Returns:
            The cached search results, or None if not found or expired
        """
        params = params or {}
        cache_key = self._get_cache_key(query, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Check if cache file exists
        if not cache_file.exists():
            logger.debug(f"Cache miss for query: {query}")
            return None
        
        try:
            # Read cache file
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache entry is expired
            cache_time = cache_data.get('timestamp', 0)
            current_time = time.time()
            
            if current_time - cache_time > self.max_age_seconds:
                logger.debug(f"Cache expired for query: {query}")
                return None
            
            logger.debug(f"Cache hit for query: {query}")
            return cache_data.get('results')
            
        except Exception as e:
            logger.warning(f"Error reading cache file: {e}")
            return None
    
    def set(self, query: str, results: List[Dict[str, Any]], params: Dict[str, Any] = None) -> None:
        """
        Store search results in the cache.
        
        Args:
            query: The search query
            results: The search results to cache
            params: Additional parameters for the search
        """
        params = params or {}
        cache_key = self._get_cache_key(query, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            # Create cache entry
            cache_data = {
                'query': query,
                'params': params,
                'results': results,
                'timestamp': time.time()
            }
            
            # Write cache file
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logger.debug(f"Cached results for query: {query}")
            
        except Exception as e:
            logger.warning(f"Error writing cache file: {e}")
    
    def clear(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clear expired cache entries.
        
        Args:
            max_age_hours: Maximum age of cache entries in hours
                           If None, uses the default max_age_hours
        
        Returns:
            Number of cache entries cleared
        """
        max_age_seconds = max_age_hours * 3600 if max_age_hours else self.max_age_seconds
        current_time = time.time()
        cleared_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                # Read cache file
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache entry is expired
                cache_time = cache_data.get('timestamp', 0)
                
                if current_time - cache_time > max_age_seconds:
                    # Remove expired cache file
                    os.remove(cache_file)
                    cleared_count += 1
                    
            except Exception as e:
                logger.warning(f"Error clearing cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {cleared_count} expired cache entries")
        return cleared_count

# Global cache instance
_cache = SearchCache()

def cached_search(func: Callable) -> Callable:
    """
    Decorator to cache search results.
    
    This decorator wraps a search function to cache its results and return
    cached results when available.
    
    Args:
        func: The search function to wrap
        
    Returns:
        Wrapped function that uses the cache
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract query from args or kwargs
        if len(args) > 0 and isinstance(args[0], (str, list)):
            queries = args[0]
        else:
            queries = kwargs.get('search_queries', kwargs.get('queries', []))
        
        # Handle both string and list queries
        if isinstance(queries, str):
            queries = [queries]
        
        # Extract parameters
        params = {k: v for k, v in kwargs.items() if k not in ['search_queries', 'queries']}
        
        # Check if all queries are in cache
        all_cached_results = []
        all_in_cache = True
        
        for query in queries:
            cached_results = _cache.get(query, params)
            if cached_results is None:
                all_in_cache = False
                break
            all_cached_results.append({'query': query, 'results': cached_results})
        
        # If all queries are in cache, return cached results
        if all_in_cache and all_cached_results:
            logger.info(f"Using cached results for {len(queries)} queries")
            return all_cached_results
        
        # Otherwise, call the original function
        results = await func(*args, **kwargs)
        
        # Cache the results
        if isinstance(results, list):
            for result in results:
                if 'query' in result and 'results' in result:
                    _cache.set(result['query'], result['results'], params)
        
        return results
    
    return wrapper

def get_cache() -> SearchCache:
    """
    Get the global cache instance.
    
    Returns:
        The global SearchCache instance
    """
    return _cache
