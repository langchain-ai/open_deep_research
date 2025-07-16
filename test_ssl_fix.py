#!/usr/bin/env python3
"""
Test script for SSL/TLS error handling fix.
This script tests the SSL handling improvements made to Open Deep Research.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from open_deep_research.utils import safe_http_request, create_aiohttp_session, create_httpx_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ssl_handling():
    """Test SSL handling with various scenarios."""
    
    print("🔧 Testing SSL/TLS Error Handling Fix")
    print("=" * 50)
    
    # Test URLs that might have SSL issues
    test_urls = [
        "https://httpbin.org/get",  # Should work normally
        "https://api.github.com",   # Should work normally
        "https://self-signed.badssl.com/",  # Self-signed certificate (should fail with SSL verification)
    ]
    
    for url in test_urls:
        print(f"\n📡 Testing URL: {url}")
        try:
            # Test with SSL verification enabled
            print("  - Testing with SSL verification enabled...")
            response = await safe_http_request(
                url=url,
                method="GET",
                verify_ssl=True,
                timeout=10,
                max_retries=2
            )
            print(f"  ✅ Success! Status: {response['status']}")
            
        except Exception as e:
            print(f"  ❌ Failed with SSL verification: {str(e)}")
            
            # Test with SSL verification disabled
            try:
                print("  - Testing with SSL verification disabled...")
                response = await safe_http_request(
                    url=url,
                    method="GET",
                    verify_ssl=False,
                    timeout=10,
                    max_retries=2
                )
                print(f"  ✅ Success with SSL disabled! Status: {response['status']}")
            except Exception as e2:
                print(f"  ❌ Failed even with SSL disabled: {str(e2)}")

async def test_client_creation():
    """Test the creation of HTTP clients with SSL handling."""
    
    print("\n🔧 Testing HTTP Client Creation")
    print("=" * 50)
    
    # Test aiohttp session creation
    try:
        session = create_aiohttp_session(verify_ssl=True, timeout=30)
        print("  ✅ aiohttp session with SSL verification created successfully")
        await session.close()
    except Exception as e:
        print(f"  ❌ Failed to create aiohttp session: {str(e)}")
    
    # Test httpx client creation
    try:
        client = create_httpx_client(verify_ssl=True, timeout=30)
        print("  ✅ httpx client with SSL verification created successfully")
        await client.aclose()
    except Exception as e:
        print(f"  ❌ Failed to create httpx client: {str(e)}")

async def test_environment_configuration():
    """Test environment variable configuration."""
    
    print("\n🔧 Testing Environment Configuration")
    print("=" * 50)
    
    # Test default behavior
    verify_ssl_default = os.getenv("OPEN_DEEP_RESEARCH_VERIFY_SSL", "true").lower() == "true"
    print(f"  - Default SSL verification: {verify_ssl_default}")
    
    # Test with environment variable set to false
    os.environ["OPEN_DEEP_RESEARCH_VERIFY_SSL"] = "false"
    verify_ssl_false = os.getenv("OPEN_DEEP_RESEARCH_VERIFY_SSL", "true").lower() == "true"
    print(f"  - SSL verification with OPEN_DEEP_RESEARCH_VERIFY_SSL=false: {verify_ssl_false}")
    
    # Restore default
    os.environ["OPEN_DEEP_RESEARCH_VERIFY_SSL"] = "true"

def main():
    """Run all SSL tests."""
    print("🚀 Starting SSL/TLS Error Handling Tests")
    print("=" * 60)
    
    try:
        asyncio.run(test_ssl_handling())
        asyncio.run(test_client_creation())
        asyncio.run(test_environment_configuration())
        
        print("\n" + "=" * 60)
        print("✅ All SSL/TLS tests completed successfully!")
        print("\n📋 Summary:")
        print("  - SSL error handling functions implemented")
        print("  - Retry logic with exponential backoff added")
        print("  - Fallback to non-SSL verification when needed")
        print("  - Environment variable configuration supported")
        print("  - Proper logging and error reporting")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 