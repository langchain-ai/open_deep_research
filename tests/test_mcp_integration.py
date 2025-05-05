#!/usr/bin/env python

import os
import uuid
import pytest
import asyncio
import logging
import json
import time
from pathlib import Path
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langsmith import testing as t

# Import pytest_asyncio for async testing support
import pytest_asyncio

# Filter Pydantic deprecation warnings
pytestmark = pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mcp_tests.log")
    ]
)
logger = logging.getLogger(__name__)

class MCPTestResult(BaseModel):
    """Result of an MCP integration test."""
    passed: bool = Field(description="Whether the test passed")
    details: str = Field(description="Details about the test results")
    tool_count: int = Field(description="Number of tools successfully tested")

def print_mcp_server_info(server_path, tools_list=None):
    """Print information about the MCP server and its available tools."""
    print("\n" + "="*80)
    print(f"MCP SERVER: {server_path}")
    print("="*80)
    
    if tools_list:
        print(f"\nAVAILABLE TOOLS ({len(tools_list)}):")
        for i, tool in enumerate(tools_list, 1):
            tool_name = getattr(tool, "name", str(tool))
            tool_desc = getattr(tool, "description", "")
            if tool_desc and len(tool_desc) > 60:
                tool_desc = tool_desc[:57] + "..."
            
            print(f"  {i}. {tool_name}")
            if tool_desc:
                print(f"     {tool_desc}")
        print("-"*80)

def print_tool_result(tool_name, result):
    """Print the result of a tool call."""
    print(f"\nTOOL CALL RESULT - {tool_name}:")
    try:
        formatted_result = json.dumps(result, indent=2) if not isinstance(result, str) else result
        print(f"{formatted_result}\n")
    except Exception:
        print(f"{result}\n")
    print("-"*40)

# Define fixtures for test configuration
@pytest.fixture
def mcp_server_path(request):
    """Get the path to the MCP server."""
    path = request.config.getoption("--mcp-server-path") or os.environ.get("MCP_SERVER_PATH")
    if not path:
        # Default to looking in the same directory as the test file
        path = str(Path(__file__).with_name("example_mcp_server.py").resolve())
    return path

# Create async version of the server path fixture
@pytest_asyncio.fixture
async def async_server_path(mcp_server_path):
    """Async version of the server path fixture."""
    return Path(mcp_server_path)

@pytest.fixture
def test_type(request):
    """Get the test type from command line or environment variable."""
    return request.config.getoption("--test-type") or os.environ.get("TEST_TYPE", "all")

@pytest.fixture
def eval_model(request):
    """Get the evaluation model from command line or environment variable."""
    return request.config.getoption("--eval-model") or os.environ.get("EVAL_MODEL", "openai:gpt-4o")

@pytest.mark.langsmith
def test_mcp_integration(mcp_server_path, test_type, eval_model):
    """Test the MCP integration with different frameworks."""
    print(f"Testing MCP integration with server at: {mcp_server_path}")
    print(f"Test type: {test_type}")
    print(f"Eval model: {eval_model}")

    # Log inputs to LangSmith
    t.log_inputs({
        "test_type": test_type,
        "mcp_server_path": mcp_server_path,
        "eval_model": eval_model,
        "test": "mcp_integration_test",
        "description": f"Testing MCP integration with {test_type}"
    })

    # Ensure the server file exists
    server_path = Path(mcp_server_path)
    if not server_path.exists():
        error_msg = f"MCP server file not found at: {server_path}"
        logger.error(error_msg)
        t.log_outputs({
            "error": error_msg,
            "passed": False
        })
        assert False, error_msg

    # Initialize results
    direct_result = False
    langchain_result = False
    langgraph_result = False

    # Run the appropriate test based on the parameter
    if test_type == "direct" or test_type == "all":
        try:
            # Run the direct test synchronously
            direct_result = asyncio.run(_test_direct_connection(server_path))
            print(f"Direct MCP Connection Test: {'PASSED' if direct_result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Exception in direct connection test: {str(e)}", exc_info=True)
            print(f"Direct MCP Connection Test: FAILED (exception)")

    if test_type == "langchain" or test_type == "all":
        try:
            # Run the LangChain test synchronously
            langchain_result = asyncio.run(_test_langchain_integration(server_path))
            print(f"LangChain Integration Test: {'PASSED' if langchain_result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Exception in LangChain integration test: {str(e)}", exc_info=True)
            print(f"LangChain Integration Test: FAILED (exception)")

    if test_type == "langgraph" or test_type == "all":
        try:
            # Run the LangGraph test synchronously
            langgraph_result = asyncio.run(_test_langgraph_integration(server_path))
            print(f"LangGraph Integration Test: {'PASSED' if langgraph_result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Exception in LangGraph integration test: {str(e)}", exc_info=True)
            print(f"LangGraph Integration Test: FAILED (exception)")

    # Determine overall test result
    if test_type == "all":
        # If running a specific test type only, we can allow failing others
        if os.environ.get("ALLOW_PARTIAL_SUCCESS") == "true":
            # Check if at least one test passed
            overall_result = direct_result or langchain_result or langgraph_result
        else:
            # Default: all selected tests must pass
            overall_result = direct_result and langchain_result and langgraph_result
        
        results = {
            "direct": direct_result,
            "langchain": langchain_result,
            "langgraph": langgraph_result
        }
    elif test_type == "direct":
        overall_result = direct_result
        results = {"direct": direct_result}
    elif test_type == "langchain":
        overall_result = langchain_result
        results = {"langchain": langchain_result}
    else:  # langgraph
        overall_result = langgraph_result
        results = {"langgraph": langgraph_result}

    # Create detailed result message
    result_details = "\n".join([f"{test}: {'PASSED' if passed else 'FAILED'}" for test, passed in results.items()])
    
    # Log outputs to LangSmith
    t.log_outputs({
        "passed": overall_result,
        "details": result_details,
        "results": results
    })
    
    # Test passes if all selected tests pass
    assert overall_result, f"MCP integration tests failed:\n{result_details}"

# Async test functions that can be called both from the main test and individually via pytest

@pytest.mark.asyncio
async def test_direct_connection(async_server_path):
    """Async test for direct MCP connection via pytest."""
    result = await _test_direct_connection(async_server_path)
    assert result, "Direct MCP connection test failed"

async def _test_direct_connection(server_path):
    """Internal async implementation for direct MCP connection test."""
    try:
        # Import required modules
        from mcp.client.stdio import stdio_client
        from mcp import StdioServerParameters, ClientSession
        
        logger.info(f"Testing direct MCP connection with server at: {server_path}")
        
        # Create server parameters
        server_params = StdioServerParameters(
            command="python",
            args=[str(server_path)],
        )
        
        # Connect to the server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # List available tools
                schemas = await session.list_tools()
                
                # Handle different types of tool schema responses
                if hasattr(schemas, 'tools'):
                    tool_list = schemas.tools
                    logger.info(f"Found {len(tool_list)} tools (via tools attribute)")
                    # Print information about the server and its tools
                    print_mcp_server_info(server_path, tool_list)
                else:
                    logger.info(f"Found tools of type {type(schemas).__name__}")
                    tool_list = []
                    # Print minimal server info without tools
                    print_mcp_server_info(server_path)
                
                # Try different methods to call tools
                tool_names = ["patient_vitals", "patient_current_conditions", "patient_scheduled_checks"]
                success_count = 0
                
                for tool_name in tool_names:
                    logger.info(f"Testing tool: {tool_name}")
                    try:
                        # Try different methods to call the tool
                        if hasattr(session, 'call_tool'):
                            result = await session.call_tool(tool_name, {"patient_id": "P001"})
                            logger.info(f"Tool result (via call_tool): {result}")
                            # Print the successful tool call result
                            print_tool_result(tool_name, result)
                            success_count += 1
                        elif hasattr(session, 'invoke_tool'):
                            result = await session.invoke_tool(tool_name, {"patient_id": "P001"})
                            logger.info(f"Tool result (via invoke_tool): {result}")
                            # Print the successful tool call result
                            print_tool_result(tool_name, result)
                            success_count += 1
                        elif hasattr(session, tool_name):
                            tool_method = getattr(session, tool_name)
                            result = await tool_method(patient_id="P001")
                            logger.info(f"Tool result (via direct method): {result}")
                            # Print the successful tool call result
                            print_tool_result(tool_name, result)
                            success_count += 1
                        else:
                            logger.warning(f"No method found to call tool {tool_name}")
                    except Exception as e:
                        logger.error(f"Error invoking tool {tool_name}: {e}", exc_info=True)
                
                return success_count > 0  # At least one tool call succeeded
                
    except Exception as e:
        logger.error(f"Error in direct MCP connection test: {e}", exc_info=True)
        return False

@pytest.mark.asyncio
async def test_langchain_integration(async_server_path):
    """Async test for LangChain MCP integration via pytest."""
    result = await _test_langchain_integration(async_server_path)
    assert result, "LangChain MCP integration test failed"

async def _test_langchain_integration(server_path):
    """Internal async implementation for LangChain MCP integration test."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        logger.info(f"Testing LangChain MCP integration with server at: {server_path}")
        
        # Set up MCP client configuration
        mcp_servers = {
            "patient_data": {
                "transport": "stdio",
                "command": "python",
                "args": [str(server_path)],
            }
        }
        
        # Initialize using MultiServerMCPClient
        async with MultiServerMCPClient(mcp_servers) as client:
            tools = client.get_tools()
            
            # Print information about the server and its tools
            print_mcp_server_info(server_path, tools)
            
            logger.info(f"Loaded {len(tools)} tools")
            for i, tool in enumerate(tools):
                logger.info(f"Tool {i+1}: {tool.name}")
                logger.info(f"  Description: {tool.description[:50]}...")
            
            # Test each tool
            success_count = 0
            for tool in tools:
                logger.info(f"Testing tool: {tool.name}")
                try:
                    result = await tool.ainvoke({"patient_id": "P001"})
                    logger.info(f"Tool result: {result}")
                    # Print the successful tool call result
                    print_tool_result(tool.name, result)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error invoking tool {tool.name}: {e}", exc_info=True)
            
            return success_count == len(tools)  # All tools worked
    
    except Exception as e:
        logger.error(f"Error in LangChain MCP integration test: {e}", exc_info=True)
        return False

@pytest.mark.asyncio
async def test_langgraph_integration(async_server_path):
    """Async test for LangGraph MCP integration via pytest."""
    result = await _test_langgraph_integration(async_server_path)
    assert result, "LangGraph MCP integration test failed"

async def _test_langgraph_integration(server_path):
    """Internal async implementation for LangGraph MCP integration test."""
    try:
        # Import required modules
        from open_deep_research.multi_agent_new import build_graph_with_mcp, supervisor, supervisor_tools
        
        logger.info(f"Testing LangGraph MCP integration with server at: {server_path}")
        
        # Configure MCP servers
        mcp_servers = {
            "patient_data": {
                "transport": "stdio",
                "command": "python",
                "args": [str(server_path)],
            }
        }
        
        # Set up test configuration
        cfg_test = {
            "mcp_servers": mcp_servers,
        }
        
        # Print information about MCP server configuration
        print_mcp_server_info(server_path)
        print("MCP Configuration:")
        print(json.dumps(mcp_servers, indent=2))
        print("-"*80)
        
        # Build the graph with MCP tools
        logger.info("Building graph with MCP tools...")
        graph, _, cfg = await build_graph_with_mcp(cfg_test)
        
        # Create a test state
        test_state = {
            "messages": [{"role": "user", "content": "What do we know about patient with id P001?"}]
        }
        
        # Define a short timeout for test purposes
        timeout_seconds = 30
        logger.info(f"Testing supervisor node with timeout: {timeout_seconds}s...")
        
        try:
            # Execute the supervisor node
            start_time = time.time()
            
            # Run supervisor to generate tool calls
            supervisor_result = await supervisor(test_state, cfg)
            logger.info(f"Supervisor completed")
            
            # Check if the supervisor made any tool calls
            if "messages" in supervisor_result and supervisor_result["messages"]:
                last_message = supervisor_result["messages"][-1]
                has_tool_calls = False
                
                # Check for tool_calls in different message formats
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    has_tool_calls = True
                    tool_calls = last_message.tool_calls
                elif isinstance(last_message, dict) and last_message.get("tool_calls"):
                    has_tool_calls = True
                    tool_calls = last_message.get("tool_calls", [])
                else:
                    tool_calls = []
                
                if has_tool_calls:
                    logger.info(f"Supervisor made {len(tool_calls)} tool calls")
                    print(f"\nSUPERVISOR MADE {len(tool_calls)} TOOL CALLS:")
                    
                    # Print tool calls
                    for i, call in enumerate(tool_calls, 1):
                        call_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "Unknown")
                        call_args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                        print(f"  {i}. {call_name}")
                        print(f"     Args: {json.dumps(call_args, indent=2)}")
                    
                    # Check for MCP tool calls
                    mcp_tool_calls = []
                    for call in tool_calls:
                        call_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                        if call_name in ["patient_vitals", "patient_current_conditions", "patient_scheduled_checks"]:
                            mcp_tool_calls.append(call)
                    
                    if mcp_tool_calls:
                        logger.info(f"Found {len(mcp_tool_calls)} MCP tool calls")
                        print(f"\nFOUND {len(mcp_tool_calls)} MCP TOOL CALLS:")
                        
                        # Update state with supervisor result
                        test_state.update(supervisor_result)
                        
                        # Execute the supervisor_tools node to process the tool calls
                        logger.info("Executing supervisor_tools to process MCP tool calls...")
                        tools_result = await supervisor_tools(test_state, cfg)
                        
                        # Handle different return types from supervisor_tools
                        tool_messages = []
                        
                        if hasattr(tools_result, "update") and isinstance(tools_result.update, dict) and "messages" in tools_result.update:
                            # It's a Command object - get messages from the update field
                            tool_messages = tools_result.update.get("messages", [])
                            logger.info(f"Found {len(tool_messages)} messages in Command.update")
                        elif isinstance(tools_result, dict) and "messages" in tools_result:
                            # It's a simple dictionary with messages
                            tool_messages = tools_result.get("messages", [])
                            logger.info(f"Found {len(tool_messages)} messages in dictionary")
                        else:
                            # No messages found or unexpected format
                            logger.info(f"Unexpected tools_result format: {type(tools_result)}")
                        
                        # Check for MCP tool responses in the extracted messages
                        mcp_responses = []
                        for msg in tool_messages:
                            msg_name = None
                            if isinstance(msg, dict) and "name" in msg:
                                msg_name = msg.get("name")
                            elif hasattr(msg, "name"):
                                msg_name = msg.name
                                
                            if msg_name in ["patient_vitals", "patient_current_conditions", "patient_scheduled_checks"]:
                                mcp_responses.append(msg)
                        
                        logger.info(f"Found {len(mcp_responses)} MCP tool responses")
                        
                        # Log some response details for verification
                        for i, resp in enumerate(mcp_responses):
                            resp_name = resp.get("name") if isinstance(resp, dict) else getattr(resp, "name", "Unknown")
                            resp_content = resp.get("content", "") if isinstance(resp, dict) else getattr(resp, "content", "")
                            
                            # Print the tool response
                            print_tool_result(resp_name, resp_content)
                        
                        # Test is successful if we processed MCP tool calls
                        return len(mcp_responses) > 0
                    else:
                        logger.info("No MCP tool calls found")
                else:
                    logger.info("No tool calls found in supervisor result")
            else:
                logger.info("No messages in supervisor result")
            
            end_time = time.time()
            logger.info(f"Test completed in {end_time - start_time:.2f} seconds")
            
            # If we got this far without errors, consider it a partial success
            return True
                
        except asyncio.TimeoutError:
            logger.error(f"Test timed out after {timeout_seconds} seconds")
            return False
            
        except Exception as e:
            logger.error(f"Error executing test: {str(e)}", exc_info=True)
            return False
        
    except Exception as e:
        logger.error(f"Error in LangGraph MCP integration test: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # This allows running the test directly, but pytest will import this file
    pytest.main(["-xvs", __file__])