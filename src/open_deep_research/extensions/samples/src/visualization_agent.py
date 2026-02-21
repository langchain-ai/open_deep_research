"""
Visualization Agent for Deep Research

This module implements a specialized agent for generating visualizations based on research data:
- Takes research results from the SearchAgent
- Determines what visualizations would be appropriate
- Generates and executes code to create visualizations
- Returns visualization metadata for inclusion in reports
"""

import os
import json
import logging
import traceback
import base64
import re
from typing import Dict, List, Any, Optional
import asyncio
import uuid
import time # Import time module

class VisualizationAgent:
    """
    Specialized agent for creating data visualizations from research results.
    
    This agent analyzes research data, determines appropriate visualizations,
    and generates code to create them using the E2B code interpreter.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Visualization Agent.
        
        Args:
            config: Configuration object containing visualization settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = "visualizations"
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    async def determine_visualization_needs(self, search_result):
        """
        Analyze search results to determine if visualizations would be helpful
        and what types of visualizations would be appropriate.
        
        Args:
            search_result: The results from a search task
            
        Returns:
            Dict containing visualization recommendations or None if not needed
        """
        # Import necessary functions
        import os
        from llm_clients import get_async_llm_client
        
        try:
            # Get configuration - respecting user's model selection
            # Using Claude Sonnet 4 as fallback since it's the best coding model available
            if self.config is not None:
                from src.graph import get_configurable
                configurable = get_configurable(self.config)
                provider = configurable.llm_provider
                model = configurable.llm_model
            else:
                # Try environment variables first
                provider = os.environ.get("LLM_PROVIDER")
                model = os.environ.get("LLM_MODEL")
                
                if not provider:
                    provider = "azure"
                # model will use default from MODEL_CONFIGS if None
            
            # Get async LLM client
            llm = await get_async_llm_client(provider, model)

            # Extract content from search result
            content = search_result.get("content", "") # Corrected: Use "content" key
            if isinstance(content, list): # Should not be a list if coming from 'content' key, but defensive
                content = "\n".join(content)
                
            # Get query and subtask info
            # search_result["query"] is the actual query string, not a dictionary
            query = search_result.get("query", "") if isinstance(search_result.get("query"), str) else ""
            subtask_name = query  # Use the query as the subtask name for now

            # Create system prompt for visualization analysis
            system_prompt = """
            You are a critical data visualization expert. Your task is to analyze research content 
            and determine IF AND ONLY IF visualizations would enhance understanding of the research content. 
            Only recommend visualizations if quantitative or structured data suitable 
            for charting (trends, comparisons, distributions) is clearly present and a visualization 
            would add value beyond the text.
            
            Return your analysis as a JSON object with the following structure:
            {
                "visualization_needed": true/false, // MUST be true ONLY if visualization adds significant value
                "rationale": "Brief explanation of why visualization is or IS NOT needed",
                "visualization_types": [ // Include ONLY if visualization_needed is true
                    {
                        "type": "chart_type", // e.g., bar_chart, line_chart, pie_chart
                        "description": "Description of what this visualization would show",
                        "data_requirements": "Description of specific data needed (e.g., market share numbers, sales figures over time)"
                    }
                ]
            }
            
            **Be critical:** If the data is sparse, qualitative, or easily understood from the text, 
            state that visualization is NOT needed (`visualization_needed: false`).
            """
            
            # Create user message with content to analyze
            user_message = f"""
            Analyze the following research content about "{subtask_name}" and determine if visualizations 
            would enhance understanding. Be critical.
            
            QUERY: {query}
            
            RESEARCH CONTENT:
            {content[:4000]}  # Limit content length
            
            Should this content include data visualizations? If so, what specific types would be most effective? 
            If not, explain why.
            """
            
            # Format messages for the LLM client
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Import the visualization needs function schema
            viz_needs_function = {
                "name": "determine_visualization_needs",
                "description": "Determine if visualizations would enhance research content and what types would be appropriate",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "visualization_needed": {
                            "type": "boolean",
                            "description": "Whether visualizations would enhance understanding of the research content"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief explanation of why visualization is or isn't needed"
                        },
                        "visualization_types": {
                            "type": "array",
                            "description": "List of recommended visualization types",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "description": "Type of visualization (e.g., bar_chart, line_chart, pie_chart, scatter_plot, map, table)"
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Description of what this visualization would show"
                                    },
                                    "data_requirements": {
                                        "type": "string",
                                        "description": "Description of data required for this visualization"
                                    }
                                },
                                "required": ["type", "description", "data_requirements"]
                            }
                        }
                    },
                    "required": ["visualization_needed", "rationale"]
                }
            }
            
            # Bind the visualization needs function
            langchain_tools = [{"type": "function", "function": viz_needs_function}]
            
            # Handle tool choice format based on provider (only azure and gemini supported)
            # Normalize provider to string in case it's an LLMProvider enum
            provider_str = provider.value if hasattr(provider, "value") else str(provider)
            if provider_str == "gemini":
                # Use string format for Google Generative AI / Gemini
                tool_choice = "determine_visualization_needs"
            else:
                # Azure OpenAI uses standard OpenAI-compatible dictionary format
                tool_choice = {"type": "function", "function": {"name": "determine_visualization_needs"}}
            
            llm_with_tool = llm.bind_tools(tools=langchain_tools, tool_choice=tool_choice)
            
            # Call LLM API with function calling
            self.logger.info(f"[VisualizationAgent] Making tool call to determine visualization needs...")
            response = await llm_with_tool.ainvoke(messages)
            self.logger.info(f"[VisualizationAgent] Raw response: {response}")
            
            # Process the response based on LLM provider
            function_args = None
            
            # Attempt 1: Standard tool calls format (OpenAI style)
            try:
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        if isinstance(tool_call, dict) and 'args' in tool_call and isinstance(tool_call['args'], dict):
                            function_args = tool_call['args']
                            self.logger.info(f"[VisualizationAgent] Parsed args from tool_call dict")
                            break
                        elif hasattr(tool_call, 'args') and isinstance(tool_call.args, dict):
                            function_args = tool_call.args
                            self.logger.info(f"[VisualizationAgent] Parsed args from tool_call.args attribute")
                            break
                        elif hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                            raw_args = tool_call.function.arguments
                            if isinstance(raw_args, str):
                                try:
                                    function_args = json.loads(raw_args)
                                    self.logger.info(f"[VisualizationAgent] Parsed args from tool_call.function.arguments (string)")
                                    break
                                except json.JSONDecodeError as json_err:
                                    self.logger.warning(f"[VisualizationAgent] Failed to decode JSON from function.arguments: {json_err}")
                            elif isinstance(raw_args, dict):
                                function_args = raw_args
                                self.logger.info(f"[VisualizationAgent] Parsed args from tool_call.function.arguments (dict)")
                                break
            except Exception as e:
                self.logger.warning(f"[VisualizationAgent] Error during standard tool call parsing: {e}")

            # Attempt 2: Direct access to first tool call args (alternative OpenAI/general style)
            if not function_args:
                try:
                    if hasattr(response, 'tool_calls') and len(response.tool_calls) > 0:
                        tool_call = response.tool_calls[0]
                        self.logger.info(f"[VisualizationAgent] Looking at first tool call: {tool_call}")
                        if isinstance(tool_call, dict) and 'args' in tool_call and isinstance(tool_call['args'], dict):
                            function_args = tool_call['args']
                            self.logger.info(f"[VisualizationAgent] Parsed args from first tool_call dict")
                except Exception as e:
                     self.logger.warning(f"[VisualizationAgent] Error during first tool call direct access: {e}")

            # Attempt 3: Anthropic-specific format with content array
            if not function_args:
                try:
                    if hasattr(response, 'content') and isinstance(response.content, list):
                        self.logger.info(f"[VisualizationAgent] Trying Anthropic content format")
                        for item in response.content:
                            if isinstance(item, dict) and item.get('type') == 'tool_use' and 'input' in item and isinstance(item['input'], dict):
                                function_args = item['input']
                                self.logger.info(f"[VisualizationAgent] Parsed input from Anthropic content dict with tool_use type")
                                break
                except Exception as e:
                    self.logger.warning(f"[VisualizationAgent] Error during Anthropic content parsing: {e}")
            
            # Attempt 4: Extract JSON from raw response content string (Common for Gemini/others)
            if not function_args:
                try:
                    if hasattr(response, 'content') and isinstance(response.content, str):
                        self.logger.info(f"[VisualizationAgent] Trying to extract JSON from raw content string")
                        # Look for JSON blocks, potentially within markdown ```json ... ```
                        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, re.DOTALL)
                        if not json_match:
                            # Look for JSON starting with '{' at the beginning of the string or after whitespace
                             json_match = re.search(r'^\s*(\{.*?\})\s*$', response.content, re.DOTALL)
                        
                        if json_match:
                            json_str = json_match.group(1)
                            try:
                                function_args = json.loads(json_str)
                                self.logger.info(f"[VisualizationAgent] Parsed JSON from raw content string")
                            except json.JSONDecodeError as json_err:
                                self.logger.warning(f"[VisualizationAgent] Failed to decode JSON extracted from raw content: {json_err}")
                                self.logger.debug(f"Extracted JSON string: {json_str}")
                        else:
                             self.logger.info(f"[VisualizationAgent] No JSON block found in raw content string.")
                except Exception as e:
                    self.logger.warning(f"[VisualizationAgent] Error during raw content JSON extraction: {e}")

            # Final Check & Fallback: If no function arguments could be parsed by any method
            if not function_args:
                self.logger.warning("[VisualizationAgent] Could not extract valid function arguments from LLM response after all attempts.")
                self.logger.info("[VisualizationAgent] Defaulting to 'visualization_needed: False' due to parsing failure.")
                # Default to NO visualization if parsing fails
                function_args = {
                    "visualization_needed": False,
                    "rationale": "Could not reliably parse LLM response to determine visualization needs.",
                    "visualization_types": []
                }
            # --- REMOVED the previous fallback that forced a bar chart ---
            # else: # Old fallback logic based on regex content analysis (now handled by LLM or the final 'False' fallback)
            #     self.logger.info(f"[VisualizationAgent] Content does not appear to need visualization from fallback analysis")
            #     function_args = { ... }
                
            # Return visualization needs analysis
            self.logger.info(f"[VisualizationAgent] Final function_args: {function_args}")
            return function_args
            
        except Exception as e:
            self.logger.error(f"[VisualizationAgent] Error determining visualization needs: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    async def generate_visualization_code(self, search_result, visualization_needs):
        """
        Generate code to create visualizations based on the visualization needs
        and search results.
        
        Args:
            search_result: The results from a search task
            visualization_needs: The output from determine_visualization_needs
            
        Returns:
            Dict containing the generated code and visualization metadata
        """
        # Import necessary functions
        import os
        from llm_clients import get_async_llm_client
        
        try:
            # Skip if visualization is not needed
            if not visualization_needs or not visualization_needs.get("visualization_needed", False):
                return None
                
            # Get configuration - respecting user's model selection
            # Using Claude Sonnet 4 as fallback since it's the best coding model available
            if self.config is not None:
                from src.graph import get_configurable
                configurable = get_configurable(self.config)
                provider = configurable.llm_provider
                model = configurable.llm_model
            else:
                # Try environment variables first
                provider = os.environ.get("LLM_PROVIDER")
                model = os.environ.get("LLM_MODEL")
                
                if not provider:
                    provider = "azure"
                # model will use default from MODEL_CONFIGS if None
            
            # Get async LLM client
            llm = await get_async_llm_client(provider, model)
            
            # Extract content from search result
            content = search_result.get("formatted_sources", "")
            if isinstance(content, list):
                content = "\n".join(content)
                
            # Get query and subtask info
            query = search_result.get("search_string", "")
            subtask = search_result.get("subtask", {})
            subtask_name = subtask.get("name", "")
            
            # Get visualization types
            viz_types = visualization_needs.get("visualization_types", [])
            viz_descriptions = [f"- {v['type']}: {v['description']}" for v in viz_types]
            viz_descriptions_str = "\n".join(viz_descriptions)
            
            # Create system prompt for code generation
            system_prompt = """
            You are an expert Python data visualization programmer. Your task is to generate Python code that creates 
            visualizations based on provided research content. 
            
            CRITICAL CODE QUALITY REQUIREMENTS:
            1. The code MUST be syntactically correct and complete (no truncated lines)
            2. All strings must be properly terminated with closing quotes
            3. All parentheses and brackets must be properly balanced
            4. All statements must be complete
            
            IMPORTANT REQUIREMENTS:
            1. **Generate Python code for ONLY THE SINGLE MOST appropriate and informative visualization** based on the provided data and the research subtask. Do not generate multiple visualization types.
            2. Write simple, robust code that handles edge cases gracefully
            3. Always use try-except blocks around ALL data extraction and visualization creation
            4. Include fallback visualizations if data extraction fails
            5. Generate PURE Python code only - no markdown, no explanations, no backticks
            6. Always include explicit default values when extracting data to prevent errors
            7. For matplotlib, always specify figure size, use plt.tight_layout(), and plt.savefig() with explicit paths
            8. For seaborn, always use the 'data' parameter with DataFrames
            9. Write code that is guaranteed to produce at least one visualization even with imperfect data
            10. Always close figures with plt.close() after saving them
            11. Use simple, reliable chart types when possible (bar charts, pie charts, line charts)
            12. **Include a comment `# CHART_DESCRIPTION: Your concise description here` on a single line right before the `plt.savefig()` call.** This description should briefly explain what the chart shows (e.g., "Market share comparison in 2023", "Revenue trend over the last 5 years").
            
            Your code must be complete, with all necessary imports, and should only use standard Python
            libraries like matplotlib, pandas, numpy, and seaborn. Do not use external APIs or data sources.
            All data should be extracted directly from the provided research content.
            
            Only output the complete Python code with no additional text or explanations.
            """
            
            # Add model-specific instructions for Google Gemini
            if provider in ("google", "gemini"):
                system_prompt += """
                
                CRITICAL ADDITIONAL INSTRUCTIONS FOR CODE SYNTAX:
                1. DO NOT place multiple statements on the same line - each statement MUST be on its own line
                2. DO NOT use inconsistent indentation
                3. ALWAYS add proper return statements in functions - with ONE return statement per line
                4. NEVER truncate strings, function definitions, or control structures
                5. ALWAYS add explicit exit conditions in loops and conditionals
                6. CHECK your final code for syntax errors before submitting
                7. AVOID complex nested list comprehensions or dictionary comprehensions
                8. MAKE SURE all if/for/while blocks have properly indented bodies
                9. PREFER simple bar charts, pie charts or line charts that are guaranteed to work
                10. NEVER use complicated data parsing that could fail
                11. USE DIFFERENT chart types based on the data (not just bar charts every time)
                
                EXAMPLES OF SYNTAX ERRORS TO AVOID:
                - BAD: `return data    return None # Return None if extraction 'fails'` (multiple returns on one line)
                - BAD: `if data: return data` (conditional and return on same line)
                - BAD: ```if all_nan: print("Warning...")``` (statement immediately after control flow)
                - GOOD: ```if all_nan:
                    print("Warning...")``` (proper indentation and line breaks)
                """
            
            # Create user message with content and visualization requirements
            user_message = f"""
            Generate Python code to create visualizations based on the following research content about "{subtask_name}":
            
            QUERY: {query}
            
            VISUALIZATION REQUIREMENTS:
            {viz_descriptions_str}
            
            RESEARCH CONTENT:
            {content[:6000]}  # Limit content length
            
            The code should save the single, most appropriate visualization to the "{self.output_dir}" directory with clear filenames.
            The visualization should have appropriate titles and labels.
            **Include a comment line starting exactly with `# CHART_DESCRIPTION:` right before saving the figure, describing the chart concisely.**
            
            Extract all necessary data points from the research content. Do not fetch any external data.
            Parse the text to extract numbers, statistics, trends, and relationships that can be visualized.
            
            Generate complete, runnable Python code with no external dependencies beyond standard libraries.
            """
            
            # Format messages for the LLM client
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Set higher max_tokens for code generation (overriding the 4000 default)
            # This will allow generating more complete code
            if hasattr(llm, 'max_tokens'):
                original_max_tokens = llm.max_tokens
                llm.max_tokens = 8192  # 8K tokens should be sufficient for most visualization code
                self.logger.info(f"[VisualizationAgent] Temporarily increased max_tokens from {original_max_tokens} to 8192 for code generation")
            
            # Call LLM API
            self.logger.info(f"[VisualizationAgent] Generating visualization code...")
            response = await llm.ainvoke(messages)
            
            # Reset max_tokens to original value
            if hasattr(llm, 'max_tokens'):
                llm.max_tokens = original_max_tokens
                self.logger.info(f"[VisualizationAgent] Reset max_tokens to {original_max_tokens}")
            
            # Extract content from the response
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
                
            # Extract code from the response
            code = self.extract_code_blocks(response_content)
            
            self.logger.info(f"[VisualizationAgent - generate_visualization_code] Generated visualization code: {code}")
            
            # Return code and metadata
            return {
                "code": code,
                "code_snippets": [{
                    "language": "python",
                    "code": code,
                    "title": f"Visualization Code for {subtask_name}"
                }],
                "visualization_types": viz_types,
                "subtask_name": subtask_name,
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"[VisualizationAgent] Error generating visualization code: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def extract_code_blocks(self, text):
        """Extract Python code blocks from text."""
        # Try to match code blocks with ```python tag
        python_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
        matches = python_pattern.findall(text)
        
        if matches:
            # Combine all code blocks
            return "\n\n".join([block.strip() for block in matches])
        
        # Try to match code blocks with just ```
        alt_pattern = re.compile(r"```\n(.*?)\n```", re.DOTALL)
        matches = alt_pattern.findall(text)
        if matches:
            extracted_code = "\n\n".join([block.strip() for block in matches])
        else:
            # If no code blocks found, assume the full text is the code
            extracted_code = text.strip()

        # Extra safeguard: Remove potential leading ```python or ``` even from extracted code
        if extracted_code.startswith("```python"):
            extracted_code = extracted_code[len("```python"):].strip()
        elif extracted_code.startswith("```"):
            extracted_code = extracted_code[len("```"):].strip()
            
        # Remove potential trailing ```
        if extracted_code.endswith("```"):
             extracted_code = extracted_code[:-len("```")].strip()

        return extracted_code
    
    async def execute_visualization_code(self, code_data):
        """
        Execute the generated visualization code using the E2B sandbox.
        
        Args:
            code_data: Dict containing the code and metadata
            
        Returns:
            Dict containing the visualization results
        """
        # Import necessary module
        from e2b_code_interpreter import Sandbox
        
        sandbox = None # Initialize sandbox variable
        start_time = time.time() # Record overall start time
        init_start_time = None
        exec_start_time = None
        pause_start_time = None
        MAX_RETRY_ATTEMPTS = 1
        retry_count = 0
        
        # Initialize results list - moved outside the retry loop
        results = []  
        
        try:
            if not code_data or not code_data.get("code"):
                return None
            
            while retry_count <= MAX_RETRY_ATTEMPTS:
                # Close previous sandbox if it exists (on retry)
                if sandbox and retry_count > 0:
                    try:
                        sandbox.close()
                        self.logger.info(f"[VisualizationAgent] Closed previous sandbox for retry attempt")
                    except Exception as close_err:
                        self.logger.error(f"[VisualizationAgent] Error closing previous sandbox: {str(close_err)}")
                    # Recreate sandbox
                    sandbox = None
                
                # Initialize E2B sandbox with specific template ID
                self.logger.info(f"[VisualizationAgent] Initializing sandbox{' for retry attempt ' + str(retry_count) if retry_count > 0 else ''}...")
                init_start_time = time.time()
                template_id = "xe0uinj2n1ufrgmmksbu" 
                sandbox = Sandbox(template_id)
                init_duration = time.time() - init_start_time
                self.logger.info(f"[VisualizationAgent] Initialized sandbox with template ID: {template_id} (took {init_duration:.2f}s)")
                
                self.logger.info(f"[VisualizationAgent] Executing visualization code in sandbox{' (retry attempt ' + str(retry_count) + ')' if retry_count > 0 else ''}...")
                exec_start_time = time.time()
                
                # Get code - on retry, generate simplified version
                code = code_data.get("code")
                
                self.logger.info(f"[VisualizationAgent] Generated visualization code: {code}")
                if retry_count > 0:
                    # For a retry, simplify the code by adding better error handling
                    self.logger.info(f"[VisualizationAgent] Generating simplified code for retry attempt")
                    simplified_code = await self.generate_simplified_visualization_code(code_data, str(execution.error) if 'execution' in locals() else "Unknown error")
                    if simplified_code:
                        code = simplified_code
                
                # Run code in sandbox
                execution = sandbox.run_code(code)
                exec_duration = time.time() - exec_start_time
                self.logger.info(f"[VisualizationAgent] Code execution finished (took {exec_duration:.2f}s)")
                
                # Check for errors
                if execution.error:
                    error_msg = str(execution.error)
                    self.logger.error(f"[VisualizationAgent] Code execution error: {error_msg}")
                    
                    # Try again if we haven't reached max retries
                    if retry_count < MAX_RETRY_ATTEMPTS:
                        self.logger.info(f"[VisualizationAgent] Will retry with simplified code (attempt {retry_count + 1} of {MAX_RETRY_ATTEMPTS})")
                        retry_count += 1
                        continue
                    
                    # If we've reached max retries, give up and return the error
                    error_result = {
                        "error": error_msg,
                        "code_data": code_data
                    }
                    return error_result  # Return error after max retries
                else:
                    # No error, process results for this successful attempt (original or retry)
                    # Clear the results list only for the first successful attempt
                    if not results:
                        results = []  # Reset results only if empty
                    
                    # --- START: Parse description from code --- 
                    chart_description = None
                    if code:
                        try:
                            desc_match = re.search(r"^#\s*CHART_DESCRIPTION:\s*(.*)$", code, re.MULTILINE | re.IGNORECASE)
                            if desc_match:
                                chart_description = desc_match.group(1).strip()
                                self.logger.info(f"[VisualizationAgent] Found chart description: {chart_description}")
                            else:
                                self.logger.warning("[VisualizationAgent] `# CHART_DESCRIPTION:` comment not found in code.")
                        except Exception as parse_err:
                             self.logger.error(f"[VisualizationAgent] Error parsing chart description from code: {parse_err}")
                    # --- END: Parse description from code --- 
                    
                    # Process visualization results from this successful execution
                    if execution.results:
                        for result in execution.results:
                            # Check for images
                            if hasattr(result, 'png') and result.png:
                                # Generate unique filename
                                unique_id = str(uuid.uuid4())[:8]
                                subtask_name = code_data.get("subtask_name", "").replace(" ", "_").lower()
                                if not subtask_name:
                                    subtask_name = "visualization"
                                filename = f"{subtask_name}_{unique_id}.png"
                                filepath = os.path.join(self.output_dir, filename)
                                
                                # Save image
                                with open(filepath, 'wb') as f:
                                    f.write(base64.b64decode(result.png))
                                    
                                results.append({
                                    "type": "image",
                                    "filepath": filepath,
                                    "filename": filename,
                                    "description": chart_description
                                })
                    
                    # If no images found, check for file outputs in the logs
                    if not results and hasattr(execution, 'logs') and hasattr(execution.logs, 'stdout'):
                        stdout = ''.join(execution.logs.stdout) if execution.logs.stdout else ""
                        
                        # Look for saved file paths in stdout
                        file_matches = re.finditer(r"[Ss]aved (?:to |as |file |image |chart |graph |plot |figure |visualization )?"
                                                r"(?:to |at |in )?([\'\"])?([^\\s\'\"]+\\.(png|jpg|jpeg|svg|pdf))\\1", stdout)
                        
                        for match in file_matches:
                            filepath = match.group(2)
                            filename = os.path.basename(filepath)
                            
                            # Check if file exists and is in the output directory or needs to be moved
                            if os.path.exists(filepath):
                                # If file is not in output directory, move it
                                if os.path.dirname(filepath) != self.output_dir:
                                    new_filepath = os.path.join(self.output_dir, filename)
                                    os.rename(filepath, new_filepath)
                                    filepath = new_filepath
                                    
                                results.append({
                                    "type": "image",
                                    "filepath": filepath,
                                    "filename": filename,
                                    "description": chart_description
                                })
                    
                    # If we successfully processed results, we're done with retries
                    if results:
                        self.logger.info(f"[VisualizationAgent] Successfully processed {len(results)} visualization results")
                    else:
                        self.logger.warning("[VisualizationAgent] No visualization results found in successful execution")
                        
                    # Break out of retry loop since we had a successful execution
                    error_result = None
                    break
            
            # Construct final result (can be error or success)
            if error_result:
                final_result = error_result
            else:
                self.logger.info(f"[VisualizationAgent] Final results: {results}")
                # Enhanced image processing for visualization activities
                processed_results = []
                for item in results:
                    # Read the image file and encode it as base64
                    try:
                        with open(item["filepath"], "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            
                            # Get image format from filename
                            img_format = os.path.splitext(item["filename"])[1].lower().replace('.', '')
                            if not img_format or img_format not in ['png', 'jpg', 'jpeg', 'svg', 'gif']:
                                img_format = 'png'  # Default format
                            
                            # Create enhanced image result with src attribute for direct rendering
                            processed_item = {
                                "type": "image",
                                "filepath": item["filepath"],
                                "filename": item["filename"],
                                "description": item["description"],
                                "format": img_format,
                                "data": image_data,
                                "src": f"data:image/{img_format};base64,{image_data}"
                            }
                            processed_results.append(processed_item)
                            self.logger.info(f"[VisualizationAgent] Successfully encoded image {item['filename']} as base64")
                    except Exception as e:
                        self.logger.error(f"[VisualizationAgent] Error encoding image {item['filename']}: {str(e)}")
                        # Still add the original item without base64 encoding
                        processed_results.append(item)
                        
                # Log the processed results
                self.logger.info(f"[VisualizationAgent] Prepared {len(processed_results)} visualization results with base64 encoding")
                
                final_result = {
                    "results": processed_results,  # Now with base64 encoded images
                    "execution_logs": {
                        "stdout": ''.join(execution.logs.stdout) if hasattr(execution, 'logs') and hasattr(execution.logs, 'stdout') else "",
                        "stderr": ''.join(execution.logs.stderr) if hasattr(execution, 'logs') and hasattr(execution.logs, 'stderr') else ""
                    },
                    "code_data": code_data,
                    "retry_count": retry_count
                }
            return final_result

        except Exception as e:
            self.logger.error(f"[VisualizationAgent] Error executing visualization code: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return error, but still try to pause in finally
            return { 
                "error": str(e),
                "code_data": code_data
            }
        finally:
            # Pause the sandbox if it was successfully created
            if sandbox:
                self.logger.info(f"[VisualizationAgent] Pausing sandbox...")
                pause_start_time = time.time()
                try:
                    # paused_id = sandbox.pause()
                    pause_duration = time.time() - pause_start_time
                    # self.logger.info(f"[VisualizationAgent] Paused sandbox with ID: {paused_id} (took {pause_duration:.2f}s)")
                except Exception as pause_err:
                    pause_duration = time.time() - pause_start_time
                    self.logger.error(f"[VisualizationAgent] Error pausing sandbox (after {pause_duration:.2f}s): {str(pause_err)}")
                    # If pausing fails, try to close it as a fallback
                    try:
                        close_start_time = time.time()
                        sandbox.close()
                        close_duration = time.time() - close_start_time
                        self.logger.warning(f"[VisualizationAgent] Closed sandbox after pause failed (took {close_duration:.2f}s).")
                    except Exception as close_err:
                         self.logger.error(f"[VisualizationAgent] Error closing sandbox after pause failed: {str(close_err)}")
            
            total_duration = time.time() - start_time
            self.logger.info(f"[VisualizationAgent] execute_visualization_code finished (total time: {total_duration:.2f}s)")
    
    async def generate_simplified_visualization_code(self, code_data, error_message):
        """
        Generate a simplified visualization code when the original code execution fails.
        
        Args:
            code_data: The original code data
            error_message: The error message from the failed execution
            
        Returns:
            Simplified Python code as a string or None if generation fails
        """
        try:
            # Import necessary functions
            import os
            from llm_clients import get_async_llm_client
            
            # Get configuration
            if self.config is not None:
                from src.graph import get_configurable
                configurable = get_configurable(self.config)
                provider = configurable.llm_provider
                model = configurable.llm_model
            else:
                # Try environment variables first
                provider = os.environ.get("LLM_PROVIDER")
                model = os.environ.get("LLM_MODEL")
                
                if not provider:
                    provider = "azure"
                # model will use default from MODEL_CONFIGS if None
            
            # Get async LLM client
            llm = await get_async_llm_client(provider, model)
            
            # Create a simplified system prompt
            system_prompt = """
            You are an expert Python visualization troubleshooter. Your task is to fix or simplify visualization code that failed to execute.
            Create a MUCH SIMPLER, guaranteed-to-work version that focuses on reliability over complexity.
            
            IMPORTANT:
            1. Create a MINIMAL, bare-bones visualization that WILL DEFINITELY RUN without errors
            2. Focus on creating just ONE simple chart (bar chart or pie chart) with minimal data processing
            3. Use explicit try-except blocks around ALL operations
            4. Use simple hard-coded data if extracting from text is problematic
            5. Return ONLY Python code - no markdown, no explanations, no backticks
            6. Keep the visualization relevant to the original topic
            7. Include ALL necessary imports at the top
            8. Make sure the code saves the figure to the specified output directory
            9. Always close the figure after saving
            10. AFTER saving the figure, print a confirmation message exactly in the format: print(f"Saved figure to {filepath}") where filepath is the actual path used.
            """
            
            # Add model-specific instructions for Google Gemini
            if provider in ("google", "gemini"):
                system_prompt += """
                
                CRITICAL ADDITIONAL INSTRUCTIONS FOR CODE SYNTAX:
                1. DO NOT place multiple statements on the same line - each statement MUST be on its own line
                2. ALWAYS use consistent indentation - 4 spaces per indentation level
                3. NEVER include code blocks that are unfinished or truncated
                4. EVERY function must have a clear return path
                5. NEVER return two values on the same line
                6. All code blocks (if, for, while, functions) must be properly closed
                7. AVOID using complex data parsing or regex logic - use synthetic data instead
                8. TEST if your solution works with minimal data before finalizing 
                9. TRY A DIFFERENT chart type from the original failed attempt
                10. USE a simple, direct approach without advanced features
                
                EXAMPLES OF CORRECT CODE STRUCTURE:
                ```python
                # Good - properly separated statements
                def extract_data():
                    try:
                        # Code here
                        return data
                    except Exception as e:
                        print(f"Error: {e}")
                        return None
                
                # Good - clear function structure with proper returns
                def create_visualization(data):
                    if data is None:
                        # Fallback data
                        data = {"x": [1, 2, 3], "y": [4, 5, 6]}
                    
                    # Create visualization
                    plt.figure(figsize=(10, 6))
                    # Chart code
                    plt.savefig("output.png")
                    plt.close()
                ```
                """
                
                # For Gemini retries, explicitly request a different chart type (not bar chart)
                subtask_name = code_data.get("subtask_name", "Unknown")
                viz_types = code_data.get("visualization_types", [])
                orig_viz_type = viz_types[0]["type"] if viz_types else "bar_chart" 
                
                # Choose a different chart type for variety
                if orig_viz_type == "bar_chart":
                    suggested_type = "line_chart or pie_chart"
                elif orig_viz_type == "line_chart":
                    suggested_type = "pie_chart or scatter_plot"
                elif orig_viz_type == "pie_chart":
                    suggested_type = "line_chart or bar_chart with different orientation"
                else:
                    suggested_type = "line_chart or bar_chart"
                    
                system_prompt += f"""
                
                IMPORTANT FOR THIS RETRY: 
                The previous {orig_viz_type} failed. Please try using a {suggested_type} instead
                for better results. Using a different visualization approach often resolves execution errors.
                """
            
            # Get the original code and visualization requirements
            original_code = code_data.get("code", "")
            subtask_name = code_data.get("subtask_name", "Unknown")
            viz_types = code_data.get("visualization_types", [])
            viz_type = viz_types[0]["type"] if viz_types else "bar_chart"
            
            # Create user message
            user_message = f"""
            The following visualization code failed with this error:
            {error_message}
            
            Original code:
            {original_code[:2000] if len(original_code) > 2000 else original_code}
            
            Task: Create a SIMPLIFIED, reliable visualization about "{subtask_name}" that focuses on {viz_type}.
            Save the visualization to "{self.output_dir}" directory.
            
            Create a minimal, guaranteed-to-work visualization that will run without errors.
            You can use synthetic/sample data if necessary, but make it relevant to the topic.
            """
            
            # Format messages for the LLM client
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Set higher max_tokens for code generation (overriding the 4000 default)
            if hasattr(llm, 'max_tokens'):
                original_max_tokens = llm.max_tokens
                llm.max_tokens = 8192  # 8K tokens should be sufficient for simplified visualization code
                self.logger.info(f"[VisualizationAgent] Temporarily increased max_tokens from {original_max_tokens} to 8192 for simplified code generation")
            
            # Call LLM API
            self.logger.info(f"[VisualizationAgent] Generating simplified visualization code...")
            response = await llm.ainvoke(messages)
            
            # Reset max_tokens to original value
            if hasattr(llm, 'max_tokens'):
                llm.max_tokens = original_max_tokens
                self.logger.info(f"[VisualizationAgent] Reset max_tokens to {original_max_tokens}")
            
            # Extract content from the response
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
                
            # Extract code from the response
            simplified_code = self.extract_code_blocks(response_content)
            
            self.logger.info(f"[VisualizationAgent] Generated simplified visualization code ({len(simplified_code)} chars)")
            return simplified_code
            
        except Exception as e:
            self.logger.error(f"[VisualizationAgent] Error generating simplified visualization: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    async def execute(self, search_result: Dict[str, Any], vis_needs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the visualization agent to generate code for visualizing search results.
        Accepts pre-determined visualization needs to avoid redundant checks.
        
        Args:
            search_result: The results from a search task.
            vis_needs: Optional pre-determined visualization needs from determine_visualization_needs.
            
        Returns:
            Dict containing the search_result, generated code, and results of execution.
        """
        self.logger.info(f"[VisualizationAgent] Executing visualization agent for search result: {search_result.get('title', 'No title')}")
        
        # Determine visualization needs ONLY if not provided
        if vis_needs is None:
            self.logger.info("[VisualizationAgent] Determining visualization needs (not provided)...")
            vis_needs = await self.determine_visualization_needs(search_result)
            self.logger.info(f"[VisualizationAgent] Determined visualization needs: {vis_needs}")
        else:
            self.logger.info("[VisualizationAgent] Using pre-determined visualization needs.")
            
        # Log the exact keys in vis_needs to help debug
        self.logger.info(f"[VisualizationAgent] Final visualization needs keys: {vis_needs.keys() if vis_needs else 'None'}")
        
        if vis_needs and vis_needs.get("visualization_needed", False):
            self.logger.info("[VisualizationAgent] Visualization needed, generating code")
            try:
                # Generate visualization code using LLM
                visualization_code = await self.generate_visualization_code(search_result, vis_needs)
                
                # Execute the visualization code in the sandbox
                results = await self.execute_visualization_code(visualization_code) # results can be success dict or error dict

                # DEBUG: Log the results before returning, handling both success and error cases
                if isinstance(results, dict) and "error" in results:
                    self.logger.warning(f"[VisualizationAgent] EXECUTE returning an error: {results.get('error')}")
                elif isinstance(results, dict) and "results" in results:
                    viz_list = results.get("results", [])
                    self.logger.info(f"[VisualizationAgent] EXECUTE RETURNING results with {len(viz_list)} items")
                    for i, result_item in enumerate(viz_list):
                        # Check if result_item is a dictionary before calling .get()
                        if isinstance(result_item, dict):
                            self.logger.info(f"[VisualizationAgent] Result {i+1}: {result_item.get('filepath', 'NO FILEPATH')} (type={type(result_item)})")
                        else:
                            self.logger.warning(f"[VisualizationAgent] Result {i+1} is not a dictionary: {result_item} (type={type(result_item)})")
                else:
                    # Handle unexpected result types
                     self.logger.warning(f"[VisualizationAgent] EXECUTE received unexpected result format: {results} (type={type(results)})")

                # Add code snippets to the enriched_data
                if visualization_code and "code" in visualization_code:
                    # Extract any chart description from the code
                    chart_description = None
                    code = visualization_code["code"]
                    try:
                        desc_match = re.search(r"^#\s*CHART_DESCRIPTION:\s*(.*)$", code, re.MULTILINE | re.IGNORECASE)
                        if desc_match:
                            chart_description = desc_match.group(1).strip()
                    except Exception as e:
                        self.logger.warning(f"Error extracting chart description: {e}")

                    # Create code snippet structure
                    code_snippet = {
                        "filename": "visualization.py",
                        "description": chart_description or "Generated visualization code",
                        "language": "python",
                        "code": code
                    }

                    # Add code snippets to results if it's a success case
                    if isinstance(results, dict) and "results" in results:
                        if "enriched_data" not in results:
                            results["enriched_data"] = {}
                        results["enriched_data"]["code_snippets"] = [code_snippet]
                        self.logger.info(f"[VisualizationAgent] Added code snippet to enriched_data for visualization")
                        # Also add code snippets directly in the return structure for the UI
                        results["code_snippets"] = [code_snippet]
                    elif isinstance(results, dict) and "error" in results:
                        # Add code snippet even in error case for debugging
                        if "enriched_data" not in results:
                            results["enriched_data"] = {}
                        results["enriched_data"]["code_snippets"] = [code_snippet]
                        self.logger.info(f"[VisualizationAgent] Added code snippet to enriched_data for error case")
                        # Also add code snippets directly in the return structure for the UI
                        results["code_snippets"] = [code_snippet]

                # Return search result with added visualization code and results
                return {
                    "search_result": search_result,
                    "code_data": visualization_code,
                    "results": results
                }
            except Exception as e:
                self.logger.error(f"[VisualizationAgent] Error in visualization execution: {str(e)}")
                return {
                    "search_result": search_result,
                    "error": f"Visualization failed: {str(e)}",
                    "results": []
                }
        else:
            self.logger.info("[VisualizationAgent] No visualization needed for this search result")
            return {
                "search_result": search_result,
                "message": "No visualization needed for this search result",
                "results": []
            } 