#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
json_cleaner.py

MAIN OBJECTIVE:
---------------
This script provides comprehensive JSON cleaning, repair, and validation
functionality for LLM outputs, handling malformed JSON with multiple
recovery strategies.

Dependencies:
-------------
- sys
- json
- re
- ast
- logging
- typing
- pydantic

MAIN FEATURES:
--------------
1) Repair common JSON formatting issues
2) Extract JSON from mixed text
3) Handle nested and escaped JSON
4) Schema validation with Pydantic
5) Multiple fallback strategies

Author:
-------
Antoine Lemor
"""

import json
import re
import ast
import logging
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, create_model


class JSONCleaner:
    """Advanced JSON cleaning and repair utility"""

    @staticmethod
    def sanitize_string_for_csv(text: Optional[str]) -> Optional[str]:
        """
        Remove or replace invalid UTF-8 characters (including surrogates) from a string.
        This prevents UnicodeEncodeError when writing to CSV.
        """
        if text is None:
            return None
        if not isinstance(text, str):
            return text

        # Remove surrogate characters and other invalid UTF-8 sequences
        try:
            # First encode to UTF-8 with 'replace' to handle surrogates
            encoded = text.encode('utf-8', errors='replace')
            # Then decode back to string
            cleaned = encoded.decode('utf-8', errors='ignore')
            return cleaned
        except Exception as e:
            logging.warning(f"Error sanitizing string: {e}")
            # Last resort: keep only ASCII characters
            return ''.join(char for char in text if ord(char) < 128)

    @staticmethod
    def sanitize_json_string(json_str: Optional[str]) -> Optional[str]:
        """
        Sanitize a JSON string by removing invalid UTF-8 characters.
        This is crucial for preventing encoding errors when the JSON contains
        invalid characters from the LLM response.
        """
        if json_str is None:
            return None
        if not isinstance(json_str, str):
            return json_str

        try:
            # First sanitize the string
            cleaned = JSONCleaner.sanitize_string_for_csv(json_str)

            # Try to parse and re-dump to ensure valid JSON
            parsed = json.loads(cleaned)

            # Recursively sanitize the parsed data
            def sanitize_dict(obj):
                if isinstance(obj, dict):
                    return {k: sanitize_dict(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize_dict(item) for item in obj]
                elif isinstance(obj, str):
                    return JSONCleaner.sanitize_string_for_csv(obj)
                else:
                    return obj

            sanitized_data = sanitize_dict(parsed)
            return json.dumps(sanitized_data, ensure_ascii=False)

        except (json.JSONDecodeError, Exception) as e:
            logging.debug(f"Could not sanitize JSON string: {e}")
            # Return the sanitized string even if it's not valid JSON
            return JSONCleaner.sanitize_string_for_csv(json_str)


def repair_json_string(json_str: str) -> Optional[str]:
    """
    Attempt to repair common JSON formatting issues.

    Common issues fixed:
    - Missing commas between properties
    - Trailing commas
    - Incomplete JSON (missing closing braces)
    - Mixed quotes
    - Unescaped quotes in values
    - Comments in JSON
    - Invalid Unicode characters
    - Explanatory text mixed with JSON
    """
    if not json_str or not json_str.strip():
        return None

    text = json_str.strip()

    # Remove BOM and invalid Unicode characters
    if text.startswith('\ufeff'):
        text = text[1:]

    # Remove Unicode replacement characters and other invalid chars
    text = text.replace('\ufffd', '').replace('�', '')
    text = text.replace('\udcff', '').replace('\\udcff', '')
    text = re.sub(r'\\u[dD][cC][0-9a-fA-F]{2}', '', text)

    # Special handling for multiple nested JSON attempts {"1": "{...}", "2": "{...}"}
    if re.match(r'^\s*\{\s*"[0-9]+"?\s*:', text):
        # Extract numbered attempts
        attempts = re.findall(r'"[0-9]+"?\s*:\s*"([^"\\]*(\\.[^"\\]*)*)"', text)
        if attempts:
            for attempt_tuple in reversed(attempts):
                attempt_str = attempt_tuple[0] if isinstance(attempt_tuple, tuple) else attempt_tuple
                try:
                    # Unescape inner JSON
                    unescaped = attempt_str.replace('\\"', '"').replace('\\\\', '\\')
                    parsed = json.loads(unescaped)
                    return json.dumps(parsed, ensure_ascii=False)
                except:
                    continue

        # Look for valid JSON objects within the structure
        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        for obj in reversed(json_objects):
            if any(field in obj for field in ['"themes', '"political_parties', '"sentiment', '"specific_themes']):
                try:
                    json.loads(obj)
                    return obj
                except:
                    continue

    # Remove comments (both // and /* */ style)
    text = re.sub(r'//[^\n"]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Remove explanatory text patterns
    explanatory_patterns = [
        r'JSON strict compliance[^{]*',
        r'Revised JSON Output[^{]*',
        r'\*\*[^*]+\*\*[^{]*',
        r'Response:[^{]*',
        r'Output:[^{]*'
    ]
    for pattern in explanatory_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # If the text contains explanatory content before JSON, extract just the JSON
    if 'JSON' in text and any(word in text for word in ['Output:', 'output:', 'Response:']):
        json_start = text.find('{')
        if json_start > 0:
            before_json = text[:json_start]
            if any(word in before_json.lower() for word in ['revised', 'output', 'response', 'json']):
                text = text[json_start:]

    # Remove malformed keys with special characters
    text = re.sub(r',?\s*"[^a-zA-Z_][^"]*"\s*:[^,}]*', '', text)
    text = re.sub(r',?\s*"[^"]*[�\\?]+[^"]*"\s*:[^,}]*', '', text)

    # Fix property names that have colons or spaces in them incorrectly
    text = re.sub(r'("[\w_]+"):\s*([^,\}]+?)(?://[^\n]*)?(?=[,\}])', r'\1: \2', text)

    # Try to fix incomplete JSON by counting braces
    open_braces = text.count('{')
    close_braces = text.count('}')
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)

    open_brackets = text.count('[')
    close_brackets = text.count(']')
    if open_brackets > close_brackets:
        text += ']' * (open_brackets - close_brackets)

    # Fix missing commas between properties (common issue)
    text = re.sub(r'"\s*\n\s*"', '",\n"', text)
    text = re.sub(r'"\s+"', '","', text)
    text = re.sub(r'}\s*"', '},"', text)
    text = re.sub(r']\s*"', '],"', text)
    text = re.sub(r'(\d)\s*"', r'\1,"', text)
    text = re.sub(r'(true|false|null)\s*"', r'\1,"', text, flags=re.IGNORECASE)

    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Fix escaped quotes that shouldn't be escaped
    text = re.sub(r'\\"([^"]*?)\\":', r'"\1":', text)

    # Ensure boolean values are lowercase
    text = text.replace('True', 'true').replace('False', 'false')
    text = text.replace('None', 'null')

    # Final validation attempt - extract best JSON object
    json_pattern = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}'
    potential_jsons = re.findall(json_pattern, text)

    best_json = None
    best_score = -1

    for candidate in potential_jsons:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                score = sum(1 for field in ['themes', 'political_parties', 'specific_themes', 'sentiment']
                           if any(field in k for k in parsed.keys()))
                if score > best_score:
                    best_score = score
                    best_json = candidate
        except:
            continue

    if best_json:
        return best_json

    return text


def extract_json_block(text: str) -> Optional[str]:
    """
    Extract JSON block from text using balanced brace counting.
    This handles nested objects correctly.
    """
    # Find the first opening brace
    start = text.find('{')
    if start == -1:
        return None

    # Count braces to find the matching closing brace
    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]

    # If we get here, JSON is incomplete - try to close it
    if brace_count > 0:
        return text[start:] + '}' * brace_count

    return None


def clean_json_output(output: str, expected_keys: List[str]) -> Optional[str]:
    """
    Transform the raw model response into a valid JSON string.
    Now uses the improved repair_json_string function first.
    """
    if not output or output.strip().lower() == 'null':
        return None

    # First, try the aggressive repair function
    repaired = repair_json_string(output)
    if repaired:
        try:
            parsed = json.loads(repaired)
            return _filter_keys(parsed, expected_keys)
        except json.JSONDecodeError:
            pass

    # Direct attempt
    try:
        raw = json.loads(output)
        return _filter_keys(raw, expected_keys)
    except json.JSONDecodeError as e:
        logging.debug(f"Initial JSON parse failed: {e}")

    # Clean up the text
    text = output.strip()

    # Remove code fences
    text = re.sub(r'^```[\w]*\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)

    # Remove thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<\/?think>', '', text, flags=re.IGNORECASE)

    # Try direct parse again after cleanup
    try:
        raw = json.loads(text)
        return _filter_keys(raw, expected_keys)
    except json.JSONDecodeError:
        pass

    # Extract and repair JSON block
    json_block = extract_json_block(text)
    if json_block:
        # Try to parse as-is
        try:
            raw = json.loads(json_block)
            return _filter_keys(raw, expected_keys)
        except json.JSONDecodeError:
            # Try to repair
            repaired = repair_json_string(json_block)
            if repaired:
                try:
                    raw = json.loads(repaired)
                    return _filter_keys(raw, expected_keys)
                except json.JSONDecodeError as e:
                    logging.debug(f"JSON repair failed: {e}")

    # Fallback: find any JSON-like structure
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, flags=re.DOTALL)

    for match in reversed(matches):  # Try from the end first
        # Try to parse
        try:
            raw = json.loads(match)
            return _filter_keys(raw, expected_keys)
        except json.JSONDecodeError:
            # Try repair
            repaired = repair_json_string(match)
            if repaired:
                try:
                    raw = json.loads(repaired)
                    return _filter_keys(raw, expected_keys)
                except json.JSONDecodeError:
                    pass

        # Try Python literal eval as last resort for this block
        try:
            py_obj = ast.literal_eval(match)
            if isinstance(py_obj, dict):
                return _filter_keys(py_obj, expected_keys)
        except Exception:
            continue

    # Last resort: try to build from scratch
    if expected_keys:
        found_values = {}
        for key in expected_keys:
            # Look for patterns like "key": "value" or "key": value
            patterns = [
                rf'"{key}"\s*:\s*"([^"]*?)"',
                rf'"{key}"\s*:\s*([^,\}}]+)',
                rf'{key}\s*:\s*"([^"]*?)"',
                rf'{key}\s*:\s*([^,\}}]+)'
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value
                    if value.lower() in ['true', 'false', 'null']:
                        found_values[key] = value.lower()
                    elif value.isdigit():
                        found_values[key] = int(value)
                    else:
                        found_values[key] = value.strip('"')
                    break

        if found_values:
            try:
                return json.dumps(found_values, ensure_ascii=False)
            except Exception:
                pass

    return None


def _filter_keys(data: Union[Dict, Any], expected_keys: List[str]) -> Optional[str]:
    """Filter dictionary to only include expected keys"""
    # Handle None/null case
    if data is None:
        logging.warning("_filter_keys received None/null")
        return None

    # Handle case where data is not a dict
    if not isinstance(data, dict):
        logging.warning(f"_filter_keys received {type(data).__name__} instead of dict")
        # Try to convert if it's a list with a single dict
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
            logging.info("_filter_keys: Converted single-item list to dict")
        else:
            # Return None if we can't convert
            return None

    if expected_keys:
        data = {k: data.get(k) for k in expected_keys}

    # Sanitize before returning
    json_str = json.dumps(data, ensure_ascii=False)
    return JSONCleaner.sanitize_json_string(json_str)


def build_dynamic_schema(expected_keys: Optional[List[str]]):
    """
    Build both:
    1. A Pydantic model (`pyd_model`) used to validate the JSON.
    2. A lean JSON schema (`ollama_schema`) compatible with Ollama.

    IMPORTANT: Keys are now optional (with default value None)
    to avoid validation errors when the model cannot fill all keys.
    """
    if not expected_keys:
        return None, None

    # 1) Pydantic model: all keys are optional with default value
    field_type = Union[str, List[Optional[str]], None]

    # Create fields with default value None
    fields = {}
    for k in expected_keys:
        fields[k] = (field_type, None)  # Tuple (type, default_value)

    pyd_model = create_model(
        'DynamicLLMSchema',
        **fields
    )

    # 2) JSON schema for Ollama: simplified format for better compatibility
    # Note: Some Ollama models struggle with complex schemas, so we simplify
    ollama_schema = {
        "type": "object",
        "properties": {
            k: {
                "type": "string",  # Simplified to just string type
                "description": f"Value for {k}"
            } for k in expected_keys
        },
        "required": [],  # Empty list = no required keys
        "additionalProperties": True,  # Allow additional properties for flexibility
    }

    return pyd_model, ollama_schema


def _extract_json_block_after_header(text: str, header_pos: int) -> Optional[str]:
    """
    Extract a complete JSON block (including nested structures) starting after a header position.
    Uses bracket counting to handle nested structures.
    """
    # Find the first '{' after the header
    start_pos = text.find('{', header_pos)
    if start_pos == -1:
        return None

    # Count brackets to find matching closing brace
    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start_pos:], start=start_pos):
        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start_pos:i + 1]

    return None


def extract_expected_keys(text: str) -> List[str]:
    """
    Attempts to robustly detect JSON keys in the text.
    Priority order:
    1) Look for "Expected JSON" or similar headers and parse the FULL JSON block (HIGHEST PRIORITY)
    2) Find all JSON blocks using bracket counting and use the schema-like ones
    3) Fall back on pattern matching for key definitions
    """
    # Method 1: Look for specific section headers and extract the full JSON block after them
    header_patterns = [
        r'\*\*Expected JSON[:\*]*',  # **Expected JSON:** or **Expected JSON**
        r'Expected JSON Keys?[:\s]*',
        r'JSON Schema[:\s]*',
        r'Output Format[:\s]*',
        r'Response Format[:\s]*',
        r'JSON Structure[:\s]*',
        r'The response should be a JSON object',
        r'Return a JSON object',
    ]

    for pattern in header_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            json_block = _extract_json_block_after_header(text, match.end())
            if json_block:
                try:
                    loaded = json.loads(json_block)
                    if isinstance(loaded, dict):
                        # Return top-level keys only (not nested keys)
                        return list(loaded.keys())
                except (json.JSONDecodeError, TypeError):
                    continue

    # Method 2: Find ALL JSON blocks using bracket counting, prioritize schema-like blocks
    def find_all_json_blocks(text: str) -> List[str]:
        """Find all top-level JSON blocks using bracket counting."""
        blocks = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                block = _extract_json_block_after_header(text, i - 1)
                if block:
                    blocks.append(block)
                    i += len(block)
                else:
                    i += 1
            else:
                i += 1
        return blocks

    all_json_blocks = find_all_json_blocks(text)
    if all_json_blocks:
        # First pass: look for schema-like blocks (empty values like "", [], null)
        schema_candidates = []
        example_candidates = []

        for json_block in all_json_blocks:
            try:
                loaded = json.loads(json_block)
                if isinstance(loaded, dict):
                    keys = [k for k in loaded.keys() if k not in ['type', 'description', 'example', 'required', 'properties']]
                    if keys:
                        # Check if this looks like a schema (mostly empty values)
                        empty_count = sum(1 for v in loaded.values() if v in ("", [], {}, None) or v == "null")
                        if empty_count >= len(loaded) * 0.5:  # At least 50% empty values = schema-like
                            schema_candidates.append(keys)
                        else:
                            example_candidates.append(keys)
            except (json.JSONDecodeError, TypeError):
                continue

        # Prefer schema-like blocks, then fall back to examples
        # For schemas, prefer the one with the most keys
        if schema_candidates:
            return max(schema_candidates, key=len)
        if example_candidates:
            # For examples, prefer the one with the most keys (usually the first/template example)
            return max(example_candidates, key=len)

    # Method 3: Look for JSON key definition patterns
    key_pattern = r'"([a-zA-Z_][a-zA-Z0-9_]*)":\s*(?:"[^"]*"|\[[^\]]*\]|\{[^}]*\}|null|true|false|\d+)'
    matches = re.findall(key_pattern, text)

    if matches:
        # Filter duplicates while preserving order
        seen = set()
        final_keys = []
        for key in matches:
            if key not in seen and not key in ['type', 'description', 'example', 'required', 'properties']:
                seen.add(key)
                final_keys.append(key)

        if final_keys:
            return final_keys

    # Method 4: Look for key lists
    list_pattern = r'(?:keys|fields|properties)[:\s]*\[([^\]]+)\]'
    list_match = re.search(list_pattern, text, flags=re.IGNORECASE)
    if list_match:
        keys_str = list_match.group(1)
        # Extract strings between quotes
        keys = re.findall(r'["\']([^"\']+)["\']', keys_str)
        if keys:
            return keys

    # Method 5: Return empty list if nothing found
    logging.warning("Could not extract JSON keys from prompt. Schema validation will be disabled.")
    return []