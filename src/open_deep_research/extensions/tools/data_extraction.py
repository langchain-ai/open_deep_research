"""Data extraction tools - extract structured data from text."""
import json
import re
from langchain_core.tools import StructuredTool
from extensions.models.tool_schemas import DataExtractionInput


def extract_data(text: str, format: str = "json") -> str:
    """Extract structured data from text (tables, lists, key-value pairs).
    
    Args:
        text: Text containing data to extract
        format: Output format ('json', 'csv', 'table')
        
    Returns:
        Extracted data in requested format
    """
    try:
        extracted = []
        
        # Method 1: Extract markdown tables
        table_pattern = r'\|(.+)\|'
        tables = re.findall(table_pattern, text, re.MULTILINE)
        
        if tables:
            # Parse markdown table
            headers = [h.strip() for h in tables[0].split('|')]
            rows = []
            
            for line in tables[2:]:  # Skip header and separator
                if '|' in line:
                    row_data = [cell.strip() for cell in line.split('|')]
                    if len(row_data) == len(headers):
                        row_dict = dict(zip(headers, row_data))
                        rows.append(row_dict)
            
            extracted = rows
        
        # Method 2: Extract key-value pairs
        if not extracted:
            kv_pattern = r'([A-Za-z\s]+):\s*(.+)'
            matches = re.findall(kv_pattern, text)
            
            if matches:
                extracted = [{key.strip(): value.strip() for key, value in matches}]
        
        # Method 3: Extract numbered/bulleted lists
        if not extracted:
            list_pattern = r'(?:[-•*]|\d+[\.):])\s*(.+)'
            items = re.findall(list_pattern, text)
            
            if items:
                extracted = [{"item": item.strip()} for item in items]
        
        # Format output
        if not extracted:
            return "[ERROR] No structured data found in text. Try providing data in table, list, or key-value format."
        
        if format == "json":
            return json.dumps(extracted, indent=2)
        
        elif format == "csv":
            if not extracted:
                return ""
            
            # Get headers
            headers = list(extracted[0].keys())
            csv_lines = [','.join(headers)]
            
            # Add rows
            for row in extracted:
                csv_lines.append(','.join(str(row.get(h, '')) for h in headers))
            
            return '\n'.join(csv_lines)
        
        elif format == "table":
            if not extracted:
                return ""
            
            # Get headers
            headers = list(extracted[0].keys())
            
            # Calculate column widths
            widths = {h: max(len(h), max(len(str(row.get(h, ''))) for row in extracted)) for h in headers}
            
            # Build table
            header_row = ' | '.join(h.ljust(widths[h]) for h in headers)
            separator = '-+-'.join('-' * widths[h] for h in headers)
            
            table_lines = [header_row, separator]
            
            for row in extracted:
                row_line = ' | '.join(str(row.get(h, '')).ljust(widths[h]) for h in headers)
                table_lines.append(row_line)
            
            return '\n'.join(table_lines)
        
        return json.dumps(extracted, indent=2)
        
    except Exception as e:
        return f"[ERROR] Error extracting data: {str(e)}"


# Create LangChain tool
extract_data_tool = StructuredTool.from_function(
    func=extract_data,
    name="extract_data",
    description="Extract structured data from text. Works with tables, lists, key-value pairs. Returns data in JSON, CSV, or table format.",
    args_schema=DataExtractionInput
)


__all__ = ['extract_data_tool', 'extract_data']
