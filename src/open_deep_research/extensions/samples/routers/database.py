"""
Database API Router

This module provides API endpoints for database upload, management, and text2sql functionality.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import logging
import json
from pydantic import BaseModel, Field

from src.tools.text2sql_tool import Text2SQLTool

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Database"])

# Global Text2SQL tool instance
text2sql_tool = Text2SQLTool()

# Pydantic models for request/response
class DatabaseUploadResponse(BaseModel):
    database_id: str
    filename: str
    file_type: str
    tables: List[str]
    message: str

class DatabaseListResponse(BaseModel):
    databases: List[Dict[str, Any]]

class DatabaseSchemaResponse(BaseModel):
    database_id: str
    filename: str
    database_schema: Dict[str, Any] = Field(alias="schema")

class Text2SQLRequest(BaseModel):
    query: str
    database_id: Optional[str] = None

class Text2SQLResponse(BaseModel):
    query: str
    sql: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    database: str
    executed_at: str

@router.post(
    "/upload",
    response_model=DatabaseUploadResponse,
    summary="Upload a database file",
    description="Upload SQLite or CSV files for text2sql querying"
)
async def upload_database(
    file: UploadFile = File(...),
    file_type: Optional[str] = Form(None)
):
    """
    Upload a database file (SQLite or CSV) for text2sql functionality.
    
    Args:
        file: The database file to upload
        file_type: Optional file type override (sqlite, csv)
        
    Returns:
        Database upload response with ID and metadata
    """
    try:
        # Validate file type
        allowed_types = ['.db', '.sqlite', '.sqlite3', '.csv', '.json']
        file_extension = '.' + file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Upload to Text2SQL tool
        database_id = text2sql_tool.upload_database(
            file_content=file_content,
            filename=file.filename,
            file_type=file_type
        )
        
        # Get database info
        db_info = text2sql_tool.databases[database_id]
        
        logger.info(f"Successfully uploaded database: {file.filename} (ID: {database_id})")
        
        return DatabaseUploadResponse(
            database_id=database_id,
            filename=file.filename,
            file_type=db_info['file_type'],
            tables=db_info['metadata']['tables'],
            message=f"Database {file.filename} uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Error uploading database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/list",
    response_model=DatabaseListResponse,
    summary="List uploaded databases",
    description="Get a list of all uploaded databases"
)
async def list_databases():
    """
    Get a list of all uploaded databases.
    
    Returns:
        List of database information
    """
    try:
        databases = text2sql_tool.list_databases()
        
        return DatabaseListResponse(databases=databases)
        
    except Exception as e:
        logger.error(f"Error listing databases: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/{database_id}/schema",
    response_model=DatabaseSchemaResponse,
    summary="Get database schema",
    description="Get detailed schema information for a specific database"
)
async def get_database_schema(database_id: str):
    """
    Get schema information for a specific database.
    
    Args:
        database_id: The ID of the database
        
    Returns:
        Database schema information
    """
    try:
        if database_id not in text2sql_tool.databases:
            raise HTTPException(status_code=404, detail="Database not found")
        
        db_info = text2sql_tool.databases[database_id]
        schema = text2sql_tool.get_database_schema(database_id)
        
        return DatabaseSchemaResponse(
            database_id=database_id,
            filename=db_info['filename'],
            database_schema=schema
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting database schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/query",
    response_model=Text2SQLResponse,
    summary="Execute text2sql query",
    description="Convert natural language to SQL and execute against uploaded databases"
)
async def execute_text2sql(request: Text2SQLRequest):
    """
    Execute a text2sql query against uploaded databases.
    
    Args:
        request: Text2SQL request with query and optional database_id
        
    Returns:
        Query results with SQL and data
    """
    try:
        # Execute the query
        result = text2sql_tool.query_database(
            db_id=request.database_id,
            natural_language_query=request.query
        )
        
        return Text2SQLResponse(**result)
        
    except Exception as e:
        logger.error(f"Error executing text2sql query: {e}")
        return Text2SQLResponse(
            query=request.query,
            error=str(e),
            database=request.database_id or "Unknown",
            executed_at=text2sql_tool._get_current_time()
        )

@router.delete(
    "/{database_id}",
    summary="Delete database",
    description="Delete an uploaded database and its files"
)
async def delete_database(database_id: str):
    """
    Delete an uploaded database.
    
    Args:
        database_id: The ID of the database to delete
        
    Returns:
        Success message
    """
    try:
        if database_id not in text2sql_tool.databases:
            raise HTTPException(status_code=404, detail="Database not found")
        
        success = text2sql_tool.delete_database(database_id)
        
        if success:
            return {"message": f"Database {database_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete database")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a method to get current time for the tool
def _get_current_time():
    from datetime import datetime
    return datetime.now().isoformat()

# Monkey patch the method into the tool
Text2SQLTool._get_current_time = staticmethod(_get_current_time)
