from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
import uuid
import logging
import json
from datetime import datetime
import os

from models.file_analysis import FileUploadResponse, FileAnalysisResponse, FileAnalysisRequest
from services.file_storage import FileStorageService
from services.content_analysis import ContentAnalysisService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["File Analysis"], prefix="/api/files")

@router.post(
    "/upload",
    response_model=FileUploadResponse,
    summary="Upload a single file for analysis",
    description="Upload a file and get content analysis using LLM"
)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analyze_immediately: bool = Form(True),
    analysis_type: str = Form("comprehensive")
):
    """
    Upload a single file for content analysis.
    
    - **file**: The file to upload (PDF, CSV, images, documents, etc.)
    - **analyze_immediately**: Whether to start analysis immediately (default: True)
    - **analysis_type**: Type of analysis ('quick', 'comprehensive', 'custom')
    
    Returns file metadata and analysis status.
    """
    try:
        logger.info(f"Received file upload: {file.filename}, content_type: {file.content_type}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Store the file and get metadata
        file_metadata = await FileStorageService.store_file(file)
        
        # Create response
        response = FileUploadResponse(
            file_id=file_metadata["file_id"],
            filename=file_metadata["filename"],
            original_name=file_metadata["original_name"],
            file_type=file_metadata["file_type"],
            file_size=file_metadata["file_size"],
            status="uploaded" if not analyze_immediately else "processing",
            upload_timestamp=file_metadata["upload_timestamp"],
            analysis_eta="2-5 minutes" if analyze_immediately else None
        )
        
        # Start analysis in background if requested
        if analyze_immediately:
            background_tasks.add_task(
                ContentAnalysisService.analyze_file,
                file_metadata["file_id"],
                analysis_type
            )
            response.status = "processing"
        
        logger.info(f"File uploaded successfully: {file_metadata['file_id']}")
        return response
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@router.post(
    "/batch-upload",
    response_model=List[FileUploadResponse],
    summary="Upload multiple files for analysis",
    description="Upload multiple files and get content analysis for each"
)
async def batch_upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    analyze_immediately: bool = Form(True),
    analysis_type: str = Form("comprehensive")
):
    """
    Upload multiple files for content analysis.
    
    - **files**: List of files to upload
    - **analyze_immediately**: Whether to start analysis immediately for all files
    - **analysis_type**: Type of analysis to perform on all files
    
    Returns list of file metadata and analysis status for each file.
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
        
        responses = []
        
        for file in files:
            if not file.filename:
                continue
                
            logger.info(f"Processing batch file: {file.filename}")
            
            # Store each file
            file_metadata = await FileStorageService.store_file(file)
            
            # Create response for this file
            response = FileUploadResponse(
                file_id=file_metadata["file_id"],
                filename=file_metadata["filename"],
                original_name=file_metadata["original_name"],
                file_type=file_metadata["file_type"],
                file_size=file_metadata["file_size"],
                status="uploaded" if not analyze_immediately else "processing",
                upload_timestamp=file_metadata["upload_timestamp"],
                analysis_eta="2-5 minutes" if analyze_immediately else None
            )
            
            responses.append(response)
            
            # Start analysis in background if requested
            if analyze_immediately:
                background_tasks.add_task(
                    ContentAnalysisService.analyze_file,
                    file_metadata["file_id"],
                    analysis_type
                )
        
        logger.info(f"Batch upload completed: {len(responses)} files processed")
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")

@router.get(
    "/{file_id}/analysis",
    response_model=FileAnalysisResponse,
    summary="Get file analysis results",
    description="Retrieve the content analysis results for a specific file"
)
async def get_file_analysis(file_id: str):
    """
    Get analysis results for a specific file.
    
    - **file_id**: The unique identifier of the uploaded file
    
    Returns detailed content analysis including descriptions, metadata, and insights.
    """
    try:
        # Get analysis results
        analysis = await ContentAnalysisService.get_analysis(file_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="File analysis not found")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis for {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")

@router.post(
    "/{file_id}/analyze",
    response_model=FileAnalysisResponse,
    summary="Start or restart file analysis",
    description="Manually trigger analysis for an uploaded file"
)
async def analyze_file(
    file_id: str,
    background_tasks: BackgroundTasks,
    request: FileAnalysisRequest
):
    """
    Start or restart analysis for a specific file.
    
    - **file_id**: The unique identifier of the uploaded file
    - **analysis_type**: Type of analysis to perform
    - **custom_prompt**: Optional custom prompt for analysis
    
    Returns analysis results or processing status.
    """
    try:
        # Check if file exists
        file_exists = await FileStorageService.file_exists(file_id)
        if not file_exists:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Start analysis
        background_tasks.add_task(
            ContentAnalysisService.analyze_file,
            file_id,
            request.analysis_type,
            request.custom_prompt
        )
        
        return FileAnalysisResponse(
            file_id=file_id,
            status="processing",
            analysis_timestamp=datetime.now(),
            content_description="Analysis in progress...",
            metadata={},
            processing_time=0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis for {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@router.get(
    "/{file_id}/content",
    summary="Download original file",
    description="Download the original uploaded file"
)
async def get_file_content(file_id: str):
    """
    Download the original uploaded file.
    
    - **file_id**: The unique identifier of the uploaded file
    
    Returns the original file for download.
    """
    try:
        file_path = await FileStorageService.get_file_path(file_id)
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get original filename for proper download
        file_metadata = await FileStorageService.get_file_metadata(file_id)
        original_name = file_metadata.get("original_name", "download")
        
        return FileResponse(
            path=file_path,
            filename=original_name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file content for {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {str(e)}")

@router.get(
    "/{file_id}/status",
    summary="Get file processing status",
    description="Check the current status of file processing and analysis"
)
async def get_file_status(file_id: str):
    """
    Get current status of file processing and analysis.
    
    - **file_id**: The unique identifier of the uploaded file
    
    Returns current processing status and progress information.
    """
    try:
        status = await ContentAnalysisService.get_analysis_status(file_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="File not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.delete(
    "/{file_id}",
    summary="Delete uploaded file",
    description="Delete an uploaded file and its analysis results"
)
async def delete_file(file_id: str):
    """
    Delete an uploaded file and its analysis results.
    
    - **file_id**: The unique identifier of the uploaded file
    
    Returns confirmation of deletion.
    """
    try:
        success = await FileStorageService.delete_file(file_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {"message": f"File {file_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@router.get(
    "/",
    summary="List uploaded files",
    description="Get list of all uploaded files with their status"
)
async def list_files(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None
):
    """
    Get list of uploaded files.
    
    - **limit**: Maximum number of files to return (default: 50)
    - **offset**: Number of files to skip (default: 0)
    - **status_filter**: Filter by status ('uploaded', 'processing', 'completed', 'failed')
    
    Returns list of files with metadata and status.
    """
    try:
        files = await FileStorageService.list_files(limit, offset, status_filter)
        return files
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")