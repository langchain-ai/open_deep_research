import os
import logging
import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import json
import base64

from services.file_parsers import parse_file
from services.file_storage import FileStorageService
from models.file_analysis import (
    FileAnalysisResponse,
    FileAnalysisStatus,
    FileStatus,
    ContentInsights,
    DocumentMetadata,
    ImageMetadata,
    DataFileMetadata,
    AudioVideoMetadata,
)

# Import LLM clients
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_clients

logger = logging.getLogger(__name__)


class ContentAnalysisService:
    """Service for analyzing file content using LLMs"""

    # Store analysis results in memory (in production, use a database)
    _analysis_cache = {}
    _status_cache = {}

    @classmethod
    async def analyze_file(
        cls,
        file_id: str,
        analysis_type: str = "comprehensive",
        custom_prompt: Optional[str] = None,
    ) -> FileAnalysisResponse:
        """
        Analyze a file's content using LLM

        Args:
            file_id: Unique identifier for the file
            analysis_type: Type of analysis ('quick', 'comprehensive', 'custom')
            custom_prompt: Custom prompt for analysis (used with 'custom' type)

        Returns:
            FileAnalysisResponse with analysis results
        """
        start_time = datetime.now()

        try:
            # Update status to processing
            cls._update_status(
                file_id, FileStatus.PROCESSING, "Starting content analysis", 10.0
            )

            # Get file metadata and path
            file_metadata = await FileStorageService.get_file_metadata(file_id)
            if not file_metadata:
                raise ValueError(f"File {file_id} not found")

            file_path = file_metadata["file_path"]
            file_type = file_metadata["file_type"]
            logger.info(f"File path: {file_path}")

            logger.info(f"Starting analysis for file {file_id} ({file_type})")

            # Update status
            cls._update_status(
                file_id, FileStatus.PROCESSING, "Parsing file content", 30.0
            )

            # Parse the file to extract content
            parsed_content = await parse_file(file_path, file_type)

            # Update status
            cls._update_status(
                file_id, FileStatus.PROCESSING, "Analyzing content with LLM", 60.0
            )

            # Generate LLM analysis
            analysis_result = await cls._analyze_with_llm(
                parsed_content, file_metadata, analysis_type, custom_prompt
            )

            # Update status
            cls._update_status(
                file_id, FileStatus.PROCESSING, "Finalizing analysis", 90.0
            )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Create response
            response = FileAnalysisResponse(
                file_id=file_id,
                status=FileStatus.COMPLETED,
                content_description=analysis_result["content_description"],
                analysis_timestamp=datetime.now(),
                metadata=analysis_result["metadata"],
                processing_time=processing_time,
            )

            # Cache the result
            cls._analysis_cache[file_id] = response

            # Update final status
            cls._update_status(
                file_id, FileStatus.COMPLETED, "Analysis complete", 100.0
            )

            logger.info(
                f"Analysis completed for file {file_id} in {processing_time:.2f}s"
            )
            return response

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(f"Error analyzing file {file_id}: {error_msg}")

            # Update status to failed
            cls._update_status(file_id, FileStatus.FAILED, error_msg, 0.0)

            processing_time = (datetime.now() - start_time).total_seconds()

            # Return error response
            response = FileAnalysisResponse(
                file_id=file_id,
                status=FileStatus.FAILED,
                content_description="Analysis failed",
                analysis_timestamp=datetime.now(),
                metadata={},
                processing_time=processing_time,
                error_message=error_msg,
            )

            cls._analysis_cache[file_id] = response
            return response

    @classmethod
    async def _analyze_with_llm(
        cls,
        parsed_content: Dict[str, Any],
        file_metadata: Dict[str, Any],
        analysis_type: str,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze parsed content using LLM

        Args:
            parsed_content: Parsed file content and metadata
            file_metadata: File metadata from storage
            analysis_type: Type of analysis to perform
            custom_prompt: Custom prompt for analysis

        Returns:
            Dictionary with analysis results
        """
        try:
            # Get LLM configuration from environment
            llm_provider = os.getenv("LLM_PROVIDER", "azure")
            llm_model = os.getenv("LLM_MODEL", None)

            logger.info(f"Using LLM: {llm_provider}/{llm_model} for analysis")
            logger.info(
                f"Analyzing content type: {parsed_content.get('content_type', 'unknown')}"
            )

            # Get LLM client
            llm = llm_clients.get_llm_client(llm_provider, llm_model)

            # Get file info
            file_path = file_metadata.get("file_path", "")
            file_type = file_metadata.get("file_type", "").lower()
            original_name = file_metadata.get("original_name", "unknown")
            content = parsed_content.get("content", "")
            content_type = parsed_content.get("content_type", "unknown")

            # Build analysis prompt based on content type and analysis level
            if content_type == "image":
                # For images, we need to send the image to a vision-capable model
                description = await cls._analyze_image_with_vision_llm(
                    llm, file_path, original_name, analysis_type, custom_prompt
                )
            else:
                # For text and other content, use text-based analysis
                description = await cls._analyze_text_content_with_llm(
                    llm,
                    content,
                    file_metadata,
                    parsed_content,
                    analysis_type,
                    custom_prompt,
                )

            # Extract metadata from parsed content
            base_metadata = {
                "file_type": file_metadata.get("file_type"),
                "file_size": file_metadata.get("file_size"),
                "content_type": parsed_content.get("content_type"),
                "analysis_method": "llm_analysis",
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            }

            # Add content-specific metadata
            parsed_metadata = parsed_content.get("metadata", {})
            base_metadata.update(parsed_metadata)

            return {"content_description": description, "metadata": base_metadata}

        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            # Fallback to simple analysis
            return cls._fallback_analysis(parsed_content, file_metadata, analysis_type)

    @classmethod
    async def _analyze_image_with_vision_llm(
        cls,
        llm,
        file_path: str,
        original_name: str,
        analysis_type: str,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """
        Analyze image content using vision-capable LLM
        """
        try:
            # Read and encode image
            with open(file_path, "rb") as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Build prompt based on analysis type
            if analysis_type == "quick":
                prompt = f"Analyze this image file '{original_name}' and provide a brief description of what you see. Include key objects, text (if any), colors, and overall content."
            elif custom_prompt:
                prompt = custom_prompt
            else:  # comprehensive
                prompt = f"""Provide a comprehensive analysis of this image file '{original_name}'. Include:

1. **Visual Content**: Describe what you see in detail - objects, people, scenes, etc.
2. **Text Content**: Extract and transcribe any visible text (OCR)
3. **Technical Details**: Colors, composition, style, quality
4. **Context & Purpose**: What appears to be the purpose or context of this image
5. **Key Information**: Any important data, numbers, or specific details
6. **Usability**: How this image might be used or what insights it provides

Please be thorough and specific in your analysis."""

            # For vision models like GPT-4V, Claude, or Gemini, we need to format the message properly
            if "google" in str(type(llm)).lower() or "gemini" in str(llm.model).lower():
                # Google Gemini format - use the direct LangChain approach
                from langchain.schema import HumanMessage
                from PIL import Image
                import io

                # Create PIL Image
                image = Image.open(io.BytesIO(image_data))

                # Save image temporarily for Gemini
                temp_image_path = file_path  # f"/tmp/temp_image_{uuid.uuid4().hex}.jpg"
                # logger.info(f"Saving image to {temp_image_path}")
                # image.save(temp_image_path, format="JPEG")

                try:
                    # Use the LangChain Google Gemini client
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ]
                    )
                    response = llm.invoke([message])
                    return (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

            elif "openai" in str(type(llm)).lower() or "gpt" in str(llm.model).lower():
                # OpenAI GPT-4V format
                from langchain.schema import HumanMessage

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ]
                )
                response = llm.invoke([message])
                return (
                    response.content if hasattr(response, "content") else str(response)
                )

            elif (
                "anthropic" in str(type(llm)).lower()
                or "claude" in str(llm.model).lower()
            ):
                # Claude format
                from langchain.schema import HumanMessage

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                    ]
                )
                response = llm.invoke([message])
                return (
                    response.content if hasattr(response, "content") else str(response)
                )

            else:
                # Fallback for models without vision support
                return f"Image analysis not supported for this LLM model. File: {original_name}. To enable image analysis, use a vision-capable model like GPT-4V, Claude, or Gemini."

        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            return f"Error analyzing image '{original_name}': {str(e)}"

    @classmethod
    async def _analyze_text_content_with_llm(
        cls,
        llm,
        content: str,
        file_metadata: Dict[str, Any],
        parsed_content: Dict[str, Any],
        analysis_type: str,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """
        Analyze text content using LLM
        """
        try:
            original_name = file_metadata.get("original_name", "unknown")
            file_type = file_metadata.get("file_type", "unknown")
            content_type = parsed_content.get("content_type", "unknown")

            # Build system prompt
            system_prompt = llm_clients.get_formatted_system_prompt()

            # Build user prompt based on analysis type
            if custom_prompt:
                user_prompt = f"File: {original_name} (Type: {file_type})\n\n{custom_prompt}\n\nContent:\n{content[:3000]}"
            elif analysis_type == "quick":
                user_prompt = f"""Provide a quick analysis of this {file_type} file named '{original_name}'.

Content preview:
{content[:1000]}

Please provide:
1. Brief summary of the content
2. Main topics or themes
3. Type of document/data
4. Key information extracted

Keep the analysis concise but informative."""
            else:  # comprehensive
                user_prompt = f"""Provide a comprehensive analysis of this {file_type} file named '{original_name}'.

Full content:
{content[:5000]}

Please provide detailed analysis including:
1. **Content Summary**: What this document/file contains
2. **Key Information**: Important data, facts, or insights
3. **Structure & Organization**: How the content is organized
4. **Main Topics**: Primary themes and subjects covered
5. **Data/Numbers**: Any significant quantitative information
6. **Context & Purpose**: Likely purpose and use case for this content
7. **Quality Assessment**: Completeness and usefulness of the information

Be thorough and extract as much valuable information as possible."""

            # Get response from LLM
            response = llm_clients.get_model_response(llm, system_prompt, user_prompt)
            return response

        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            return f"Error analyzing content: {str(e)}"

    @classmethod
    def _fallback_analysis(
        cls,
        parsed_content: Dict[str, Any],
        file_metadata: Dict[str, Any],
        analysis_type: str,
    ) -> Dict[str, Any]:
        """
        Fallback analysis when LLM is not available
        """
        content = parsed_content.get("content", "")
        content_type = parsed_content.get("content_type", "unknown")
        original_name = file_metadata.get("original_name", "unknown")

        # Build basic analysis
        if analysis_type == "quick":
            description = f"Quick analysis of {content_type} file: {original_name}. "
            description += f"Content length: {len(content)} characters. "
            if content_type == "structured_data":
                description += "Contains structured data suitable for analysis."
            elif content_type == "text":
                description += "Text document with readable content."
            elif content_type == "image":
                description += "Image file. For detailed analysis, ensure LLM with vision capabilities is configured."
            else:
                description += f"File of type {content_type}."
        else:
            description = f"Basic analysis of {content_type} file: {original_name}.\n\n"
            description += f"File Details:\n"
            description += f"- Size: {file_metadata.get('file_size', 0)} bytes\n"
            description += f"- Type: {file_metadata.get('file_type', 'unknown')}\n"
            description += f"- Content type: {content_type}\n\n"

            if content_type == "structured_data":
                metadata = parsed_content.get("metadata", {})
                description += f"Data Structure:\n"
                description += f"- Rows: {metadata.get('row_count', 'unknown')}\n"
                description += f"- Columns: {metadata.get('column_count', 'unknown')}\n"
                if "columns" in metadata:
                    description += (
                        f"- Column names: {', '.join(metadata['columns'][:5])}\n"
                    )
            elif content_type == "text":
                description += f"Text Content:\n"
                description += f"- Word count: {parsed_content.get('word_count', 0)}\n"
                description += (
                    f"- Character count: {parsed_content.get('character_count', 0)}\n"
                )
                if len(content) > 100:
                    description += f"- Preview: {content[:100]}...\n"
            elif content_type == "image":
                description += "Image file detected. For comprehensive image analysis including OCR and visual content description, configure an LLM with vision capabilities (GPT-4V, Claude, or Gemini)."

            description += f"\nNote: This is a basic analysis. For AI-powered insights, ensure LLM integration is properly configured."

        # Extract metadata from parsed content
        base_metadata = {
            "file_type": file_metadata.get("file_type"),
            "file_size": file_metadata.get("file_size"),
            "content_type": parsed_content.get("content_type"),
            "analysis_method": "fallback_basic",
            "word_count": parsed_content.get("word_count", 0),
            "character_count": parsed_content.get("character_count", 0),
        }

        # Add content-specific metadata
        parsed_metadata = parsed_content.get("metadata", {})
        base_metadata.update(parsed_metadata)

        return {"content_description": description, "metadata": base_metadata}

    @classmethod
    def _update_status(
        cls, file_id: str, status: FileStatus, stage: str, progress: float
    ):
        """Update processing status for a file"""
        status_info = FileAnalysisStatus(
            file_id=file_id,
            status=status,
            progress_percentage=progress,
            current_stage=stage,
            started_at=cls._status_cache.get(file_id, {}).get(
                "started_at", datetime.now()
            ),
            estimated_completion=None,  # Could calculate based on progress
        )

        cls._status_cache[file_id] = {
            "status": status_info,
            "started_at": status_info.started_at,
        }

        logger.info(f"Status update for {file_id}: {status} - {stage} ({progress}%)")

    @classmethod
    async def get_analysis(cls, file_id: str) -> Optional[FileAnalysisResponse]:
        """Get analysis results for a file"""
        return cls._analysis_cache.get(file_id)

    @classmethod
    async def get_analysis_status(cls, file_id: str) -> Optional[Dict[str, Any]]:
        """Get current analysis status for a file"""
        status_data = cls._status_cache.get(file_id)
        if status_data:
            return status_data["status"].dict()
        return None

    @classmethod
    async def delete_analysis(cls, file_id: str) -> bool:
        """Delete analysis results for a file"""
        removed_analysis = cls._analysis_cache.pop(file_id, None)
        removed_status = cls._status_cache.pop(file_id, None)
        return removed_analysis is not None or removed_status is not None
