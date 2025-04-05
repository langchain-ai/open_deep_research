# src/open_deep_research/api/app.py

import os
import json
import uuid
import asyncio
import pickle
from typing import Dict, List, Any, AsyncIterable
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import shutil

# Import LangGraph components
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt
# Import open_deep_research components
import sys
sys.path.append(".")  # Add current directory to path to ensure imports work

try:
    from open_deep_research.graph import builder
    from open_deep_research.document_processing.loaders import PDFLoader, ExcelLoader
    from open_deep_research.document_processing.vector_store import FAISSVectorStore
    from open_deep_research.document_processing.rag import HybridRetrievalSystem
    from open_deep_research.document_processing.graph_integration import LocalDocumentConfig, process_local_documents
except ImportError as e:
    print(f"Error importing open_deep_research components: {e}")
    raise

app = FastAPI(title="Enhanced Open Deep Research API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory saver for graph checkpoints
memory = MemorySaver()

# Compile graph
graph = builder.compile(checkpointer=memory)

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")

class ReportRequest(BaseModel):
    """Report request model"""
    topic: str = Field(..., description="Topic for the report")
    search_api: str = Field(default="tavily", description="Search API to use")
    planner_provider: str = Field(default="anthropic", description="Provider for planner model")
    planner_model: str = Field(default="claude-3-7-sonnet-latest", description="Model for planning")
    writer_provider: str = Field(default="anthropic", description="Provider for writer model")
    writer_model: str = Field(default="claude-3-5-sonnet-latest", description="Model for writing")
    max_search_depth: int = Field(default=1, description="Maximum search depth")
    report_structure: str = Field(default=None, description="Custom report structure")
    local_documents: LocalDocumentConfig = Field(
        default_factory=LocalDocumentConfig,
        description="Configuration for local documents"
    )

class ChatRequest(BaseModel):
    """Chat request model"""
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    stream: bool = Field(default=True, description="Whether to stream the response")
    local_documents: LocalDocumentConfig = Field(
        default_factory=LocalDocumentConfig,
        description="Configuration for local documents"
    )

async def create_thread() -> Dict[str, Any]:
    """Create a new thread ID"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Open Deep Research API!"}


@app.post("/api/report")
async def generate_report(request: ReportRequest):
    """Generate a report on a given topic with streaming"""
    
    # Create thread configuration
    thread = {"configurable": {
        "thread_id": str(uuid.uuid4()),
        "search_api": request.search_api,
        "planner_provider": request.planner_provider,
        "planner_model": request.planner_model,
        "writer_provider": request.writer_provider,
        "writer_model": request.writer_model,
        "max_search_depth": request.max_search_depth,
        "local_documents": request.local_documents.dict()
    }}
    
    # Add custom report structure if provided
    if request.report_structure:
        thread["configurable"]["report_structure"] = request.report_structure
    
    # Process local documents if provided
    if request.local_documents.enabled and request.local_documents.paths:
        try:
            process_local_documents(request.local_documents)
        except Exception as e:
            return {"error": f"Error processing local documents: {str(e)}"}
    
    # Stream function to convert graph events to Vercel AI SDK compatible format
    async def stream_events():
        # Send initial data to conform to Vercel AI SDK protocol
        yield f"data: {json.dumps({'type': 'stream_start'})}\n\n"
        
        try:
            # Stream report planning phase
            plan_text = ""
            async for event in graph.astream({"topic": request.topic}, thread, stream_mode="updates"):
                if "report_plan" in event:
                    plan_text = event["report_plan"]
                    chunk = {"type": "text", "text": "Planning report...\n\n" + plan_text}
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            # Stream report writing phase
            await asyncio.sleep(0.5)  # Small delay for better UX
            msg = json.dumps({'type': 'text', 'text': '\n\nGenerating report...\n\n'})
            yield "data: " + msg + "\n\n"

            
            # Generate the report content
            from langgraph.types import Command
            full_report = ""
            async for event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
                if "report" in event:
                    new_content = event["report"].replace(full_report, "", 1) if full_report else event["report"]
                    full_report = event["report"]
                    
                    if new_content:
                        chunk = {"type": "text", "text": new_content}
                        yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final completion message
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
        except Exception as e:
            # Handle errors
            error_msg = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_msg)}\n\n"
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
    
    # Return streaming response
    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream"
    )

@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload multiple documents for processing"""
    # Create uploads directory if it doesn't exist
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    processed_files = []
    errors = []
    
    for file in files:
        if not file.filename.endswith(('.pdf', '.xlsx', '.xls')):
            errors.append(f"Unsupported file type: {file.filename}")
            continue
            
        try:
            # Save file
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            processed_files.append(file_path)
        except Exception as e:
            errors.append(f"Error saving {file.filename}: {str(e)}")
    
    if not processed_files:
        raise HTTPException(
            status_code=400,
            detail="No valid files were uploaded. Supported formats: PDF, XLSX, XLS"
        )
    
    # Process all uploaded documents
    config = LocalDocumentConfig(
        enabled=True,
        paths=processed_files,
        vector_store_dir="vector_store"
    )
    
    try:
        process_local_documents(config)
        return {
            "status": "success",
            "message": f"Successfully processed {len(processed_files)} documents",
            "processed_files": [os.path.basename(f) for f in processed_files],
            "errors": errors if errors else None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing documents: {str(e)}",
            "processed_files": [os.path.basename(f) for f in processed_files],
            "errors": errors if errors else None
        }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with local document RAG support"""
    
    # Get the user's latest message
    latest_message = next((m for m in reversed(request.messages) if m.role == "user"), None)
    if not latest_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Prepare query for RAG
    query = latest_message.content
    
    # Process local documents if enabled
    vector_store = None
    if request.local_documents.enabled:
        try:
            vector_store = process_local_documents(request.local_documents)
        except Exception as e:
            return {"error": f"Error processing local documents: {str(e)}"}
    
    # Stream function to convert events to Vercel AI SDK compatible format
    async def stream_events():
        # Send initial data to conform to Vercel AI SDK protocol
        yield f"data: {json.dumps({'type': 'stream_start'})}\n\n"
        
        try:
            # Perform local search if vector store is available
            local_docs = []
            if vector_store:
                local_docs = vector_store.search(query, k=5)
            
            # Get web search results using the open_deep_research search
            # We'll need to adapt this part based on the actual search implementation
            web_docs = []
            
            # For now, we'll just use local docs if available
            docs = local_docs + web_docs
            
            # Generate context from documents
            context = "\n\n".join([doc.page_content for doc in docs[:5]])
            
            # Prepare final prompt with context and query
            prompt = f"""Use the following information to answer the user's question:

Context:
{context}

User Question: {query}

Answer:"""
            
            # Generate response with an LLM
            # We'll use the same model as specified in the request
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
            
            # Stream the response
            response = ""
            async for chunk in model.astream(prompt):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if content:
                        response += content
                        chunk_data = {"type": "text", "text": content}
                        yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send final completion message
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
        except Exception as e:
            # Handle errors
            error_msg = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_msg)}\n\n"
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
    
    # Return streaming response
    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)