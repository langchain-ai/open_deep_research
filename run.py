import uvicorn
from src.open_deep_research.api.app import app

if __name__ == "__main__":
    uvicorn.run(
        "src.open_deep_research.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    ) 