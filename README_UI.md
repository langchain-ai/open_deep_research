# Deep Research Agent UI

A simple web-based user interface for interacting with the Open Deep Research agent API.

## Features

- **Research Query Input**: Enter your research topic or question
- **Implementation Selection**: Choose between workflow-based or multi-agent implementations
- **Search API Selection**: Select from various search APIs (Tavily, Perplexity, Exa, etc.)
- **Real-time Progress Tracking**: Monitor research progress with a progress bar
- **Thinking Steps Visualization**: View detailed thinking steps with emojis for different agents/tools
- **Formatted Report Display**: See the final research report with proper markdown formatting
- **Tabbed Interface**: Switch between thinking steps and final report views

## Setup Instructions

1. **Prerequisites**:
   - The Open Deep Research agent API must be running locally
   - A modern web browser (Chrome, Firefox, Edge, etc.)

2. **Start the API Server**:
   ```bash
   # Install dependencies and start the LangGraph server
   uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
   ```
   
   Or for Windows/Linux:
   ```bash
   # Install dependencies 
   pip install -e .
   pip install -U "langgraph-cli[inmem]" 
   
   # Start the LangGraph server
   langgraph dev
   ```

3. **Launch the UI**:
   - Simply open the `index.html` file in your web browser
   - Alternatively, you can serve it using a simple HTTP server:
     ```bash
     # Using Python's built-in HTTP server
     python -m http.server
     ```
     Then navigate to `http://localhost:8000` in your browser

## Usage Guide

1. **Enter a Research Topic**:
   - Type your research question or topic in the text area
   - Be specific to get the best results

2. **Select Implementation Type**:
   - **Workflow-based (Graph)**: Uses a structured plan-and-execute workflow
   - **Multi-Agent**: Uses a supervisor-researcher architecture with parallel processing

3. **Choose a Search API**:
   - Select from various search providers (Tavily, Perplexity, Exa, ArXiv, PubMed, etc.)
   - Different APIs may be better suited for different types of research

4. **Start Research**:
   - Click the "Start Research" button to begin the process
   - The UI will create a new thread and start the research run

5. **Monitor Progress**:
   - The progress bar shows the estimated completion percentage
   - Thinking steps are displayed in real-time with emojis indicating different agents/tools
   - Each step includes a timestamp and detailed information

6. **View Results**:
   - When research is complete, the final report will be available in the "Final Report" tab
   - You can switch between tabs to view thinking steps and the final report
   - The report is formatted with proper markdown rendering

7. **Cancel Research**:
   - If needed, you can cancel the research process by clicking the "Cancel Research" button

## Agent/Tool Emoji Legend

- 🧠 Supervisor - Manages the overall research process
- 🔍 Researcher - Conducts research on specific topics
- 📝 Planner - Plans the structure and content of the report
- ✍️ Writer - Writes and formats report sections
- 🌐 Search Tool - Performs web searches for information
- 📄 Section Builder - Creates and organizes report sections
- ❓ Query Generator - Creates search queries
- 📊 Feedback Analyzer - Evaluates and improves content
- 🤖 Default - Other agents or tools

## Troubleshooting

- **API Connection Issues**: Ensure the API server is running at `http://127.0.0.1:2024`
- **CORS Errors**: If you encounter CORS issues, try using a CORS browser extension or proxy
- **Long-running Research**: Complex topics may take several minutes to complete
- **Browser Compatibility**: If you experience issues, try using a different browser

## Notes

- This UI is designed for local use and interacts with the API running on the same machine
- No data is sent to external servers beyond what the research agent itself uses
- The UI does not require any server-side components beyond the existing API