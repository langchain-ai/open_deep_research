# AI Engineer Pitch Generator Tutorial

This tutorial walks through how to use the AI Engineer Pitch Deck Generator to create persuasive pitch decks for Demo Day presentations. Follow this guide to understand the key components, configuration options, and workflow of the pitch generator.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setting Up](#setting-up)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Customization](#customization)
6. [Best Practices](#best-practices)

## Project Overview

The AI Engineer Pitch Deck Generator helps bootcamp students create compelling pitch decks by:

- Generating structured 5-7 slide pitch decks
- Focusing content on executive risks, user pain points, and team strengths
- Providing visual suggestions for each slide
- Formatting content for Discord sharing
- Incorporating human feedback

## Setting Up

### Installation

To use the AI Engineer Pitch Deck Generator, ensure you have the Open Deep Research package installed:

```bash
# Install from GitHub repository
git clone https://github.com/yourusername/open_deep_research.git
cd open_deep_research
pip install -e .

# Or install using pip if published
pip install open-deep-research
```

### Configuration

You'll need API keys for the LLMs and search tools:

```bash
# Configure API keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

## Basic Usage

### Create a Simple Pitch Deck

```python
from open_deep_research.pitch_graph import graph
from open_deep_research.configuration import SearchAPI

# Define configuration
config = {
    "configurable": {
        "writer_provider": "anthropic",
        "writer_model": "claude-3-5-sonnet-latest",
        "planner_provider": "anthropic",
        "planner_model": "claude-3-7-sonnet-latest",
        "search_api": SearchAPI.TAVILY,
        "number_of_queries": 2,
        "include_visuals": True,
        "discord_integration": True
    }
}

# Define your project topic
project_topic = "AI-powered coding assistant for bootcamp students"

# Generate the pitch deck
async def generate_pitch():
    result = await graph.ainvoke(
        {"topic": project_topic},
        config=config
    )
    return result

# Run the generation
pitch_result = await generate_pitch()

# Access the components of the pitch
slides = pitch_result["slides"]
tagline = pitch_result["tagline"]
discord_post = pitch_result["discord_post"]
```

### Display the Pitch Deck

```python
# Display the complete pitch deck
from IPython.display import Markdown

Markdown(pitch_result["final_pitch"])

# Display just the tagline
print(f"Tagline: {tagline}")

# Display each slide individually
for i, slide in enumerate(slides, 1):
    print(f"Slide {i}: {slide.name}")
    print(f"Content: {slide.content}")
    if slide.visual_suggestion:
        print(f"Visual: {slide.visual_suggestion}")
    print("---")
```

## Advanced Features

### MCP Integration

For enhanced capabilities, you can use the Model Context Protocol integration:

```python
# Start the MCP server in a separate terminal
# $ python -m open_deep_research.pitch_mcp

# Enable MCP in your configuration
config["configurable"]["enable_mcp"] = True
config["configurable"]["mcp_server_url"] = "http://localhost:8000"

# Use MCP tools directly
from modelcontextprotocol.client import MCPClient
from modelcontextprotocol.schema import ExecuteToolParams

# Connect to MCP server
mcp_client = MCPClient(
    server_url="http://localhost:8000",
    client_name="pitch-deck-example"
)

# Get available tools
tools = await mcp_client.list_tools()
print("Available MCP tools:")
for tool in tools:
    print(f"- {tool.name}: {tool.description}")

# Example: Use market data tool
market_data_params = ExecuteToolParams(
    name="fetch_market_data",
    arguments={
        "query": "AI education market size",
        "slide_type": "Market"
    }
)
market_data_result = await mcp_client.execute_tool(market_data_params)
print(market_data_result.content)
```

### Human Feedback Integration

The pitch deck generator supports human feedback. When running the full workflow, it will pause for feedback on the pitch plan:

```python
# This will pause for feedback during execution
feedback_result = await graph.ainvoke(
    {"topic": project_topic},
    config=config
)
```

## Customization

### Modifying Slide Structure

You can customize the slide structure by modifying the configuration:

```python
# Custom slide structure
config["configurable"]["pitch_structure"] = """
1. Problem/Opportunity
2. Solution Overview
3. Technology Architecture 
4. Market Analysis
5. Competitive Advantage
6. Team
7. Call to Action
"""

# Generate with custom structure
custom_result = await graph.ainvoke(
    {"topic": project_topic},
    config=config
)
```

### Adjusting Content Style

To change the content style (more technical, more business-focused, etc.):

```python
# Adjust content style
from open_deep_research.prompts import slide_writer_instructions

# Make a more technical version
technical_instructions = slide_writer_instructions.replace(
    "Use simple, bold language that executives can understand",
    "Include technical details and architecture insights while maintaining clarity"
)

# Use custom prompts (advanced)
from langchain.prompts import PromptTemplate
custom_prompt = PromptTemplate.from_template(technical_instructions)
# Then use custom_prompt in your workflow
```

## Best Practices

### Creating Effective Pitch Decks

Follow these best practices for Demo Day success:

1. **Focus on the Problem**: Make sure the problem slide is compelling and relatable
2. **Quantify Your Claims**: Include specific metrics and data points when possible
3. **Simplify Technical Concepts**: Use analogies and visuals for complex ideas
4. **Practice Your Delivery**: Use the generated content as a starting point for practice
5. **Incorporate Feedback**: Use the feedback mechanism to refine your pitch

### Example Workflow

An ideal workflow for using the generator:

1. Generate an initial pitch deck
2. Review and provide feedback to refine
3. Export to your presentation tool of choice
4. Create visuals based on the suggestions
5. Practice delivery with the slide content as script
6. Share on Discord to get peer feedback
7. Refine based on community input

## Conclusion

The AI Engineer Pitch Deck Generator is a powerful tool for bootcamp students to create professional-quality pitch decks quickly. By combining structured workflows, research capabilities, and audience-focused content, it helps students deliver compelling presentations that address executive concerns, user needs, and team capabilities.
