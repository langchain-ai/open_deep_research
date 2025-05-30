# AI Engineer Pitch Deck Generator

This extension of the Open Deep Research framework creates compelling pitch decks for AI Makerspace Bootcamp students, helping them deliver clear, impactful presentations for Demo Day.

## Overview

The AI Engineer Pitch Deck Generator leverages LangGraph's structured workflow to create persuasive pitch decks tailored to different audiences (investors, executives, users). It builds on the research capabilities of Open Deep Research while adapting the output format for pitch presentations.

![Pitch Deck Generator Workflow](https://github.com/user-attachments/assets/3c734c3c-57aa-4bc0-85dd-74e2ec2c0880)

## Features

- **Structured Pitch Format**: Generates 5-7 slides covering key elements (Problem, Solution, Market, etc.)
- **Audience-Focused Content**: Tailors messaging to address executives' risks, users' pain points, and team strengths
- **Research-Backed Claims**: Uses web search to gather relevant market data and validate assertions
- **Visual Suggestions**: Recommends charts, mockups, or diagrams to enhance each slide
- **Discord Integration**: Formats pitch content for sharing and gathering community feedback
- **Human Feedback Loop**: Allows for iterative refinement based on stakeholder input
- **MCP Integration**: Optional support for specialized tools via Model Context Protocol

## Key Components

The implementation consists of several specialized files:

- **pitch_graph.py**: LangGraph workflow for pitch deck generation
- **pitch_state.py**: State definitions for the pitch deck generation process
- **pitch_mcp.py**: Optional MCP server with specialized pitch deck tools
- **pitch_graph.ipynb**: Example notebook demonstrating the pitch deck generator

## Project Background & Design Intent

The AI Engineer Pitch Deck Generator addresses a critical challenge for bootcamp students: creating concise, persuasive pitch decks aligned with the perspectives of executives, users, and team capabilities. By combining the structured workflow of LangGraph with specialized prompts and optional MCP tools, students can quickly create professional-quality pitch decks that:

1. **Highlight Critical Business Value**: Address the "executive nightmare" that decision-makers care about
2. **Emphasize User Impact**: Show deep understanding of user pain points and experiences
3. **Showcase Team Strengths**: Position the team's unique capabilities as the ideal solution
4. **Drive Engagement**: Maintain the right balance of technical depth and strategic vision

Success metrics include 80% of students delivering pitches scoring 4/5 for clarity and impact, significant reduction in preparation time, and higher confidence levels among presenting students.

## Usage

### Basic Usage

1. Import the pitch deck generator:

```python
from open_deep_research.pitch_graph import graph
```

2. Configure the generator:

```python
config = {
    "configurable": {
        "writer_provider": "anthropic",
        "writer_model": "claude-3-5-sonnet-latest",
        "search_api": "tavily",
        "number_of_queries": 2,
        "include_visuals": True,
        "discord_integration": True
    }
}
```

3. Generate a pitch deck:

```python
result = await graph.ainvoke(
    {"topic": "AI-powered pitch deck generator for bootcamp students"},
    config=config
)
```

### Advanced Usage with MCP

For enhanced capabilities, you can enable MCP integration:

1. Start the MCP server:

```bash
python -m open_deep_research.pitch_mcp
```

2. Configure the generator to use MCP:

```python
config = {
    "configurable": {
        # Basic configuration
        "writer_provider": "anthropic",
        "writer_model": "claude-3-5-sonnet-latest",
        
        # MCP configuration
        "enable_mcp": True,
        "mcp_server_url": "http://localhost:8000"
    }
}
```

3. Generate a pitch deck with MCP-enhanced capabilities:

```python
result = await graph.ainvoke(
    {"topic": "AI-powered pitch deck generator for bootcamp students"},
    config=config
)
```

## Design Choices

The AI Engineer Pitch Deck Generator builds on the graph-based workflow of Open Deep Research, which was determined to be the superior approach for this use case because:

1. **Human Feedback Integration**: Essential for refining pitch decks through iterative feedback
2. **Structured Output Format**: Perfect for the standardized slide structure of pitch decks
3. **Quality-Focused Approach**: Prioritizes accuracy and impact over generation speed
4. **Audience-Tailored Content**: Enables focused messaging for different stakeholder types

## Customization

The generator can be customized in several ways:

- **Slide Structure**: Modify the `pitch_planner_instructions` to change slide types or focus
- **Content Style**: Adjust `slide_writer_instructions` to change the tone or content format
- **Visual Suggestions**: Update `visual_suggestion_instructions` for different visual styles
- **Discord Formatting**: Customize `discord_post_instructions` to match your community style
- **MCP Tools**: Extend `pitch_mcp.py` with additional specialized tools

## Implementation Details

### State Definitions

The `pitch_state.py` module defines the core data structures:

```python
class Slide(BaseModel):
    """A slide in a pitch deck"""
    name: str
    description: str
    research: bool
    content: str = ""
    visual_suggestion: Optional[str] = None
    
class Slides(BaseModel):
    """Collection of slides for a pitch deck"""
    slides: List[Slide]

class Tagline(BaseModel):
    """Tagline for a pitch deck"""
    content: str

class DiscordPost(BaseModel):
    """Format for a Discord post about the pitch deck"""
    title: str
    message: str
    picture_suggestion: str
```

### Prompt Templates

The generator uses specialized prompts for each step:

- **Pitch Planner**: Creates a 5-7 slide structure addressing executive, user, and team perspectives
- **Tagline Generator**: Creates a memorable 5-10 word tagline capturing the core value proposition
- **Slide Writer**: Produces concise (50-100 word) persuasive slide content
- **Slide Grader**: Evaluates slides on clarity, conciseness, impact, relevance, and persuasiveness
- **Visual Suggester**: Recommends tailored visual elements for each slide
- **Discord Formatter**: Structures content for community sharing

### Graph Workflow

The workflow in `pitch_graph.py` orchestrates the generation process through several nodes:

1. **generate_pitch_plan**: Creates the initial slide structure
2. **human_feedback**: Gets user input on plan quality
3. **process_slides_for_research**: Identifies slides needing web research
4. **process_all_slides**: Generates content for each slide
5. **grade_all_slides**: Evaluates slide quality and suggests improvements
6. **generate_tagline**: Creates a catchy project tagline
7. **generate_discord_post**: Formats content for community sharing

### MCP Integration

The optional Model Context Protocol integration (`pitch_mcp.py`) provides specialized tools:

```python
class PitchDeckMCPServer(MCPServer):
    """MCP server with specialized tools for pitch deck generation."""
    
    async def fetch_market_data(self, query: str, slide_type: str) -> str:
        """Fetch market data relevant to a particular slide."""
        # Implementation that connects to market data sources
        
    async def generate_slide_visuals(self, slide_content: str, slide_type: str) -> str:
        """Generate visual suggestions for a slide based on its content."""
        # Implementation that suggests appropriate visuals
        
    async def format_for_discord(self, pitch_content: Dict[str, Any]) -> Dict[str, str]:
        """Format pitch content for Discord sharing."""
        # Implementation that optimizes Discord formatting
        
    async def get_pitch_templates(self, pitch_type: str = "demo_day") -> List[Dict[str, Any]]:
        """Retrieve pitch deck templates and examples."""
        # Implementation that provides tailored templates
        
    async def role_based_feedback(self, slide_content: str, role: str) -> str:
        """Generate feedback from different stakeholder perspectives."""
        # Implementation that simulates feedback from different roles
```

## Best Practices for Pitch Decks

The generator embodies these pitch deck best practices:

1. **Executive Focus**: Address the "nightmare" keeping decision-makers up at night
   - Identify critical business risks being solved
   - Quantify the cost of inaction
   - Show clear ROI or strategic advantage

2. **User-Centered Storytelling**: Connect with the human journey
   - Highlight emotional and practical pain points
   - Show transformation through your solution
   - Use relatable scenarios over abstract concepts

3. **Team Credibility**: Establish why your team is uniquely qualified
   - Highlight domain expertise and technical capabilities
   - Demonstrate track record in relevant areas
   - Show complementary skills across team members

4. **Visual Impact**: Use visuals to enhance not replace messaging
   - One key visual concept per slide
   - Data visualizations for market/impact slides
   - Solution mockups or diagrams for technical slides

5. **Presentation Delivery**:
   - 3 minutes maximum verbal presentation
   - Practice the "elevator pitch" version (30 seconds)
   - Prepare for common questions from judges

## Success Metrics

The AI Engineer Pitch Deck Generator is designed to help bootcamp students achieve:

- 80% of students deliver pitches scoring 4/5 or higher for clarity and impact
- 70% reduction in time spent creating pitch materials
- 90% of pitches effectively address all three key perspectives
- 75% of students report increased confidence in pitch delivery

## Examples

### Example Pitch Structure

```
Slide 1: Problem
The $200B enterprise software market suffers from a 62% implementation failure rate, with companies losing an average of $680K per failed project. The primary cause? Complex deployment processes that IT teams can't efficiently navigate.

Slide 2: Solution
DeployBuddy is an AI-powered deployment assistant that transforms complex software implementation into guided workflows. Our system analyzes documentation, creates custom deployment plans, and provides real-time troubleshooting—reducing deployment time by 58%.

Slide 3: Market
The deployment automation market is growing at 22% CAGR, reaching $15B by 2026. Our initial target: mid-market enterprises (500-2000 employees) spending $120K+ annually on implementation services, representing a $4.3B serviceable market.

Slide 4: Impact
Early customers report: 72% faster deployment times, 81% reduction in support tickets, and 68% lower cost of implementation. For a typical mid-market company, that's $86K saved per major software deployment.

Slide 5: Team
Our founding team combines 25+ years of enterprise deployment experience with ML engineering expertise. Jane (CEO) led implementation at ServiceNow, Alex (CTO) built deployment automation at Microsoft, and Sam (ML Lead) developed troubleshooting systems at Google.

Slide 6: Call to Action
We're seeking $500K to expand our ML capabilities and onboard 20 enterprise customers in the next 12 months. With your support, we'll transform software deployment from a nightmare to a competitive advantage.
```

### Example Taglines

- "Deploy Fast, Fail Never"
- "Implementation Nightmares, Meet Your Match"
- "Complex Software, Simple Deployment"
- "From Documentation to Deployment in Minutes"

## Next Steps

Future enhancements could include:

1. **Image Generation**: Integrate with DALL-E or other image generation tools to create actual visuals
2. **Slide Export**: Add PowerPoint or Google Slides export functionality
3. **Practice Coach**: Add a module for rehearsal feedback and delivery tips
4. **Expanded MCP Tools**: More specialized tools for competitive analysis, investor research, etc.
5. **Role-Playing Simulation**: Generate potential investor questions and practice responses
6. **Multilingual Support**: Generate pitch decks in multiple languages
7. **Industry-Specific Templates**: Pre-configured templates for different AI application domains
8. **Integration with Demo Tools**: Connect with code demonstration or product mockup tools
