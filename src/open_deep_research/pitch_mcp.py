from typing import Dict, Any, Optional, List
import os
import sys
import asyncio

from modelcontextprotocol.servers import MCPServer
from modelcontextprotocol.servers.tools import Tool

class PitchDeckMCPServer(MCPServer):
    """MCP server with specialized tools for pitch deck generation."""
    
    def __init__(self):
        """Initialize the MCP server with pitch deck tools."""
        super().__init__("pitch-deck-tools")
        
        # Register tools
        self.register_tool(
            Tool(
                name="fetch_market_data",
                description="Fetches relevant market data for a pitch deck slide",
                function=self.fetch_market_data
            )
        )
        
        self.register_tool(
            Tool(
                name="generate_slide_visuals",
                description="Suggests appropriate visuals for pitch deck slides",
                function=self.generate_slide_visuals
            )
        )
        
        self.register_tool(
            Tool(
                name="format_for_discord",
                description="Formats pitch content for Discord sharing",
                function=self.format_for_discord
            )
        )
        
        self.register_tool(
            Tool(
                name="get_pitch_templates",
                description="Retrieves pitch deck templates and examples",
                function=self.get_pitch_templates
            )
        )
        
        self.register_tool(
            Tool(
                name="role_based_feedback",
                description="Provides feedback on slides from different perspectives (investor, executive, user)",
                function=self.role_based_feedback
            )
        )
    
    async def fetch_market_data(self, query: str, slide_type: str) -> str:
        """
        Fetch market data relevant to a particular slide.
        
        Args:
            query: The search query for market data
            slide_type: The type of slide (e.g., "Market", "Impact")
            
        Returns:
            Formatted market data as a string
        """
        # TODO: Implement actual market data fetching
        # This could connect to databases, APIs, or specialized search
        
        sample_data = {
            "Market": {
                "AI tools market": "The global AI tools market is projected to reach $120B by 2026, growing at a CAGR of 25.6%.",
                "Educational AI": "The AI in education sector is expected to grow from $1.8B in 2023 to $12B by 2028.",
                "Pitch deck tools": "Over 85% of successful startups use specialized tools for pitch creation."
            },
            "Impact": {
                "Student success": "Students with professional pitch training are 3.7x more likely to secure funding.",
                "Time savings": "AI tools reduce pitch preparation time by 68% on average.",
                "Quality improvement": "AI-assisted pitches score 42% higher on clarity and persuasiveness metrics."
            }
        }
        
        # Return sample data based on slide type
        if slide_type in sample_data and query.lower() in sample_data[slide_type]:
            return sample_data[slide_type][query.lower()]
        
        return f"Market data for {query} related to {slide_type} slides: [Sample market statistics would be inserted here]"
    
    async def generate_slide_visuals(self, slide_content: str, slide_type: str) -> str:
        """
        Generate visual suggestions for a slide based on its content.
        
        Args:
            slide_content: The content of the slide
            slide_type: The type of slide (e.g., "Problem", "Solution")
            
        Returns:
            Visual suggestion as a string
        """
        # TODO: Implement actual visual generation or suggestions
        # This could connect to image generation APIs or databases of visuals
        
        visual_suggestions = {
            "Problem": "Bar chart showing the frequency and impact of the problem across different user segments.",
            "Solution": "Simplified workflow diagram showing before/after process with your solution highlighted.",
            "Market": "Growth trend line with TAM/SAM/SOM visualization using concentric circles.",
            "Competition": "2x2 matrix positioning your solution against competitors on key value dimensions.",
            "Why Us": "Team photo with experience highlights or skills matrix showing complementary expertise.",
            "Call to Action": "Timeline roadmap showing investment milestones and projected growth."
        }
        
        if slide_type in visual_suggestions:
            return visual_suggestions[slide_type]
        
        return f"Recommended visual for '{slide_type}' slide: [Visual suggestion based on content analysis]"
    
    async def format_for_discord(self, pitch_content: Dict[str, Any]) -> Dict[str, str]:
        """
        Format pitch content for Discord sharing.
        
        Args:
            pitch_content: Dictionary with pitch content including tagline, problem, solution
            
        Returns:
            Formatted Discord post as a dictionary
        """
        # Extract pitch content
        tagline = pitch_content.get("tagline", "")
        problem = pitch_content.get("problem", "")
        solution = pitch_content.get("solution", "")
        
        # Format for Discord
        title = f"🚀 {tagline}"
        message = f"**Problem:** {problem}\n\n**Solution:** {solution}\n\n💬 Would love your feedback on this pitch! What resonates most with you?"
        picture_suggestion = "Consider including a mockup of your solution or a key visual from your deck."
        
        return {
            "title": title,
            "message": message,
            "picture_suggestion": picture_suggestion
        }
    
    async def get_pitch_templates(self, pitch_type: str = "demo_day") -> List[Dict[str, Any]]:
        """
        Retrieve pitch deck templates and examples.
        
        Args:
            pitch_type: The type of pitch (e.g., "demo_day", "investor", "executive")
            
        Returns:
            List of template structures
        """
        templates = {
            "demo_day": [
                {
                    "name": "Problem-Solution-Impact",
                    "description": "5-slide format focused on problem clarity and solution impact",
                    "structure": [
                        {"name": "Problem", "description": "Clear statement of the problem and who experiences it"},
                        {"name": "Solution", "description": "Your innovative approach to solving the problem"},
                        {"name": "Market", "description": "Size of the opportunity and who will pay"},
                        {"name": "Why Us", "description": "Your team's unique abilities to execute"},
                        {"name": "Call to Action", "description": "What you're seeking from the audience"}
                    ]
                },
                {
                    "name": "Hook-Challenge-Solution-Proof",
                    "description": "7-slide storytelling format with strong evidence focus",
                    "structure": [
                        {"name": "Hook", "description": "Attention-grabbing opener with tagline"},
                        {"name": "Problem", "description": "Customer pain point with evidence"},
                        {"name": "Solution", "description": "Your approach with key differentiators"},
                        {"name": "How It Works", "description": "Simple explanation of your technology"},
                        {"name": "Market", "description": "Size and growth of target market"},
                        {"name": "Traction", "description": "Evidence of early success or validation"},
                        {"name": "Ask", "description": "Specific request for resources or support"}
                    ]
                }
            ]
        }
        
        return templates.get(pitch_type, [])
    
    async def role_based_feedback(self, slide_content: str, role: str) -> str:
        """
        Generate feedback from different stakeholder perspectives.
        
        Args:
            slide_content: The content of the slide
            role: The perspective to provide feedback from ("investor", "executive", "user")
            
        Returns:
            Role-specific feedback as a string
        """
        feedback_frames = {
            "investor": {
                "focus": ["market size", "scalability", "business model", "return potential"],
                "questions": [
                    "How big is the addressable market?",
                    "What's the path to profitability?",
                    "How defensible is your solution?"
                ]
            },
            "executive": {
                "focus": ["implementation cost", "risk mitigation", "ROI", "integration"],
                "questions": [
                    "What resources are required?",
                    "How does this address my current business challenges?",
                    "What are the risks of implementation?"
                ]
            },
            "user": {
                "focus": ["user experience", "pain points", "learning curve", "value"],
                "questions": [
                    "How does this make my life easier?",
                    "How long will it take to see results?",
                    "Is this better than my current solution?"
                ]
            }
        }
        
        if role in feedback_frames:
            frame = feedback_frames[role]
            focus_points = ", ".join(frame["focus"])
            sample_question = frame["questions"][0]
            
            return f"From a {role}'s perspective, this slide should focus more on {focus_points}. Consider addressing: '{sample_question}'"
        
        return f"Feedback from {role} perspective: [Role-specific feedback would be generated here]"

async def start_server(host: str = "localhost", port: int = 8000):
    """Start the MCP server."""
    server = PitchDeckMCPServer()
    await server.start(host=host, port=port)
    
    print(f"🚀 Pitch Deck MCP Server running at http://{host}:{port}")
    print("Available tools:")
    for tool in server.list_tools():
        print(f"- {tool.name}: {tool.description}")
    
    try:
        # Keep the server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await server.stop()
        print("Server stopped.")

if __name__ == "__main__":
    # Get host and port from command line arguments or use defaults
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    # Start the server
    asyncio.run(start_server(host, port))
