from typing import Annotated, List, Optional, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

# Import from the existing state module to reuse common components
from open_deep_research.state import SearchQuery, Queries, Feedback

class Slide(BaseModel):
    """A slide in a pitch deck"""
    name: str = Field(
        description="Name/title for this slide in the pitch deck.",
    )
    description: str = Field(
        description="Brief overview of what this slide should cover.",
    )
    research: bool = Field(
        description="Whether to perform web research for this slide."
    )
    content: str = Field(
        description="The content of the slide.",
        default=""
    )
    visual_suggestion: Optional[str] = Field(
        description="Suggestion for a visual element to enhance the slide.",
        default=None
    )

class Slides(BaseModel):
    """Collection of slides for a pitch deck"""
    slides: List[Slide] = Field(
        description="Slides of the pitch deck.",
    )

class Tagline(BaseModel):
    """Tagline for a pitch deck"""
    content: str = Field(
        description="A short, bold, memorable tagline that captures the essence of the project."
    )

class DiscordPost(BaseModel):
    """Format for a Discord post about the pitch deck"""
    title: str = Field(
        description="Title for the Discord post (typically the tagline)."
    )
    message: str = Field(
        description="Main content for the Discord post, highlighting problem and solution."
    )
    picture_suggestion: str = Field(
        description="Suggestion for an image to include with the post."
    )

class PitchStateInput(TypedDict):
    """Input for the pitch deck generator"""
    topic: str  # Pitch topic

class PitchStateOutput(TypedDict):
    """Output from the pitch deck generator"""
    final_pitch: str  # The complete pitch deck
    discord_post: dict  # Formatted Discord post

class PitchState(TypedDict):
    """State for the pitch deck generation process"""
    topic: str  # Pitch topic
    feedback: Optional[str]  # Feedback on the pitch plan
    slides: list[Slide]  # List of slides in the pitch deck
    completed_slides: Annotated[list, operator.add]  # Slides that have been completed
    tagline: Optional[str]  # Tagline for the pitch
    discord_post: Optional[dict]  # Formatted post for Discord
    final_pitch: Optional[str]  # The complete pitch deck

class SlideState(TypedDict):
    """State for individual slide creation"""
    topic: str  # Pitch topic
    slide: Slide  # The slide being worked on
    search_iterations: int  # Number of search iterations done
    search_queries: list[SearchQuery]  # List of search queries
    source_str: str  # String of formatted source content from web search
    completed_slides: list[Slide]  # Final key we duplicate in outer state for Send() API
    current_slide: Optional[Slide]  # The slide currently being enhanced

class SlideOutputState(TypedDict):
    """Output from slide creation"""
    completed_slides: list[Slide]  # Completed slides to add to the main state
