"""
Configuration classes for the Technical Due Diligence (TDD) Agent System.

This module defines the configuration classes used to customize the
behavior of the TDD agents.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from open_deep_research.configuration import Configuration

logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
class AgentConfiguration:
    """Configuration for a specific agent in the TDD system."""
    model: str = "llama-3.3-70b-versatile"
    model_kwargs: Optional[Dict[str, Any]] = None

@dataclass(kw_only=True)
class ToolConfiguration:
    """Configuration for a specific tool in the TDD system."""
    enabled: bool = True
    parameters: Optional[Dict[str, Any]] = None

@dataclass(kw_only=True)
class TDDConfiguration(Configuration):
    """Configuration for the TDD Agent system.
    
    This class extends the base Configuration class with TDD-specific settings.
    """
    # Inherit from base Configuration
    
    # Planning agent configuration
    planning_model: str = "llama-3.3-70b-versatile"
    planning_model_kwargs: Optional[Dict[str, Any]] = None
    
    # Domain agent configurations
    tech_stack_model: str = "llama-3.3-70b-versatile"
    tech_stack_model_kwargs: Optional[Dict[str, Any]] = None
    
    architecture_model: str = "llama-3.3-70b-versatile"
    architecture_model_kwargs: Optional[Dict[str, Any]] = None
    
    sdlc_model: str = "llama-3.3-70b-versatile"
    sdlc_model_kwargs: Optional[Dict[str, Any]] = None
    
    infrastructure_model: str = "llama-3.3-70b-versatile"
    infrastructure_model_kwargs: Optional[Dict[str, Any]] = None
    
    security_model: str = "llama-3.3-70b-versatile"
    security_model_kwargs: Optional[Dict[str, Any]] = None
    
    ip_model: str = "llama-3.3-70b-versatile"
    ip_model_kwargs: Optional[Dict[str, Any]] = None
    
    teams_model: str = "llama-3.3-70b-versatile"
    teams_model_kwargs: Optional[Dict[str, Any]] = None
    
    # Writer agent configuration
    writer_model: str = "llama-3.3-70b-versatile"
    writer_model_kwargs: Optional[Dict[str, Any]] = None
    
    # Domains to include in the TDD process
    domains: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        # Set default domains if not provided
        if self.domains is None:
            self.domains = [
                "tech_stack", "architecture", "sdlc", "infrastructure", 
                "security", "ip", "teams"
            ]
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get the configuration for a specific agent type.
        
        Args:
            agent_type: The type of agent (e.g., "planning", "tech_stack")
            
        Returns:
            A dictionary with the agent configuration
        """
        model_attr = f"{agent_type}_model"
        model_kwargs_attr = f"{agent_type}_model_kwargs"
        
        model = getattr(self, model_attr, self.planning_model)
        model_kwargs = getattr(self, model_kwargs_attr, {}) or {}
        
        return {
            "model": model,
            "model_kwargs": model_kwargs
        }
