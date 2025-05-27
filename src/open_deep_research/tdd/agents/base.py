"""
Base Agent for the Technical Due Diligence (TDD) Agent System.

This module defines the base agent class that all TDD agents inherit from.
"""

import logging
from typing import Dict, List, Any, Optional

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from open_deep_research.logging_config import get_logger

logger = logging.getLogger(__name__)

class TDDAgent:
    """Base class for all TDD agents."""
    
    def __init__(self, model: str, model_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize the agent.
        
        Args:
            model: The model to use for the agent
            model_kwargs: Additional arguments for the model
        """
        self.model = model
        self.model_kwargs = model_kwargs or {}
        
        # Use ChatGroq for Groq models, otherwise use ChatOpenAI
        if "llama" in model.lower():
            self.llm = ChatGroq(model=model, **self.model_kwargs)
        else:
            self.llm = ChatOpenAI(model=model, **self.model_kwargs)
            
        self.logger = get_logger(self.__class__.__name__)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the agent.
        
        This method should be overridden by subclasses.
        
        Returns:
            The system prompt for the agent
        """
        raise NotImplementedError("Subclasses must implement get_system_prompt")
    
    def create_chain(self, tools: List[Any]) -> Any:
        """Create the agent chain.
        
        Args:
            tools: The tools available to the agent
            
        Returns:
            The agent chain
        """
        system_prompt = self.get_system_prompt()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain
