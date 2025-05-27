"""
Command-line interface for Open Deep Research and TDD Agent System.

This module provides a unified command-line interface for running both
Open Deep Research and the TDD Agent System.
"""

import logging
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from open_deep_research.logging_config import setup_logging
from open_deep_research.configuration import Configuration
from open_deep_research.multi_agent import run_research
from open_deep_research.tdd.configuration import TDDConfiguration
from open_deep_research.tdd.agents import run_tdd
from open_deep_research.tdd.integration import TDDResearchAdapter

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Open Deep Research or TDD Agent System")
    
    # Common arguments
    parser.add_argument("--query", type=str, required=True,
                        help="The query to research or run TDD for")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--output", type=str, default="research_report.json",
                        help="Path to output file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="research",
                        choices=["research", "tdd", "integrated"],
                        help="Mode to run in (research, tdd, or integrated)")
    
    # TDD-specific arguments
    parser.add_argument("--domains", type=str, nargs="+",
                        help="Domains to include in the TDD assessment")
    parser.add_argument("--deal-type", type=str, default="acquisition",
                        choices=["acquisition", "merger", "investment", "partnership"],
                        help="Type of deal for TDD")
    parser.add_argument("--assessment-depth", type=str, default="standard",
                        choices=["light", "standard", "deep"],
                        help="Depth of TDD assessment")
    parser.add_argument("--risk-framework", type=str, default="OWASP",
                        help="Risk assessment framework for TDD")
    
    return parser.parse_args()

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        return {}
    
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Configuration file {config_path} does not exist")
        return {}
    
    with open(config_path, "r") as f:
        return json.load(f)

async def run_research_mode(args, config_dict):
    """Run in research mode.
    
    Args:
        args: Command line arguments
        config_dict: Configuration dictionary
        
    Returns:
        Research results
    """
    # Create configuration
    config = Configuration(**config_dict)
    
    # Run research
    logger.info(f"Running research for query: {args.query}")
    result = await run_research(args.query, config)
    
    return result

async def run_tdd_mode(args, config_dict):
    """Run in TDD mode.
    
    Args:
        args: Command line arguments
        config_dict: Configuration dictionary
        
    Returns:
        TDD results
    """
    # Override configuration with TDD-specific arguments
    if args.domains:
        config_dict["domains"] = args.domains
    
    config_dict["deal_type"] = args.deal_type
    config_dict["assessment_depth"] = args.assessment_depth
    config_dict["risk_assessment_framework"] = args.risk_framework
    
    # Create configuration
    config = TDDConfiguration(**config_dict)
    
    # Run TDD
    logger.info(f"Running TDD for query: {args.query}")
    result = await run_tdd(args.query, config)
    
    return result

async def run_integrated_mode(args, config_dict):
    """Run in integrated mode.
    
    Args:
        args: Command line arguments
        config_dict: Configuration dictionary
        
    Returns:
        Integrated results
    """
    # Override configuration with TDD-specific arguments
    if args.domains:
        config_dict["domains"] = args.domains
    
    config_dict["deal_type"] = args.deal_type
    config_dict["assessment_depth"] = args.assessment_depth
    config_dict["risk_assessment_framework"] = args.risk_framework
    
    # Create configuration
    config = TDDConfiguration(**config_dict)
    
    # Create adapter
    adapter = TDDResearchAdapter(config)
    
    # Run integrated research
    logger.info(f"Running integrated research for query: {args.query}")
    result = await adapter.run_tdd_research(args.query)
    
    return result

async def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load configuration
    config_dict = load_config(args.config)
    
    # Run in the selected mode
    if args.mode == "research":
        result = await run_research_mode(args, config_dict)
    elif args.mode == "tdd":
        result = await run_tdd_mode(args, config_dict)
    elif args.mode == "integrated":
        result = await run_integrated_mode(args, config_dict)
    else:
        logger.error(f"Invalid mode: {args.mode}")
        return
    
    # Save the result
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Process completed, result saved to {output_path}")
    
    # Print a summary
    print("\n=== Process Summary ===")
    print(f"Mode: {args.mode}")
    print(f"Query: {args.query}")
    print(f"Output: {output_path}")
    
    if args.mode == "tdd" or args.mode == "integrated":
        print(f"Domains: {args.domains or 'All'}")
        print(f"Deal Type: {args.deal_type}")
        print(f"Assessment Depth: {args.assessment_depth}")
        print(f"Risk Framework: {args.risk_framework}")
    
    print("=======================\n")

if __name__ == "__main__":
    asyncio.run(main())
