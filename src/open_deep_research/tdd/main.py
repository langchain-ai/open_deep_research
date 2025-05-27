"""
Main module for the Technical Due Diligence (TDD) Agent System.

This module provides the main entry point for running the TDD process.
"""

import logging
import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

from open_deep_research.logging_config import setup_logging
from open_deep_research.tdd.configuration import TDDConfiguration
from open_deep_research.tdd.run import run_tdd
from open_deep_research.tdd.vdr import VirtualDataRoom

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the TDD Agent System")
    parser.add_argument("--query", type=str, required=True,
                        help="The query to run TDD for")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--output", type=str, default="tdd_report.json",
                        help="Path to output file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--domains", type=str, nargs="+",
                        help="Domains to include in the assessment")
    parser.add_argument("--deal-type", type=str, default="acquisition",
                        choices=["acquisition", "merger", "investment", "partnership"],
                        help="Type of deal")
    parser.add_argument("--assessment-depth", type=str, default="standard",
                        choices=["light", "standard", "deep"],
                        help="Depth of assessment")
    parser.add_argument("--risk-framework", type=str, default="OWASP",
                        help="Risk assessment framework")
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

async def main():
    """Main entry point for the TDD Agent System."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load configuration
    config_dict = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.domains:
        config_dict["domains"] = args.domains
    
    config_dict["deal_type"] = args.deal_type
    config_dict["assessment_depth"] = args.assessment_depth
    config_dict["risk_assessment_framework"] = args.risk_framework
    
    # Create configuration
    config = TDDConfiguration(**config_dict)
    
    # Run the TDD process
    logger.info(f"Running TDD for query: {args.query}")
    result = await run_tdd(args.query, config)
    
    # Save the result
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"TDD process completed, result saved to {output_path}")
    
    # Print a summary
    print("\n=== TDD Process Summary ===")
    print(f"Query: {args.query}")
    print(f"Domains: {config.domains}")
    print(f"Deal Type: {config.deal_type}")
    print(f"Assessment Depth: {config.assessment_depth}")
    print(f"Risk Framework: {config.risk_assessment_framework}")
    print(f"Output: {output_path}")
    print("===========================\n")
    
    # Print the executive summary
    if "final_report" in result:
        summary_lines = result["final_report"].split("\n")
        for i, line in enumerate(summary_lines):
            if "Executive Summary" in line or "EXECUTIVE SUMMARY" in line:
                start_index = i
                break
        else:
            start_index = 0
        
        for i, line in enumerate(summary_lines[start_index:]):
            if "Table of Contents" in line or "TABLE OF CONTENTS" in line:
                end_index = start_index + i
                break
        else:
            end_index = len(summary_lines)
        
        print("\n=== Executive Summary ===")
        print("\n".join(summary_lines[start_index:end_index]))
        print("========================\n")

if __name__ == "__main__":
    asyncio.run(main())
