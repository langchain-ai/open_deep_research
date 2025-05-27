"""
Node functions for the Technical Due Diligence (TDD) Agent System.

This module defines the node functions used in the TDD graph, including
initialization, running agents, and processing results.
"""

import logging
from typing import Dict, List, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from open_deep_research.tdd.state.graph_state import TDDGraphState
from open_deep_research.tdd.agents import (
    PlanningAgent, TechStackAgent, ArchitectureAgent, SDLCAgent,
    InfrastructureAgent, SecurityAgent, IPAgent, TeamsAgent, WriterAgent
)

logger = logging.getLogger(__name__)

async def initialize(state: TDDGraphState) -> TDDGraphState:
    """
    Initialize the TDD process.
    
    This function sets up the initial state for the TDD process,
    including the domains to investigate and the initial messages.
    
    Args:
        state: The initial state
        
    Returns:
        The initialized state
    """
    logger.info("Initializing TDD process")
    
    # Get the query and configuration
    query = state.get("query", "")
    config = state.get("config", {})
    
    # Set up the domains to investigate
    domains = config.get("domains", [
        "tech_stack", "architecture", "sdlc", "infrastructure", 
        "security", "ip", "teams"
    ])
    
    # Initialize the state
    state["domains"] = domains
    state["current_domain"] = "planning"  # Start with planning
    state["domain_results"] = {}
    state["findings"] = []
    state["evidence"] = []
    state["interdependencies"] = []
    state["gaps"] = []
    state["plan"] = {}
    state["replanning_needed"] = False
    state["replanning_consideration"] = False
    
    # Add the initial message
    state["messages"] = [
        HumanMessage(content=f"Conduct a technical due diligence on: {query}")
    ]
    
    logger.info(f"TDD process initialized with domains: {domains}")
    return state

async def run_tdd_planner(state: TDDGraphState) -> TDDGraphState:
    """
    Run the TDD Planner agent.
    
    This function runs the TDD Planner agent to create or update the
    plan for the TDD process.
    
    Args:
        state: The current state of the TDD investigation
        
    Returns:
        The updated state with the TDD plan
    """
    logger.info("Running TDD Planner")
    
    # Get the query and configuration
    query = state.get("query", "")
    config = state.get("config", {})
    domains = state.get("domains", [])
    
    # Create the planning agent
    model = config.get("planning_model", config.get("model", "gpt-4"))
    model_kwargs = config.get("planning_model_kwargs", config.get("model_kwargs", {}))
    planning_agent = PlanningAgent(model=model, model_kwargs=model_kwargs)
    
    # Prepare the messages for the planning agent
    messages = state.get("messages", [])
    
    # Truncate message history if it's getting too large to prevent request size errors
    # Keep only the most recent messages and the initial query
    if len(messages) > 10:  # Arbitrary threshold, adjust as needed
        logger.info(f"Truncating message history from {len(messages)} to 5 messages to prevent request size errors")
        # Always keep the first message (initial query) and the last 4 messages
        messages = [messages[0]] + messages[-4:]
    
    # Add domain results if available for replanning
    domain_results = state.get("domain_results", {})
    if domain_results:
        # Add a summary of the current findings, gaps, and interdependencies
        findings_count = len(state.get("findings", []))
        gaps_count = len(state.get("gaps", []))
        interdependencies_count = len(state.get("interdependencies", []))
        
        summary = f"Current state of the TDD investigation:\n"
        summary += f"- {findings_count} findings\n"
        summary += f"- {gaps_count} information gaps\n"
        summary += f"- {interdependencies_count} interdependencies\n\n"
        
        # Add details about each domain
        summary += "Domain results:\n"
        for domain, results in domain_results.items():
            domain_findings = len(results.get("findings", []))
            domain_gaps = len(results.get("gaps", []))
            domain_interdependencies = len(results.get("interdependencies", []))
            
            summary += f"- {domain}: {domain_findings} findings, {domain_gaps} gaps, "
            summary += f"{domain_interdependencies} interdependencies\n"
        
        messages.append(SystemMessage(content=summary))
    
    # Run the planning agent
    chain = planning_agent.create_chain([])
    result = await chain.ainvoke({"messages": messages})
    
    # Parse the result
    # In a real implementation, we would parse the result to extract the plan
    # For now, we'll just use the result as the plan
    plan = {
        "domains": domains,
        "description": result
    }
    
    # Update the state
    state["plan"] = plan
    state["current_domain"] = domains[0] if domains else "none"
    state["replanning_needed"] = False
    
    # Add the planning result to the messages
    state["messages"] = messages + [AIMessage(content=result)]
    
    logger.info(f"TDD Planner completed with plan for domains: {domains}")
    return state

async def run_domain_agent(state: TDDGraphState) -> TDDGraphState:
    """
    Run a domain-specific agent.
    
    This function runs the appropriate domain-specific agent for the
    current domain being investigated.
    
    Args:
        state: The current state of the TDD investigation
        
    Returns:
        The updated state with the domain agent's findings
    """
    # Get the current domain and configuration
    current_domain = state.get("current_domain", "")
    config = state.get("config", {})
    
    logger.info(f"Running domain agent for {current_domain}")
    
    # Skip if no current domain
    if not current_domain or current_domain == "none":
        logger.warning("No current domain specified, skipping domain agent")
        return state
    
    # Get the appropriate agent for the domain
    # Use domain-specific model if available, otherwise fall back to default model
    domain_model_key = f"{current_domain}_model"
    domain_model_kwargs_key = f"{current_domain}_model_kwargs"
    
    model = config.get(domain_model_key, config.get("model", "gpt-4"))
    model_kwargs = config.get(domain_model_kwargs_key, config.get("model_kwargs", {}))
    
    domain_agent = None
    if current_domain == "tech_stack":
        domain_agent = TechStackAgent(model=model, model_kwargs=model_kwargs)
    elif current_domain == "architecture":
        domain_agent = ArchitectureAgent(model=model, model_kwargs=model_kwargs)
    elif current_domain == "sdlc":
        domain_agent = SDLCAgent(model=model, model_kwargs=model_kwargs)
    elif current_domain == "infrastructure":
        domain_agent = InfrastructureAgent(model=model, model_kwargs=model_kwargs)
    elif current_domain == "security":
        domain_agent = SecurityAgent(model=model, model_kwargs=model_kwargs)
    elif current_domain == "ip":
        domain_agent = IPAgent(model=model, model_kwargs=model_kwargs)
    elif current_domain == "teams":
        domain_agent = TeamsAgent(model=model, model_kwargs=model_kwargs)
    else:
        logger.warning(f"Unknown domain: {current_domain}, skipping domain agent")
        return state
    
    # Get the messages and plan
    messages = state.get("messages", [])
    plan = state.get("plan", {})
    
    # Truncate message history if it's getting too large to prevent request size errors
    # Keep only the most recent messages and the initial query
    if len(messages) > 10:  # Arbitrary threshold, adjust as needed
        logger.info(f"Truncating message history from {len(messages)} to 5 messages to prevent request size errors")
        # Always keep the first message (initial query) and the last 4 messages
        messages = [messages[0]] + messages[-4:]
    
    # Add a message with the domain-specific instructions
    domain_message = f"Investigate the {current_domain} domain according to the TDD plan."
    messages.append(HumanMessage(content=domain_message))
    
    try:
        # Run the domain agent
        chain = domain_agent.create_chain([])
        result = await chain.ainvoke({"messages": messages})
        
        # Parse the result
        # In a real implementation, we would parse the result to extract findings, etc.
        # For now, we'll just use a simple structure
        domain_result = {
            "domain": current_domain,
            "findings": [],  # Would be extracted from the result
            "gaps": [],  # Would be extracted from the result
            "interdependencies": [],  # Would be extracted from the result
            "summary": result
        }
        
        # Update the state
        domain_results = state.get("domain_results", {})
        domain_results[current_domain] = domain_result
        state["domain_results"] = domain_results
        
        # Add the domain agent result to the messages
        state["messages"] = messages + [AIMessage(content=result)]
        
        # The result is a string, not a dictionary, so we can't use .get() on it
        # Instead, we'll use the domain_result we created
        if domain_result.get("gaps") or domain_result.get("interdependencies"):
            # Set a flag to consider replanning during reflection
            state["replanning_consideration"] = True
            
        logger.info(f"Specialized domain agent for {current_domain} completed successfully")
        logger.info(f"Found {len(domain_result.get('findings', []))} findings, {len(domain_result.get('gaps', []))} gaps, "
                  f"and {len(domain_result.get('interdependencies', []))} interdependencies")
        
    except Exception as e:
        logger.error(f"Error running domain agent for {current_domain}: {e}")
        # Set replanning needed flag to true on error
        state["replanning_needed"] = True
    
    return state

async def run_reflection(state: TDDGraphState) -> TDDGraphState:
    """
    Run the reflection process to evaluate progress and determine next steps.
    
    This function analyzes the current state of the TDD investigation,
    including completed domains, findings, gaps, and interdependencies,
    to determine whether to continue with another domain, return to planning,
    or move to the final report writing.
    
    Args:
        state: The current state of the TDD investigation
        
    Returns:
        The updated state with reflection insights
    """
    logger.info("Running reflection to evaluate TDD progress")
    
    # Get the current domain and all domains
    current_domain = state.get("current_domain")
    domains = state.get("domains", [])
    domain_results = state.get("domain_results", {})
    
    # Prepare a summary of the current state
    completed_domains = list(domain_results.keys())
    remaining_domains = [d for d in domains if d not in completed_domains]
    
    # Count findings, gaps, and interdependencies
    total_findings = sum(len(results.get("findings", [])) for results in domain_results.values())
    total_gaps = sum(len(results.get("gaps", [])) for results in domain_results.values())
    total_interdependencies = sum(len(results.get("interdependencies", [])) for results in domain_results.values())
    
    logger.info(f"Reflection summary: {len(completed_domains)}/{len(domains)} domains completed")
    logger.info(f"Total findings: {total_findings}, gaps: {total_gaps}, interdependencies: {total_interdependencies}")
    
    # Check if replanning is needed based on gaps and interdependencies
    if state.get("replanning_consideration", False) and (total_gaps > 0 or total_interdependencies > 0):
        # Determine if replanning is needed based on the severity of gaps and interdependencies
        # For now, we'll use a simple threshold
        if total_gaps > 3 or total_interdependencies > 5:
            logger.info("Reflection indicates significant gaps or interdependencies. Replanning recommended.")
            state["replanning_needed"] = True
        else:
            logger.info("Gaps and interdependencies present but not severe enough for replanning.")
            state["replanning_needed"] = False
    
    # If we've completed all domains and there are no significant issues, prepare for the writer
    if not remaining_domains and not state.get("replanning_needed", False):
        logger.info("All domains completed successfully. Ready for final report writing.")
        
        # Prepare a summary for the writer
        state["reflection_summary"] = {
            "completed_domains": completed_domains,
            "remaining_domains": [],
            "total_findings": total_findings,
            "total_gaps": total_gaps,
            "total_interdependencies": total_interdependencies,
            "recommendation": "Proceed to final report writing"
        }
    elif state.get("replanning_needed", False):
        logger.info("Reflection recommends returning to the TDD planner for replanning.")
        
        # Prepare a summary for the planner
        state["reflection_summary"] = {
            "completed_domains": completed_domains,
            "remaining_domains": remaining_domains,
            "total_findings": total_findings,
            "total_gaps": total_gaps,
            "total_interdependencies": total_interdependencies,
            "recommendation": "Return to planner for replanning"
        }
    else:
        # Continue with the next domain
        if remaining_domains:
            # If current_domain is in domains, find the next one
            if current_domain in domains:
                current_index = domains.index(current_domain)
                # Find the next domain that hasn't been processed yet
                for i in range(current_index + 1, len(domains)):
                    if domains[i] in remaining_domains:
                        next_domain = domains[i]
                        state["current_domain"] = next_domain
                        logger.info(f"Reflection recommends continuing with next domain: {next_domain}")
                        break
                else:
                    # If we didn't find a next domain, use the first remaining domain
                    next_domain = remaining_domains[0]
                    state["current_domain"] = next_domain
                    logger.info(f"Moving to remaining domain: {next_domain}")
            else:
                # If current_domain is not in domains, start with the first remaining domain
                next_domain = remaining_domains[0]
                state["current_domain"] = next_domain
                logger.info(f"Starting with domain: {next_domain}")
        else:
            logger.info("No more domains to process. Ready for final report writing.")
        
        # Prepare a summary for the next domain or writer
        state["reflection_summary"] = {
            "completed_domains": completed_domains,
            "remaining_domains": remaining_domains,
            "total_findings": total_findings,
            "total_gaps": total_gaps,
            "total_interdependencies": total_interdependencies,
            "recommendation": "Continue with next domain" if remaining_domains else "Proceed to final report writing"
        }
    
    return state

async def run_writer(state: TDDGraphState) -> TDDGraphState:
    """
    Run the writer agent.
    
    This function runs the Writer Agent to synthesize the findings from
    all domains into a cohesive final report.
    
    Args:
        state: The current state of the TDD investigation
        
    Returns:
        The updated state with the final report
    """
    logger.info("Running Writer Agent")
    
    # Get the configuration
    config = state.get("config", {})
    domain_results = state.get("domain_results", {})
    
    # Create the writer agent
    model = config.get("writer_model", config.get("model", "gpt-4"))
    model_kwargs = config.get("writer_model_kwargs", config.get("model_kwargs", {}))
    writer_agent = WriterAgent(model=model, model_kwargs=model_kwargs)
    
    # Prepare the messages for the writer agent
    messages = state.get("messages", [])
    
    # Truncate message history if it's getting too large to prevent request size errors
    # Keep only the most recent messages and the initial query
    if len(messages) > 10:  # Arbitrary threshold, adjust as needed
        logger.info(f"Truncating message history from {len(messages)} to 5 messages to prevent request size errors")
        # Always keep the first message (initial query) and the last 4 messages
        messages = [messages[0]] + messages[-4:]
    
    # Add a summary of the findings from all domains
    summary = "Synthesize the findings from all domains into a cohesive final report.\n\n"
    summary += "Domain results:\n"
    
    for domain, results in domain_results.items():
        summary += f"\n## {domain.upper()}\n"
        summary += f"Summary: {results.get('summary', 'No summary available')}\n"
        
        # Add findings
        findings = results.get("findings", [])
        if findings:
            summary += f"\nFindings ({len(findings)}):\n"
            for i, finding in enumerate(findings):
                summary += f"{i+1}. {finding.get('title', 'Untitled')}: "
                summary += f"{finding.get('description', 'No description')}\n"
        
        # Add gaps
        gaps = results.get("gaps", [])
        if gaps:
            summary += f"\nGaps ({len(gaps)}):\n"
            for i, gap in enumerate(gaps):
                summary += f"{i+1}. {gap.get('title', 'Untitled')}: "
                summary += f"{gap.get('description', 'No description')}\n"
        
        # Add interdependencies
        interdependencies = results.get("interdependencies", [])
        if interdependencies:
            summary += f"\nInterdependencies ({len(interdependencies)}):\n"
            for i, interdependency in enumerate(interdependencies):
                summary += f"{i+1}. {interdependency.get('title', 'Untitled')}: "
                summary += f"{interdependency.get('description', 'No description')}\n"
    
    messages.append(SystemMessage(content=summary))
    messages.append(HumanMessage(content="Write a comprehensive final report for the technical due diligence."))
    
    # Run the writer agent
    chain = writer_agent.create_chain([])
    result = await chain.ainvoke({"messages": messages})
    
    # Update the state
    state["final_report"] = result
    
    # Add the writer result to the messages
    state["messages"] = messages + [AIMessage(content=result)]
    
    logger.info("Writer Agent completed successfully")
    return state
