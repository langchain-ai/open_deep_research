"""
Virtual Data Room (VDR) for the Technical Due Diligence (TDD) Agent System.

This module provides a repository for all TDD reports and findings,
allowing for storage, retrieval, and analysis of due diligence data.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import datetime

from open_deep_research.tdd.state.models import (
    DomainReport, Finding, Evidence, Interdependency, Gap
)

logger = logging.getLogger(__name__)

class VirtualDataRoom:
    """Repository for all TDD reports and findings.
    
    The VDR stores all reports, findings, evidence, and other artifacts
    generated during the technical due diligence process. It provides
    methods for storing, retrieving, and analyzing this data.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize the Virtual Data Room.
        
        Args:
            storage_dir: Directory to store VDR data. If None, a default
                         directory will be created in the user's home directory.
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".open_deep_research" / "vdr"
        
        self.storage_dir = storage_dir
        self.domain_reports: Dict[str, DomainReport] = {}
        self.findings: Dict[str, Finding] = {}
        self.evidence: Dict[str, Evidence] = {}
        self.interdependencies: Dict[str, Interdependency] = {}
        self.gaps: Dict[str, Gap] = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        logger.info(f"Initialized Virtual Data Room in {self.storage_dir}")
    
    def add_domain_report(self, report: DomainReport) -> None:
        """Add a domain report to the VDR.
        
        Args:
            report: The domain report to add
        """
        self.domain_reports[report.id] = report
        
        # Add findings and evidence from the report
        for finding in report.findings:
            self.findings[finding.id] = finding
        
        for evidence_item in report.evidence:
            self.evidence[evidence_item.id] = evidence_item
        
        logger.info(f"Added domain report {report.id} for domain {report.domain}")
    
    def add_finding(self, finding: Finding) -> None:
        """Add a finding to the VDR.
        
        Args:
            finding: The finding to add
        """
        self.findings[finding.id] = finding
        logger.info(f"Added finding {finding.id}: {finding.title}")
    
    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence to the VDR.
        
        Args:
            evidence: The evidence to add
        """
        self.evidence[evidence.id] = evidence
        logger.info(f"Added evidence {evidence.id}: {evidence.title}")
    
    def add_interdependency(self, interdependency: Interdependency) -> None:
        """Add an interdependency to the VDR.
        
        Args:
            interdependency: The interdependency to add
        """
        self.interdependencies[interdependency.id] = interdependency
        logger.info(f"Added interdependency {interdependency.id}: {interdependency.title}")
    
    def add_gap(self, gap: Gap) -> None:
        """Add an information gap to the VDR.
        
        Args:
            gap: The information gap to add
        """
        self.gaps[gap.id] = gap
        logger.info(f"Added information gap {gap.id}: {gap.title}")
    
    def get_domain_report(self, report_id: str) -> Optional[DomainReport]:
        """Get a domain report from the VDR.
        
        Args:
            report_id: ID of the report to retrieve
            
        Returns:
            The domain report, or None if not found
        """
        return self.domain_reports.get(report_id)
    
    def get_finding(self, finding_id: str) -> Optional[Finding]:
        """Get a finding from the VDR.
        
        Args:
            finding_id: ID of the finding to retrieve
            
        Returns:
            The finding, or None if not found
        """
        return self.findings.get(finding_id)
    
    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Get evidence from the VDR.
        
        Args:
            evidence_id: ID of the evidence to retrieve
            
        Returns:
            The evidence, or None if not found
        """
        return self.evidence.get(evidence_id)
    
    def get_interdependency(self, interdependency_id: str) -> Optional[Interdependency]:
        """Get an interdependency from the VDR.
        
        Args:
            interdependency_id: ID of the interdependency to retrieve
            
        Returns:
            The interdependency, or None if not found
        """
        return self.interdependencies.get(interdependency_id)
    
    def get_gap(self, gap_id: str) -> Optional[Gap]:
        """Get an information gap from the VDR.
        
        Args:
            gap_id: ID of the information gap to retrieve
            
        Returns:
            The information gap, or None if not found
        """
        return self.gaps.get(gap_id)
    
    def save(self) -> None:
        """Save the VDR to disk."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.storage_dir / timestamp
        os.makedirs(save_dir, exist_ok=True)
        
        # Save domain reports
        reports_dir = save_dir / "reports"
        os.makedirs(reports_dir, exist_ok=True)
        for report_id, report in self.domain_reports.items():
            with open(reports_dir / f"{report_id}.json", "w") as f:
                f.write(report.json())
        
        # Save findings
        findings_dir = save_dir / "findings"
        os.makedirs(findings_dir, exist_ok=True)
        for finding_id, finding in self.findings.items():
            with open(findings_dir / f"{finding_id}.json", "w") as f:
                f.write(finding.json())
        
        # Save evidence
        evidence_dir = save_dir / "evidence"
        os.makedirs(evidence_dir, exist_ok=True)
        for evidence_id, evidence_item in self.evidence.items():
            with open(evidence_dir / f"{evidence_id}.json", "w") as f:
                f.write(evidence_item.json())
        
        # Save interdependencies
        interdependencies_dir = save_dir / "interdependencies"
        os.makedirs(interdependencies_dir, exist_ok=True)
        for interdependency_id, interdependency in self.interdependencies.items():
            with open(interdependencies_dir / f"{interdependency_id}.json", "w") as f:
                f.write(interdependency.json())
        
        # Save gaps
        gaps_dir = save_dir / "gaps"
        os.makedirs(gaps_dir, exist_ok=True)
        for gap_id, gap in self.gaps.items():
            with open(gaps_dir / f"{gap_id}.json", "w") as f:
                f.write(gap.json())
        
        logger.info(f"Saved VDR to {save_dir}")
    
    def load(self, load_dir: Path) -> None:
        """Load the VDR from disk.
        
        Args:
            load_dir: Directory to load VDR data from
        """
        # Load domain reports
        reports_dir = load_dir / "reports"
        if reports_dir.exists():
            for report_file in reports_dir.glob("*.json"):
                with open(report_file, "r") as f:
                    report_data = json.load(f)
                    report = DomainReport(**report_data)
                    self.domain_reports[report.id] = report
        
        # Load findings
        findings_dir = load_dir / "findings"
        if findings_dir.exists():
            for finding_file in findings_dir.glob("*.json"):
                with open(finding_file, "r") as f:
                    finding_data = json.load(f)
                    finding = Finding(**finding_data)
                    self.findings[finding.id] = finding
        
        # Load evidence
        evidence_dir = load_dir / "evidence"
        if evidence_dir.exists():
            for evidence_file in evidence_dir.glob("*.json"):
                with open(evidence_file, "r") as f:
                    evidence_data = json.load(f)
                    evidence_item = Evidence(**evidence_data)
                    self.evidence[evidence_item.id] = evidence_item
        
        # Load interdependencies
        interdependencies_dir = load_dir / "interdependencies"
        if interdependencies_dir.exists():
            for interdependency_file in interdependencies_dir.glob("*.json"):
                with open(interdependency_file, "r") as f:
                    interdependency_data = json.load(f)
                    interdependency = Interdependency(**interdependency_data)
                    self.interdependencies[interdependency.id] = interdependency
        
        # Load gaps
        gaps_dir = load_dir / "gaps"
        if gaps_dir.exists():
            for gap_file in gaps_dir.glob("*.json"):
                with open(gap_file, "r") as f:
                    gap_data = json.load(f)
                    gap = Gap(**gap_data)
                    self.gaps[gap.id] = gap
        
        logger.info(f"Loaded VDR from {load_dir}")
    
    def get_all_domain_reports(self) -> List[DomainReport]:
        """Get all domain reports in the VDR.
        
        Returns:
            List of all domain reports
        """
        return list(self.domain_reports.values())
    
    def get_all_findings(self) -> List[Finding]:
        """Get all findings in the VDR.
        
        Returns:
            List of all findings
        """
        return list(self.findings.values())
    
    def get_all_evidence(self) -> List[Evidence]:
        """Get all evidence in the VDR.
        
        Returns:
            List of all evidence
        """
        return list(self.evidence.values())
    
    def get_all_interdependencies(self) -> List[Interdependency]:
        """Get all interdependencies in the VDR.
        
        Returns:
            List of all interdependencies
        """
        return list(self.interdependencies.values())
    
    def get_all_gaps(self) -> List[Gap]:
        """Get all information gaps in the VDR.
        
        Returns:
            List of all information gaps
        """
        return list(self.gaps.values())
