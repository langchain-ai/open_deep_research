"""Pydantic schemas for LLM-powered structured data extraction."""
from typing import List
from pydantic import BaseModel, Field


class ExtractedTable(BaseModel):
    """A single table of structured data extracted from research text."""
    table_name: str = Field(
        description="Descriptive name for this table (e.g., 'GDP Comparison by Country')"
    )
    headers: List[str] = Field(
        description="Column header names (e.g., ['Country', 'GDP_Trillion', 'Growth_Percent'])"
    )
    rows: List[List[str]] = Field(
        description="Data rows, each row is a list of string values aligned with headers"
    )


class ExtractedDataset(BaseModel):
    """All structured data extracted from a research report.

    May contain multiple tables covering different aspects of the report.
    """
    tables: List[ExtractedTable] = Field(
        default_factory=list,
        description="List of extracted tables, each with its own headers and rows"
    )
