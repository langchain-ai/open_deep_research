"""Tests for the extensions API router endpoints."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient


def get_test_client():
    """Create a test client for the extensions router only."""
    from fastapi import FastAPI
    from routers.extensions_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestAnalyzeEndpoint:
    """Test POST /api/extensions/analyze."""

    @patch("routers.extensions_api.DataAnalysisAgent", autospec=True)
    def test_analyze_returns_200(self, mock_agent_cls):
        """Endpoint should return 200 with valid pipeline result."""
        mock_instance = MagicMock()
        mock_instance.run_pipeline.return_value = {
            "status": "completed",
            "extracted_data": "Year,Value\n2020,10",
            "data_profile": "2 rows, 2 columns",
            "charts": [],
            "chart_explanations": {},
            "outlier_analysis": None,
            "output": "Pipeline complete",
            "execution_time": 1.5,
            "error": None,
        }
        mock_agent_cls.return_value = mock_instance

        # Patch the import inside the endpoint
        with patch("routers.extensions_api.DataAnalysisAgent", mock_agent_cls):
            client = get_test_client()
            response = client.post(
                "/api/extensions/analyze",
                json={"data": "Year,Value\n2020,10\n2021,20"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    def test_analyze_missing_data_returns_422(self):
        """Endpoint should return 422 when data field is missing."""
        client = get_test_client()
        response = client.post("/api/extensions/analyze", json={})
        assert response.status_code == 422


class TestChartServing:
    """Test GET /api/extensions/charts/{filename}."""

    def test_chart_not_found_returns_404(self):
        client = get_test_client()
        response = client.get("/api/extensions/charts/nonexistent_abc12345.html")
        assert response.status_code == 404

    def test_chart_serving_with_existing_file(self, tmp_path):
        """Test that chart serving works with an existing file."""
        # Create temp chart file
        chart_dir = tmp_path / "outputs" / "charts"
        chart_dir.mkdir(parents=True)
        chart_file = chart_dir / "bar_test1234.html"
        chart_file.write_text("<html><body>chart</body></html>")

        with patch("routers.extensions_api.CHARTS_DIR", chart_dir):
            client = get_test_client()
            response = client.get("/api/extensions/charts/bar_test1234.html")
            assert response.status_code == 200


class TestReportServing:
    """Test GET /api/extensions/reports/{filename}."""

    def test_report_not_found_returns_404(self):
        client = get_test_client()
        response = client.get("/api/extensions/reports/nonexistent.html")
        assert response.status_code == 404

    def test_report_serving_with_existing_file(self, tmp_path):
        report_dir = tmp_path / "outputs" / "reports"
        report_dir.mkdir(parents=True)
        report_file = report_dir / "report_test1234.html"
        report_file.write_text("<html><body>report</body></html>")

        with patch("routers.extensions_api.REPORTS_DIR", report_dir):
            client = get_test_client()
            response = client.get("/api/extensions/reports/report_test1234.html")
            assert response.status_code == 200


class TestDownloadEndpoints:
    """Test download endpoints."""

    def test_download_research_not_found(self):
        client = get_test_client()
        response = client.get("/api/extensions/download/abcd1234/research")
        assert response.status_code == 404

    def test_download_analysis_not_found(self):
        client = get_test_client()
        response = client.get("/api/extensions/download/abcd1234/analysis")
        assert response.status_code == 404

    def test_download_research_with_existing_file(self, tmp_path):
        report_dir = tmp_path / "outputs" / "reports"
        report_dir.mkdir(parents=True)
        md_file = report_dir / "report_test1234_research.md"
        md_file.write_text("# Research Report\n\nFindings here.")

        with patch("routers.extensions_api.REPORTS_DIR", report_dir):
            client = get_test_client()
            response = client.get("/api/extensions/download/test1234/research")
            assert response.status_code == 200

    def test_download_analysis_with_existing_file(self, tmp_path):
        report_dir = tmp_path / "outputs" / "reports"
        report_dir.mkdir(parents=True)
        html_file = report_dir / "report_test1234.html"
        html_file.write_text("<html><body>Analysis</body></html>")

        with patch("routers.extensions_api.REPORTS_DIR", report_dir):
            client = get_test_client()
            response = client.get("/api/extensions/download/test1234/analysis")
            assert response.status_code == 200


class TestResearchAnalyzeEndpoint:
    """Test POST /api/extensions/research-analyze (integration-level mocks)."""

    def test_research_analyze_missing_query_returns_422(self):
        client = get_test_client()
        response = client.post("/api/extensions/research-analyze", json={})
        assert response.status_code == 422

    def test_research_analyze_accepts_valid_request(self):
        """Just validate the request model accepts the query field."""
        from routers.extensions_api import ResearchAnalyzeRequest
        req = ResearchAnalyzeRequest(query="test query")
        assert req.query == "test query"

    def test_analyze_response_model(self):
        """Validate response model fields."""
        from routers.extensions_api import AnalyzeResponse
        resp = AnalyzeResponse(
            status="completed",
            extracted_data="csv",
            data_profile="profile",
            charts=["chart1.html"],
            chart_explanations={"chart1.html": {"title": "Test", "explanation": "Explanation"}},
            output="Done",
            execution_time=5.0,
        )
        assert resp.status == "completed"
        assert len(resp.charts) == 1

    def test_research_analyze_response_model(self):
        from routers.extensions_api import ResearchAnalyzeResponse
        resp = ResearchAnalyzeResponse(
            status="completed",
            research_report="# Report",
            analysis_report_path="outputs/reports/report_abc.html",
            charts=["chart1.html"],
            sources=["https://example.com"],
            sub_queries=["subtopic 1"],
            download_urls={"research": "/api/extensions/download/abc/research"},
            execution_time=10.0,
        )
        assert resp.status == "completed"
        assert len(resp.sub_queries) == 1
