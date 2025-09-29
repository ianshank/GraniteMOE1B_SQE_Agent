"""Unit tests for dotenv loading in train.py."""

import builtins
import importlib
import sys
from types import ModuleType
from typing import Any

import pytest


def _reload_train() -> ModuleType:
    """Reload train module to re-evaluate import-time code if needed."""
    if "train" in sys.modules:
        return importlib.reload(sys.modules["train"])  # type: ignore[arg-type]
    import train  # type: ignore

    return train


def test_load_dotenv_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing python-dotenv is handled gracefully."""
    train = _reload_train()

    # Simulate ImportError when calling load_dotenv
    def _raise_import_error(*_: Any, **__: Any) -> None:
        raise ImportError("python-dotenv not installed")

    monkeypatch.setattr(train, "load_dotenv", _raise_import_error, raising=True)

    # Ensure no exception is raised
    train.run_training(argv=["--epochs", "0"])  # minimal run


def test_load_dotenv_syntax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify syntax errors in .env are handled and only warn."""
    train = _reload_train()

    def _raise_value_error(*_: Any, **__: Any) -> None:
        raise ValueError("Invalid .env syntax")

    monkeypatch.setattr(train, "load_dotenv", _raise_value_error, raising=True)

    # Ensure no exception is raised
    train.run_training(argv=["--epochs", "0"])  # minimal run

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from types import SimpleNamespace
import time
from typing import List, Dict, Any

from src.integration.workflow_orchestrator import WorkflowOrchestrator, TeamConfiguration
from src.integration.team_connectors import TeamConnector
from src.models.test_case_schemas import TestCase, TestStep, TestCasePriority, TestCaseType


class MockConnector(TeamConnector):
    """Mock connector for testing"""
    
    def __init__(self, requirements: List[Dict[str, Any]], push_success: bool = True):
        self.requirements = requirements
        self.push_success = push_success
        self.push_called = False
        self.pushed_test_cases = None
    
    def fetch_requirements(self) -> List[Dict[str, Any]]:
        """Return mock requirements"""
        return self.requirements
    
    def push_test_cases(self, test_cases: List[TestCase]) -> bool:
        """Mock push test cases"""
        self.push_called = True
        self.pushed_test_cases = test_cases
        return self.push_success


class MockFailingConnector(TeamConnector):
    """Mock connector that always fails"""
    
    def fetch_requirements(self) -> List[Dict[str, Any]]:
        """Raise exception on fetch"""
        raise Exception("Failed to fetch requirements")
    
    def push_test_cases(self, test_cases: List[TestCase]) -> bool:
        """Raise exception on push"""
        raise Exception("Failed to push test cases")


def create_mock_test_case(id: str = "TC-001", summary: str = "Test Case 1") -> TestCase:
    """Helper to create a mock test case"""
    return TestCase(
        id=id,
        summary=summary,
        priority=TestCasePriority.MEDIUM,
        test_type=TestCaseType.FUNCTIONAL,
        steps=[TestStep(step_number=1, action="Action", expected_result="Result")],
        expected_results="Expected results",
        team_context="test_team"
    )


@pytest.fixture
def mock_agent():
    """Create a mock test generation agent"""
    agent = Mock()
    agent.generate_test_cases_for_team = AsyncMock(return_value=[
        create_mock_test_case("TC-001", "Test Case 1"),
        create_mock_test_case("TC-002", "Test Case 2")
    ])
    return agent


@pytest.fixture
def orchestrator(mock_agent):
    """Create a workflow orchestrator with mock agent"""
    return WorkflowOrchestrator(mock_agent)


class TestWorkflowOrchestrator:
    """Unit tests for WorkflowOrchestrator"""

    def test_create_team_configuration_forces_local_connector_when_flag_enabled(self, mock_agent, tmp_path):
        """Local-only mode should override remote connectors with LocalFileSystemConnector."""

        orchestrator = WorkflowOrchestrator(mock_agent, local_only=True)
        team_cfg = {
            "name": "ads_team",
            "connector": {
                "type": "github",
                "repo_owner": "o",
                "repo_name": "r",
                "token": "x",
                "input_directory": str(tmp_path),
            },
        }

        config = orchestrator.create_team_configuration(team_cfg)
        from src.integration.team_connectors import LocalFileSystemConnector

        assert isinstance(config.connector, LocalFileSystemConnector)
        assert config.connector.team_name == "ads_team"
        assert str(config.connector.input_directory) == str(tmp_path)

    def test_create_team_configuration_local_only_uses_default_input_when_missing(self, mock_agent):
        orchestrator = WorkflowOrchestrator(mock_agent, local_only=True)
        team_cfg = {
            "name": "cms_team",
            "connector": {
                "type": "jira",
                "base_url": "https://example.com",
                "username": "user",
                "api_token": "token",
                "project_key": "CMS",
            },
        }

        config = orchestrator.create_team_configuration(team_cfg)
        from src.integration.team_connectors import LocalFileSystemConnector

        assert isinstance(config.connector, LocalFileSystemConnector)
        assert config.connector.team_name == "cms_team"
        assert str(config.connector.input_directory) == "data/requirements/cms_team"

    def test_initialization(self, mock_agent):
        """Test orchestrator initialization"""
        orchestrator = WorkflowOrchestrator(mock_agent)
        assert orchestrator.agent == mock_agent
        assert orchestrator.team_configs == {}
        assert orchestrator.results_cache == {}
    
    def test_register_team(self, orchestrator):
        """Test team registration"""
        connector = MockConnector([{"id": "REQ-1", "summary": "Requirement 1"}])
        config = TeamConfiguration(
            team_name="team1",
            connector=connector,
            rag_enabled=True,
            cag_enabled=False,
            auto_push=True
        )
        
        orchestrator.register_team(config)
        
        assert "team1" in orchestrator.team_configs
        assert orchestrator.team_configs["team1"] == config
    
    def test_register_multiple_teams(self, orchestrator):
        """Test registering multiple teams"""
        connector1 = MockConnector([{"id": "REQ-1", "summary": "Req 1"}])
        connector2 = MockConnector([{"id": "REQ-2", "summary": "Req 2"}])
        
        config1 = TeamConfiguration("team1", connector1)
        config2 = TeamConfiguration("team2", connector2)
        
        orchestrator.register_team(config1)
        orchestrator.register_team(config2)
        
        assert len(orchestrator.team_configs) == 2
        assert "team1" in orchestrator.team_configs
        assert "team2" in orchestrator.team_configs
    
    @pytest.mark.asyncio
    async def test_process_all_teams_empty(self, orchestrator):
        """Test processing with no registered teams"""
        results = await orchestrator.process_all_teams()
        assert results == {}
    
    @pytest.mark.asyncio
    async def test_process_single_team_success(self, orchestrator, mock_agent):
        """Test successful processing of a single team"""
        requirements = [
            {"id": "REQ-1", "summary": "Login feature", "description": "User login"},
            {"id": "REQ-2", "summary": "Logout feature", "description": "User logout"}
        ]
        connector = MockConnector(requirements)
        config = TeamConfiguration("team1", connector, auto_push=False)
        
        orchestrator.register_team(config)
        results = await orchestrator.process_all_teams()
        
        assert "team1" in results
        assert len(results["team1"]) == 2  # Mock agent returns 2 test cases
        
        # Verify agent was called with correct parameters
        mock_agent.generate_test_cases_for_team.assert_called_once()
        call_args = mock_agent.generate_test_cases_for_team.call_args
        assert call_args[0][0] == "team1"  # team name
        assert len(call_args[0][1]) == 2  # requirements text list
        
        # Verify results are cached
        assert "team1" in orchestrator.results_cache
        assert orchestrator.results_cache["team1"] == results["team1"]
    
    @pytest.mark.asyncio
    async def test_process_team_with_auto_push(self, orchestrator):
        """Test processing with auto-push enabled"""
        requirements = [{"id": "REQ-1", "summary": "Feature 1"}]
        connector = MockConnector(requirements, push_success=True)
        config = TeamConfiguration("team1", connector, auto_push=True)
        
        orchestrator.register_team(config)
        results = await orchestrator.process_all_teams()
        
        # Verify push was called
        assert connector.push_called
        assert connector.pushed_test_cases == results["team1"]
    
    @pytest.mark.asyncio
    async def test_process_team_with_failed_push(self, orchestrator):
        """Test processing when push fails"""
        requirements = [{"id": "REQ-1", "summary": "Feature 1"}]
        connector = MockConnector(requirements, push_success=False)
        config = TeamConfiguration("team1", connector, auto_push=True)
        
        orchestrator.register_team(config)
        results = await orchestrator.process_all_teams()
        
        # Results should still be returned even if push fails
        assert "team1" in results
        assert len(results["team1"]) == 2
    
    @pytest.mark.asyncio
    async def test_process_team_with_exception(self, orchestrator):
        """Test processing when connector raises exception"""
        connector = MockFailingConnector()
        config = TeamConfiguration("failing_team", connector)
        
        orchestrator.register_team(config)
        results = await orchestrator.process_all_teams()
        
        # Should return empty list for failing team
        assert "failing_team" in results
        assert results["failing_team"] == []
    
    @pytest.mark.asyncio
    async def test_process_multiple_teams_parallel(self, orchestrator, mock_agent):
        """Test parallel processing of multiple teams"""
        # Create multiple teams
        teams = []
        for i in range(3):
            connector = MockConnector([{"id": f"REQ-{i}", "summary": f"Req {i}"}])
            config = TeamConfiguration(f"team{i}", connector)
            orchestrator.register_team(config)
            teams.append(f"team{i}")
        
        # Track call times to verify parallel execution
        call_times = []
        
        async def mock_generate(*args):
            call_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate processing time
            return [create_mock_test_case()]
        
        mock_agent.generate_test_cases_for_team = mock_generate
        
        start_time = time.time()
        results = await orchestrator.process_all_teams()
        total_time = time.time() - start_time
        
        # All teams should be processed
        for team in teams:
            assert team in results
        
        # Execution time should be less than sequential (0.1 * 3 = 0.3)
        # Adding buffer for test execution overhead
        assert total_time < 0.25  # Should complete in ~0.1s if parallel
    
    @pytest.mark.asyncio
    async def test_process_team_with_empty_requirements(self, orchestrator):
        """Test processing when no requirements are found"""
        connector = MockConnector([])  # Empty requirements
        config = TeamConfiguration("empty_team", connector)
        
        orchestrator.register_team(config)
        results = await orchestrator.process_all_teams()
        
        assert "empty_team" in results
        assert results["empty_team"] == []
    
    @pytest.mark.asyncio
    async def test_requirements_traceability(self, orchestrator):
        """Test that requirements are properly traced to test cases"""
        requirements = [
            {"id": "REQ-A", "summary": "Feature A"},
            {"id": "REQ-B", "summary": "Feature B"},
            {"id": "REQ-C", "summary": "Feature C"}
        ]
        connector = MockConnector(requirements)
        config = TeamConfiguration("trace_team", connector)
        
        orchestrator.register_team(config)
        results = await orchestrator.process_all_teams()
        
        test_cases = results["trace_team"]
        # Mock agent returns 2 test cases, so only first 2 reqs will be traced
        assert test_cases[0].requirements_traced == ["REQ-A"]
        assert test_cases[1].requirements_traced == ["REQ-B"]
    
    def test_generate_quality_report_empty(self, orchestrator):
        """Test quality report generation with no results"""
        report = orchestrator.generate_quality_report()
        
        assert report["total_test_cases"] == 0
        assert report["teams_processed"] == 0
        assert report["team_metrics"] == {}
        assert "generation_timestamp" in report
        assert report["report_status"] == "no_data"
    
    @pytest.mark.asyncio
    async def test_generate_quality_report_with_data(self, orchestrator):
        """Test quality report generation with actual data"""
        # Process some teams first
        connector = MockConnector([{"id": "REQ-1", "summary": "Req 1"}])
        config = TeamConfiguration("report_team", connector)
        orchestrator.register_team(config)
        
        await orchestrator.process_all_teams()
        
        # Generate report
        report = orchestrator.generate_quality_report()
        
        assert report["total_test_cases"] == 2  # Mock agent returns 2 test cases
        assert report["teams_processed"] == 1
        assert report["teams_with_results"] == 1
        assert "report_team" in report["team_metrics"]
        
        team_metrics = report["team_metrics"]["report_team"]
        assert team_metrics["test_case_count"] == 2
        assert "medium" in team_metrics["priority_distribution"]
        assert "functional" in team_metrics["type_distribution"]
        assert team_metrics["average_steps_per_test"] == 1.0  # Each mock test has 1 step
        assert team_metrics["total_steps"] == 2
        
        assert "generation_timestamp" in report
        assert report["report_status"] == "success"
    
    @pytest.mark.asyncio
    async def test_generate_quality_report_multiple_teams(self, orchestrator):
        """Test quality report with multiple teams and varied test cases"""
        # Create custom test cases with different priorities and types
        test_cases_team1 = [
            TestCase(
                id="TC-1",
                summary="Test 1",
                priority=TestCasePriority.HIGH,
                test_type=TestCaseType.FUNCTIONAL,
                steps=[TestStep(step_number=1, action="A", expected_result="R"), TestStep(step_number=2, action="B", expected_result="R")],
                expected_results="Results",
                team_context="team1"
            ),
            TestCase(
                id="TC-2",
                summary="Test 2",
                priority=TestCasePriority.MEDIUM,
                test_type=TestCaseType.INTEGRATION,
                steps=[TestStep(step_number=1, action="A", expected_result="R")],
                expected_results="Results",
                team_context="team1"
            )
        ]
        
        test_cases_team2 = [
            TestCase(
                id="TC-3",
                summary="Test 3",
                priority=TestCasePriority.LOW,
                test_type=TestCaseType.UNIT,
                steps=[TestStep(step_number=1, action="A", expected_result="R"), TestStep(step_number=2, action="B", expected_result="R"), TestStep(step_number=3, action="C", expected_result="R")],
                expected_results="Results",
                team_context="team2"
            )
        ]
        
        # Set up mock agent to return different test cases for different teams
        async def mock_generate(team_name, requirements):
            if team_name == "team1":
                return test_cases_team1
            else:
                return test_cases_team2
        
        orchestrator.agent.generate_test_cases_for_team = mock_generate
        
        # Register teams
        connector1 = MockConnector([{"id": "REQ-1", "summary": "Req 1"}])
        connector2 = MockConnector([{"id": "REQ-2", "summary": "Req 2"}])
        
        orchestrator.register_team(TeamConfiguration("team1", connector1))
        orchestrator.register_team(TeamConfiguration("team2", connector2))
        
        await orchestrator.process_all_teams()
        
        # Generate report
        report = orchestrator.generate_quality_report()
        
        assert report["total_test_cases"] == 3
        assert report["teams_processed"] == 2
        assert report["teams_with_results"] == 2
        
        # Check team1 metrics
        team1_metrics = report["team_metrics"]["team1"]
        assert team1_metrics["test_case_count"] == 2
        assert team1_metrics["priority_distribution"] == {"high": 1, "medium": 1}
        assert team1_metrics["type_distribution"] == {"functional": 1, "integration": 1}
        assert team1_metrics["average_steps_per_test"] == 1.5
        assert team1_metrics["total_steps"] == 3
        
        # Check team2 metrics
        team2_metrics = report["team_metrics"]["team2"]
        assert team2_metrics["test_case_count"] == 1
        assert team2_metrics["priority_distribution"] == {"low": 1}
        assert team2_metrics["type_distribution"] == {"unit": 1}
        assert team2_metrics["average_steps_per_test"] == 3.0
        assert team2_metrics["total_steps"] == 3
    
    @pytest.mark.asyncio
    async def test_error_handling_during_push(self, orchestrator):
        """Test error handling when push raises exception"""
        requirements = [{"id": "REQ-1", "summary": "Feature 1"}]
        
        # Create a connector that raises exception during push
        connector = Mock()
        connector.fetch_requirements = Mock(return_value=requirements)
        connector.push_test_cases = Mock(side_effect=Exception("Push failed"))
        
        config = TeamConfiguration("error_team", connector, auto_push=True)
        orchestrator.register_team(config)
        
        # Should not raise exception, but handle it gracefully
        results = await orchestrator.process_all_teams()
        
        # Results should still be available
        assert "error_team" in results
        assert len(results["error_team"]) == 2  # Mock agent returns 2 test cases
    
    def test_team_configuration_defaults(self):
        """Test TeamConfiguration default values"""
        connector = MockConnector([])
        config = TeamConfiguration("test_team", connector)
        
        assert config.team_name == "test_team"
        assert config.connector == connector
        assert config.rag_enabled is True
        assert config.cag_enabled is True
        assert config.auto_push is False
    
    @pytest.mark.asyncio
    async def test_logging_coverage(self, orchestrator, caplog):
        """Test that all major operations are logged"""
        import logging
        caplog.set_level(logging.DEBUG)
        
        # Register and process a team
        connector = MockConnector([{"id": "REQ-1", "summary": "Feature"}])
        config = TeamConfiguration("log_team", connector, auto_push=True)
        
        orchestrator.register_team(config)
        await orchestrator.process_all_teams()
        orchestrator.generate_quality_report()
        
        # Check for key log messages
        log_messages = [record.message for record in caplog.records]
        
        # Registration logs
        assert any("Team 'log_team' registered" in msg for msg in log_messages)
        
        # Processing logs
        assert any("Starting test case generation for 1 registered teams" in msg for msg in log_messages)
        assert any("Starting test case generation for team: log_team" in msg for msg in log_messages)
        assert any("Fetched 1 requirements for team: log_team" in msg for msg in log_messages)
        assert any("Generated 2 test cases for team: log_team" in msg for msg in log_messages)
        
        # Auto-push logs
        assert any("Auto-pushing 2 test cases for team: log_team" in msg for msg in log_messages)
        
        # Report generation logs
        assert any("Generating quality report" in msg for msg in log_messages)
        assert any("Quality report generated" in msg for msg in log_messages)
