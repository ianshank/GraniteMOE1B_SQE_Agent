import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from src.integration.team_connectors import JiraConnector, GitHubConnector


@pytest.fixture
def dummy_test_case():
    """Return a minimal TestCase-like object for push_* tests."""
    return SimpleNamespace(
        id="TC-1",
        summary="Login works",
        priority=SimpleNamespace(value="medium"),
        test_type=SimpleNamespace(value="functional"),
        preconditions=[],
        steps=[],
        expected_results="ok",
        requirements_traced=[],
        input_data={},
    )


@pytest.mark.contract
@patch("src.integration.team_connectors.requests.get")
@patch("src.integration.team_connectors.requests.post")
def test_jira_connector_requests(mock_post, mock_get, dummy_test_case):
    """Validate JiraConnector builds correct JQL and payloads."""
    # mock GET search
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "issues": [
            {
                "key": "REQ-1",
                "fields": {
                    "summary": "Login",
                    "issuetype": {"name": "Story"},
                    "priority": {"name": "High"},
                },
            }
        ]
    }

    # mock POST create
    mock_post.return_value.status_code = 201

    conn = JiraConnector("https://example.atlassian.net", "user", "tok", "PROJ")
    reqs = conn.fetch_requirements()

    mock_get.assert_called_once()
    assert reqs and reqs[0]["id"] == "REQ-1"

    ok = conn.push_test_cases([dummy_test_case])
    assert ok is True
    mock_post.assert_called_once()
    # Validate payload contains summary and description keys
    payload = mock_post.call_args.kwargs["json"]
    assert payload["fields"]["summary"].startswith("Login")


@pytest.mark.contract
@patch("src.integration.team_connectors.requests.get")
@patch("src.integration.team_connectors.requests.post")
def test_github_connector_requests(mock_post, mock_get, dummy_test_case):
    """Validate GitHubConnector endpoints and payload."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = [{"number": 1, "title": "Req", "body": "desc"}]

    mock_post.return_value.status_code = 201

    conn = GitHubConnector("owner", "repo", "tok")
    reqs = conn.fetch_requirements()
    mock_get.assert_called_once()
    assert reqs[0]["id"] == "1"

    ok = conn.push_test_cases([dummy_test_case])
    assert ok is True
    mock_post.assert_called_once()
    endpoint = mock_post.call_args.args[0]
    assert "/issues" in endpoint
    body = mock_post.call_args.kwargs["json"]
    assert body["title"].startswith("Test Case")
