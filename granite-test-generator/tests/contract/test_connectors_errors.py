from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import json
import requests


def _dummy_case():
    return SimpleNamespace(
        id="TC-1",
        summary="Case",
        priority=SimpleNamespace(value="medium"),
        test_type=SimpleNamespace(value="functional"),
        preconditions=[],
        steps=[],
        expected_results="",
        requirements_traced=[],
        input_data={},
    )


@patch("src.integration.team_connectors.requests.get")
def test_jira_fetch_json_decode_error(mock_get):
    """JiraConnector.fetch_requirements raises when JSON cannot be decoded."""
    from src.integration.team_connectors import JiraConnector

    resp = MagicMock(status_code=200)
    resp.json.side_effect = json.JSONDecodeError("x", "y", 0)
    mock_get.return_value = resp

    conn = JiraConnector("https://example", "u", "t", "PROJ")
    try:
        conn.fetch_requirements()
        assert False, "Expected exception for invalid JSON"
    except Exception as e:  # noqa: BLE001 - intentional broad for error path
        assert "Invalid JSON" in str(e)


@patch("src.integration.team_connectors.requests.get")
def test_github_fetch_json_decode_error(mock_get):
    """GitHubConnector.fetch_requirements raises on invalid JSON."""
    from src.integration.team_connectors import GitHubConnector

    resp = MagicMock(status_code=200)
    resp.json.side_effect = json.JSONDecodeError("x", "y", 0)
    mock_get.return_value = resp

    conn = GitHubConnector("o", "r", "tok")
    try:
        conn.fetch_requirements()
        assert False, "Expected exception for invalid JSON"
    except Exception as e:  # noqa: BLE001
        assert "Invalid JSON" in str(e)


@patch("src.integration.team_connectors.requests.post")
def test_github_push_partial_failure(mock_post):
    """Ensure partial failures cause a False return and continue pushing."""
    from src.integration.team_connectors import GitHubConnector

    # First call raises, second succeeds
    err = requests.exceptions.RequestException("fail")
    ok_resp = MagicMock(status_code=201)
    ok_resp.raise_for_status.return_value = None
    mock_post.side_effect = [err, ok_resp]

    conn = GitHubConnector("o", "r", "tok")
    ok = conn.push_test_cases([_dummy_case(), _dummy_case()])
    assert ok is False
    assert mock_post.call_count == 2


@patch("src.integration.team_connectors.requests.get")
def test_jira_fetch_request_exception(mock_get):
    """Ensure request exceptions are surfaced with helpful message."""
    from src.integration.team_connectors import JiraConnector

    mock_get.side_effect = requests.exceptions.RequestException("boom")
    conn = JiraConnector("https://example", "u", "t", "PROJ")
    try:
        conn.fetch_requirements()
        assert False, "Expected exception for request failure"
    except Exception as e:  # noqa: BLE001
        assert "Failed to fetch from Jira" in str(e)

