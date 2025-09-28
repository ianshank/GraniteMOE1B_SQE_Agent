from unittest.mock import patch, MagicMock
import pytest


@patch("src.integration.team_connectors.requests.get")
def test_jira_fetch_empty_issues(mock_get):
    from src.integration.team_connectors import JiraConnector

    resp = MagicMock(status_code=200)
    resp.json.return_value = {"issues": []}
    mock_get.return_value = resp

    conn = JiraConnector("https://ex", "u", "tok", "PROJ")
    reqs = conn.fetch_requirements()
    assert reqs == []


@patch("src.integration.team_connectors.requests.get")
def test_jira_fetch_missing_fields_raises(mock_get):
    from src.integration.team_connectors import JiraConnector

    resp = MagicMock(status_code=200)
    resp.json.return_value = {"issues": [{"fields": {"summary": "S"}}]}  # priority, issuetype missing
    mock_get.return_value = resp

    conn = JiraConnector("https://ex", "u", "tok", "PROJ")
    with pytest.raises(Exception):
        conn.fetch_requirements()


@patch("src.integration.team_connectors.requests.get")
def test_github_fetch_missing_keys_raises(mock_get):
    from src.integration.team_connectors import GitHubConnector

    resp = MagicMock(status_code=200)
    # Missing 'number' and 'title'
    resp.json.return_value = [{"id": 1}]
    mock_get.return_value = resp

    conn = GitHubConnector("o", "r", "tok")
    with pytest.raises(Exception):
        conn.fetch_requirements()

