from unittest.mock import patch
import requests
import pytest


@patch("src.integration.team_connectors.requests.get")
def test_github_fetch_unauthorized_guidance(mock_get):
    from src.integration.team_connectors import GitHubConnector

    resp = requests.Response()
    resp.status_code = 401
    err = requests.exceptions.HTTPError(response=resp)
    mock_get.side_effect = err

    conn = GitHubConnector("o", "r", "tok")
    with pytest.raises(Exception) as exc:
        conn.fetch_requirements()
    assert "Unauthorized (401)" in str(exc.value)
    assert "token" in str(exc.value)


@patch("src.integration.team_connectors.requests.get")
def test_jira_fetch_unauthorized_guidance(mock_get):
    from src.integration.team_connectors import JiraConnector

    resp = requests.Response()
    resp.status_code = 401
    err = requests.exceptions.HTTPError(response=resp)
    mock_get.side_effect = err

    conn = JiraConnector("https://ex", "u", "tok", "P")
    with pytest.raises(Exception) as exc:
        conn.fetch_requirements()
    assert "Unauthorized (401)" in str(exc.value)
    assert "API token" in str(exc.value)

