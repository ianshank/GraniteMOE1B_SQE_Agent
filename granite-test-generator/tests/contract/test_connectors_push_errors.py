from unittest.mock import patch, MagicMock
import requests
from types import SimpleNamespace


def _tc():
    return SimpleNamespace(
        id="TC1",
        summary="S",
        priority=SimpleNamespace(value="medium"),
        test_type=SimpleNamespace(value="functional"),
        preconditions=[],
        steps=[],
        expected_results="",
        requirements_traced=[],
        input_data={},
    )


@patch("src.integration.team_connectors.requests.post")
def test_github_push_mixed_errors_continues(mock_post):
    from src.integration.team_connectors import GitHubConnector

    ok = MagicMock(status_code=201)
    ok.raise_for_status.return_value = None
    mock_post.side_effect = [requests.exceptions.Timeout(), ok, requests.exceptions.HTTPError("500")]  # type: ignore

    conn = GitHubConnector("o", "r", "tok")
    ok_flag = conn.push_test_cases([_tc(), _tc(), _tc()])
    assert ok_flag is False
    assert mock_post.call_count == 3


@patch("src.integration.team_connectors.requests.post")
def test_jira_push_mixed_errors_continues(mock_post):
    from src.integration.team_connectors import JiraConnector

    ok = MagicMock(status_code=201)
    ok.raise_for_status.return_value = None
    mock_post.side_effect = [requests.exceptions.HTTPError("400"), ok]

    conn = JiraConnector("https://ex", "u", "tok", "P")
    ok_flag = conn.push_test_cases([_tc(), _tc()])
    assert ok_flag is False
    assert mock_post.call_count == 2

