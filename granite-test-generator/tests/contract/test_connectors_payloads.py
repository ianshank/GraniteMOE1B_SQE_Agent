from unittest.mock import patch, MagicMock
from types import SimpleNamespace


def _tc():
    return SimpleNamespace(
        id="TC1",
        summary="Login",
        priority=SimpleNamespace(value="medium"),
        test_type=SimpleNamespace(value="functional"),
        preconditions=["User exists"],
        steps=[SimpleNamespace(step_number=1, action="Open page", expected_result="Page shown")],
        expected_results="Success",
        requirements_traced=["R1"],
        input_data={"a": 1},
    )


@patch("src.integration.team_connectors.requests.post")
def test_jira_push_payload_format_and_timeout(mock_post):
    from src.integration.team_connectors import JiraConnector

    ok = MagicMock(status_code=201)
    ok.raise_for_status.return_value = None
    mock_post.return_value = ok

    conn = JiraConnector("https://ex", "u", "tok", "PROJ")
    conn.push_test_cases([_tc()])

    assert mock_post.called
    args, kwargs = mock_post.call_args
    assert kwargs.get("timeout") == 30
    payload = kwargs["json"]
    assert payload["fields"]["project"]["key"] == "PROJ"
    assert payload["fields"]["issuetype"]["name"] == "Test"
    desc = payload["fields"]["description"]
    assert "*Test Steps:*" in desc
    assert "1. Open page" in desc and "_Expected:_ Page shown" in desc


@patch("src.integration.team_connectors.requests.post")
def test_github_push_payload_format_and_timeout(mock_post):
    from src.integration.team_connectors import GitHubConnector

    ok = MagicMock(status_code=201)
    ok.raise_for_status.return_value = None
    mock_post.return_value = ok

    conn = GitHubConnector("o", "r", "tok")
    conn.push_test_cases([_tc()])

    assert mock_post.called
    args, kwargs = mock_post.call_args
    assert kwargs.get("timeout") == 30
    body = kwargs["json"]
    assert body["title"].startswith("Test Case: Login")
    assert "### Test Steps" in body["body"]
    assert "1. Open page" in body["body"] and "Expected" in body["body"]
    assert "test-case" in body["labels"] and "functional" in body["labels"] and "medium" in body["labels"]

