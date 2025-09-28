from types import SimpleNamespace
from unittest.mock import patch, MagicMock
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


@patch("src.integration.team_connectors.requests.post")
def test_jira_push_partial_failure(mock_post):
    """JiraConnector.push_test_cases returns False on partial push failures."""
    from src.integration.team_connectors import JiraConnector

    err = requests.exceptions.RequestException("fail")
    ok_resp = MagicMock(status_code=201)
    ok_resp.raise_for_status.return_value = None
    mock_post.side_effect = [err, ok_resp]

    conn = JiraConnector("https://example", "u", "t", "PROJ")
    ok = conn.push_test_cases([_dummy_case(), _dummy_case()])
    assert ok is False
    assert mock_post.call_count == 2

