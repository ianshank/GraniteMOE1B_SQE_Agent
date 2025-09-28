import json
from src.integration.team_connectors import LocalFileSystemConnector


def _make_connector() -> LocalFileSystemConnector:
    # No FS access required for normalization; any directory string is acceptable
    return LocalFileSystemConnector(directory=".", team_name="unit")


def test_normalize_description_dict_roundtrip():
    conn = _make_connector()
    payload = {"a": 1, "b": {"c": 2}}
    out = conn._normalize_description(payload)
    # Returns a JSON string that round-trips to the same structure
    assert isinstance(out, str)
    assert json.loads(out) == payload


def test_normalize_description_string_and_primitive_passthrough():
    conn = _make_connector()
    text = "Simple description"
    out_text = conn._normalize_description(text)
    assert out_text == text

    # Non-string primitives are coerced to string
    num = 42
    out_num = conn._normalize_description(num)
    assert isinstance(out_num, str)
    assert out_num == "42"

