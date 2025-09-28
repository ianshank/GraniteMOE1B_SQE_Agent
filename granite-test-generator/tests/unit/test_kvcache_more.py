from pathlib import Path
import builtins as _builtins
from typing import Any

from src.utils.kv_cache import KVCache


def test_store_write_failure_returns_empty_key(tmp_path: Path, monkeypatch):
    cache = KVCache(cache_dir=str(tmp_path))

    real_open = _builtins.open

    def failing_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):  # type: ignore[override]
        # Fail only for .pkl write attempts
        if str(file).endswith(".pkl") and "wb" in mode:
            raise OSError("disk error")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)

    key = cache.store("c", {"x": 1}, response="r")
    assert key == ""  # write failure should return empty key
    assert cache.metadata == {}  # no metadata persisted


def test_retrieve_by_tags_missing_file(tmp_path: Path):
    cache = KVCache(cache_dir=str(tmp_path))
    # Manually inject metadata for a non-existent file
    cache.metadata = {"dead": {"tags": ["a"], "timestamp": 0.0, "size": 1}}
    res = cache.retrieve_by_tags(["a"])  # should handle missing file gracefully
    assert res == []


def test_store_large_content_updates_size(tmp_path: Path):
    cache = KVCache(cache_dir=str(tmp_path))
    big = "x" * 10000
    key = cache.store(big, {"k": 1}, response="ok")
    assert isinstance(key, str) and key
    # Metadata entry exists and size recorded equals len(content)
    assert cache.metadata[key]["size"] == len(big)
    # Retrieval works
    entry = cache.retrieve(big, {"k": 1})
    assert entry is not None and entry["response"] == "ok"

