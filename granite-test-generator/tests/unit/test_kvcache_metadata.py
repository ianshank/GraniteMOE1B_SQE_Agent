from pathlib import Path
import json

from src.utils.kv_cache import KVCache


def test_kvcache_load_metadata_corruption(tmp_path: Path):
    """Corrupted metadata.json is handled gracefully and reset to empty."""
    # Write invalid JSON to metadata file
    (tmp_path / "metadata.json").write_text("{not: valid}")
    cache = KVCache(cache_dir=str(tmp_path))
    assert cache.metadata == {}


def test_kvcache_save_metadata_failure(tmp_path: Path, monkeypatch):
    """Saving metadata failures are logged and do not crash store()."""
    cache = KVCache(cache_dir=str(tmp_path))

    import builtins as _builtins

    real_open = _builtins.open

    def flaky_open(file, mode="r", *args, **kwargs):  # type: ignore[no-redef]
        if str(file) == str(cache.metadata_file) and "w" in mode:
            raise OSError("disk full")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", flaky_open)

    key = cache.store("c", {"x": 1}, response="r")
    # Store still returns a key even if metadata save failed
    assert isinstance(key, str) and len(key) > 0

