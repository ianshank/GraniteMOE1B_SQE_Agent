from pathlib import Path
import time

from src.utils.kv_cache import KVCache


def test_evict_oldest_missing_file(tmp_path: Path):
    """_evict_oldest_entry removes metadata when file is missing and logs warning."""
    cache = KVCache(cache_dir=str(tmp_path))
    # Simulate two metadata entries; oldest will be removed
    cache.metadata = {
        "old": {"tags": [], "timestamp": time.time() - 100, "size": 1},
        "new": {"tags": [], "timestamp": time.time(), "size": 1},
    }
    # No corresponding old.pkl present
    cache.save_metadata()

    cache._evict_oldest_entry()
    assert "old" not in cache.metadata
    assert "new" in cache.metadata

