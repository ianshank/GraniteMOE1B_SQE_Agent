from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.rag.code_indexer import CodebaseIndexer


def test_code_indexer_indexes_files(tmp_path: Path):
    # Arrange: create a small Python file
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    py_file = src_dir / "app.py"
    py_file.write_text(
        """
from fastapi import FastAPI
app = FastAPI()

@app.get("/ping")
def ping():
    try:
        return {"ok": True}
    except Exception:
        return {"ok": False}
""",
        encoding="utf-8",
    )

    received: List[Tuple[str, Dict[str, Any]]] = []

    def add_documents(payload: List[Tuple[str, Dict[str, Any]]]):
        received.extend(payload)

    indexer = CodebaseIndexer(add_documents, chunk_size=200, chunk_overlap=50)

    # Act
    stats = indexer.index_paths([str(src_dir)], exclude_globs=["**/.git/**"])

    # Assert
    assert stats["files"] == 1
    assert stats["snippets"] >= 1
    assert stats["errors"] == 0
    assert len(received) >= 1
    # Ensure metadata has expected keys
    _, meta = received[0]
    assert "language" in meta and meta["language"] == "python"
    assert "pattern_type" in meta
    assert "file_path" in meta and meta["file_path"].endswith("app.py")


