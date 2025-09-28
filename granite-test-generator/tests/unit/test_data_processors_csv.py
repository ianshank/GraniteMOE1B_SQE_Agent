import pytest
from pathlib import Path

from src.data.data_processors import TestCaseDataProcessor
from src.utils.chunking import DocumentChunk


class _Chunker:
    def chunk_user_stories(self, text, metadata):
        return [DocumentChunk(content=text, metadata=metadata, chunk_id="1", source_type="user_story", team_context=metadata.get("team"))]


@pytest.mark.skipif(pytest.importorskip("pandas", reason="pandas required for CSV tests") is None, reason="pandas not available")
def test_process_user_stories_csv(tmp_path: Path):
    csv = tmp_path / "stories.csv"
    csv.write_text("id,story,team,priority,epic\nS1,Do X,T,medium,E1\n", encoding="utf-8")

    proc = TestCaseDataProcessor(_Chunker())
    out = proc.process_user_stories(str(csv))
    assert len(out) == 1
    content, meta = out[0]
    assert meta["story_id"] == "S1"
    assert meta["team"] == "T"
    assert meta["priority"] == "medium"
    assert meta["epic"] == "E1"

