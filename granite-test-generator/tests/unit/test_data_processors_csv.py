import pytest
from pathlib import Path

from src.data.data_processors import TestCaseDataProcessor
from src.utils.chunking import DocumentChunk


# Require pandas for these CSV tests; skip module if not installed
pytest.importorskip("pandas", reason="pandas required for CSV tests")


class _Chunker:
    def chunk_user_stories(self, text, metadata):
        return [
            DocumentChunk(
                content=text, metadata=metadata, chunk_id="1", source_type="user_story", team_context=metadata.get("team")
            )
        ]


def test_process_user_stories_csv_single(tmp_path: Path):
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


def test_process_user_stories_csv_multi_rows_extra_columns(tmp_path: Path):
    csv = tmp_path / "stories.csv"
    csv.write_text(
        "\n".join(
            [
                "id,story,team,priority,epic,extra_col",
                "S1,Do X,T,medium,E1,foo",
                "S2,Do Y,U,high,E2,bar",
            ]
        ),
        encoding="utf-8",
    )

    proc = TestCaseDataProcessor(_Chunker())
    out = proc.process_user_stories(str(csv))
    assert len(out) == 2
    (_, m1), (_, m2) = out
    assert m1["story_id"] == "S1" and m1["team"] == "T" and m1["priority"] == "medium" and m1["epic"] == "E1"
    assert m2["story_id"] == "S2" and m2["team"] == "U" and m2["priority"] == "high" and m2["epic"] == "E2"
