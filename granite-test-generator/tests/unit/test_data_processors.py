from pathlib import Path
from typing import List, Dict

from src.data.data_processors import TestCaseDataProcessor
from src.utils.chunking import DocumentChunk


class FakeChunker:
    def chunk_requirements(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        return [
            DocumentChunk(
                content=text + "-req",
                metadata={**metadata, "section_index": 0},
                chunk_id="rid",
                source_type="requirements",
                team_context=metadata.get("team", "unknown"),
            )
        ]

    def chunk_user_stories(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        return [
            DocumentChunk(
                content=text + "-story",
                metadata={**metadata, "story_index": 0},
                chunk_id="sid",
                source_type="user_story",
                team_context=metadata.get("team", "unknown"),
            )
        ]


def test_process_requirements_files(tmp_path: Path):
    # Create team dir and a requirements file
    team = tmp_path / "TeamA"
    team.mkdir()
    f = team / "doc1.txt"
    f.write_text("REQ-1: Login capability", encoding="utf-8")

    proc = TestCaseDataProcessor(FakeChunker())
    out = proc.process_requirements_files(str(tmp_path))
    assert len(out) == 1
    content, meta = out[0]
    assert content.endswith("-req")
    assert meta["doc_id"] == "doc1"
    assert meta["team"] == "TeamA"
    assert meta["file_path"].endswith("doc1.txt")


def test_process_user_stories_json(tmp_path: Path):
    stories = tmp_path / "stories.json"
    stories.write_text(
        """
[
  {"id": "S1", "story": "As a user, I want...", "team": "T", "priority": "high", "epic": "E1"}
]
""".strip(),
        encoding="utf-8",
    )
    proc = TestCaseDataProcessor(FakeChunker())
    out = proc.process_user_stories(str(stories))
    assert len(out) == 1
    content, meta = out[0]
    assert content.endswith("-story")
    assert meta["story_id"] == "S1"
    assert meta["team"] == "T"
    assert meta["priority"] == "high"
    assert meta["epic"] == "E1"


def test_process_user_stories_unsupported_format(tmp_path: Path):
    bad = tmp_path / "stories.txt"
    bad.write_text("n/a", encoding="utf-8")
    proc = TestCaseDataProcessor(FakeChunker())
    try:
        proc.process_user_stories(str(bad))
        assert False, "Expected ValueError for unsupported format"
    except ValueError as e:  # noqa: BLE001
        assert "Supported formats" in str(e)

