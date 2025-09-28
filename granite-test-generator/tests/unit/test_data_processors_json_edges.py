from pathlib import Path
from typing import List, Dict

from src.data.data_processors import TestCaseDataProcessor
from src.utils.chunking import DocumentChunk


class FakeChunker:
    def chunk_user_stories(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        return [
            DocumentChunk(
                content=text,
                metadata=metadata,
                chunk_id="u1",
                source_type="user_story",
                team_context=metadata.get("team", "unknown"),
            )
        ]


def test_user_stories_json_missing_optional_and_nulls(tmp_path: Path):
    # story None -> fallback to description; missing team/priority/epic -> defaults
    stories = tmp_path / "stories.json"
    stories.write_text(
        """
[
  {"id": "S1", "story": null, "description": "Beschreibung", "team": null},
  {"id": "S2", "story": "As a user...", "priority": null, "epic": null}
]
""".strip(),
        encoding="utf-8",
    )

    proc = TestCaseDataProcessor(FakeChunker())
    out = proc.process_user_stories(str(stories))
    assert len(out) == 2

    (_, m1), (_, m2) = out
    # Item 1: text from description, defaults for team (unknown), priority (medium), epic ('')
    assert m1.get("team") == "unknown"
    # Item 2: explicit story, defaults for priority (medium) and epic ('')
    assert m2.get("priority") == "medium"
    assert m2.get("epic") == ""

