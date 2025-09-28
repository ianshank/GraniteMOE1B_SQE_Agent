from pathlib import Path

from src.integration.team_connectors import LocalFileSystemConnector


def test_local_connector_reads_text_and_md(tmp_path: Path):
    """LocalFileSystemConnector loads .txt/.md files as requirements."""
    # Create sample files
    (tmp_path / "a.txt").write_text("Req A\nMore detail", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title B\nBody", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")

    conn = LocalFileSystemConnector(directory=str(tmp_path), team_name="teamX")
    reqs = conn.fetch_requirements()

    # Two requirements read; team name assigned
    assert len(reqs) == 2
    teams = {r["team"] for r in reqs}
    assert teams == {"teamX"}
    ids = {r["id"] for r in reqs}
    assert ids == {"a.txt", "b.md"}

