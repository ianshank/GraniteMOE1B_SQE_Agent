from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.rag.code_indexer import CodebaseIndexer


def test_team_specific_indexing_filters_language_and_team(tmp_path: Path):
    # Create two team directories with files
    ads_dir = tmp_path / "services" / "ads"
    cms_dir = tmp_path / "services" / "cms"
    ads_dir.mkdir(parents=True)
    cms_dir.mkdir(parents=True)

    (ads_dir / "handler.py").write_text("""
def serve_ad():
    # python impl
    return True
""", encoding="utf-8")

    (cms_dir / "page.js").write_text("""
function renderPage() {
  // js impl
  return true;
}
""", encoding="utf-8")

    collected: List[Dict[str, Any]] = []

    def add_documents(payload):
        for doc, meta in payload:
            collected.append(meta)

    indexer = CodebaseIndexer(add_documents, chunk_size=100, chunk_overlap=20)

    # Index team-specific roots with base metadata
    indexer.index_paths([str(ads_dir)], exclude_globs=["**/.git/**"], base_metadata={'team_context': 'ads', 'language': 'python'})
    indexer.index_paths([str(cms_dir)], exclude_globs=["**/.git/**"], base_metadata={'team_context': 'cms', 'language': 'javascript'})

    # Verify language and team attached correctly
    ads_metas = [m for m in collected if m.get('team_context') == 'ads']
    cms_metas = [m for m in collected if m.get('team_context') == 'cms']

    assert all(m.get('language') == 'python' for m in ads_metas)
    assert all(m.get('language') == 'javascript' for m in cms_metas)
    # No cross-bleed in team_context labeling
    assert not any(m.get('team_context') == 'cms' and (m.get('file_path') or '').endswith('handler.py') for m in collected)


