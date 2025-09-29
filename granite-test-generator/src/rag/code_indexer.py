"""
Pluggable code indexer for code-aware RAG.

This module extracts lightweight code patterns and metadata from source files
and forwards prepared document+metadata pairs to a caller-provided callback.
It contains no persistence or vector-store logic by design, keeping it
read-only and easily testable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    # Preferred split package (when available)
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:  # pragma: no cover
    try:
        # Legacy path in LangChain core
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:  # pragma: no cover
        # Lightweight fallback splitter with similar interface
        class RecursiveCharacterTextSplitter:  # type: ignore
            def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None) -> None:
                self.chunk_size = int(chunk_size)
                self.chunk_overlap = int(chunk_overlap)
                self.separators = separators or ["\n\n", "\n", " "]

            def split_text(self, text: str) -> List[str]:
                if not text:
                    return []
                # If separators available, first try to split on largest then merge windows
                chunks: List[str] = [text]
                for sep in self.separators:
                    next_chunks: List[str] = []
                    for c in chunks:
                        next_chunks.extend(c.split(sep))
                    chunks = [c for c in next_chunks if c]
                    if len(chunks) > 1:
                        break
                # Sliding window packing to approximate target sizes
                packed: List[str] = []
                i = 0
                while i < len(text):
                    end = min(len(text), i + self.chunk_size)
                    packed.append(text[i:end])
                    if end == len(text):
                        break
                    i = max(i + self.chunk_size - self.chunk_overlap, i + 1)
                return packed

from src.rag.models import CodeLanguage, CodePattern


logger = logging.getLogger(__name__)


class CodebaseIndexer:
    """Indexes code files into chunked documents for external storage.

    The caller supplies `add_documents` which accepts a list of tuples:
    (page_content: str, metadata: Dict[str, Any]). The indexer focuses on
    chunking, pattern extraction, and metadata construction only.
    """

    def __init__(
        self,
        add_documents: Callable[[List[Tuple[str, Dict[str, Any]]]], None],
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        ext_to_lang: Optional[Dict[str, CodeLanguage]] = None,
    ) -> None:
        self.add_documents = add_documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
            or ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""],
        )
        self.ext_to_lang = ext_to_lang or {
            ".py": CodeLanguage.python,
            ".js": CodeLanguage.javascript,
            ".ts": CodeLanguage.typescript,
            ".java": CodeLanguage.java,
            ".cs": CodeLanguage.csharp,
            ".cpp": CodeLanguage.cpp,
            ".cc": CodeLanguage.cpp,
            ".go": CodeLanguage.go,
            ".rs": CodeLanguage.rust,
            ".sql": CodeLanguage.sql,
            ".html": CodeLanguage.html,
            ".css": CodeLanguage.css,
        }

    def index_paths(
        self,
        roots: Iterable[str],
        exclude_globs: Iterable[str] | None = None,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Index code across provided roots.

        Args:
            roots: Iterable of directory roots.
            exclude_globs: Glob patterns to exclude (e.g., node_modules, venv).

        Returns:
            Stats dict with counts and error details.
        """
        stats: Dict[str, Any] = {
            "files": 0,
            "snippets": 0,
            "errors": 0,
            "errors_detail": [],
        }

        exclude_globs = list(exclude_globs or [])
        files: List[Path] = []

        for root in roots:
            root_path = Path(root)
            if not root_path.exists():
                logger.debug("Skipping non-existent root: %s", root_path)
                continue
            for p in root_path.rglob("*"):
                if p.is_file() and p.suffix in self.ext_to_lang:
                    files.append(p)

        def _excluded(p: Path) -> bool:
            s = str(p)
            return any(Path(s).match(g) for g in exclude_globs) if exclude_globs else False

        payload: List[Tuple[str, Dict[str, Any]]] = []
        for fp in files:
            if _excluded(fp):
                logger.debug("Excluded by glob: %s", fp)
                continue
            try:
                lang = self.ext_to_lang[fp.suffix]
                content = fp.read_text(encoding="utf-8", errors="ignore")
                chunks = self.splitter.split_text(content)
                for idx, chunk in enumerate(chunks):
                    meta = self._build_metadata(content, chunk, lang, str(fp), idx)
                    if base_metadata:
                        # Merge and allow language override if provided
                        merged = {**meta, **base_metadata}
                        if 'language' in base_metadata and base_metadata['language']:
                            merged['language'] = str(base_metadata['language'])
                        meta = merged
                    doc = self._make_doc(chunk, meta)
                    payload.append((doc, meta))
                stats["snippets"] += len(chunks)
                stats["files"] += 1
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Failed to process %s: %s", fp, e)
                stats["errors"] += 1
                stats["errors_detail"].append({"file": str(fp), "error": str(e)})

        if payload:
            logger.debug("Forwarding %d docs to add_documents()", len(payload))
            self.add_documents(payload)
        else:
            logger.debug("No documents extracted to forward")

        logger.info(
            "Code indexing completed: files=%d, snippets=%d, errors=%d",
            stats["files"], stats["snippets"], stats["errors"],
        )
        return stats

    def _build_metadata(
        self,
        full: str,
        chunk: str,
        lang: CodeLanguage,
        path: str,
        idx: int,
    ) -> Dict[str, Any]:
        line_start = full[: full.find(chunk)].count("\n") + 1
        line_end = line_start + chunk.count("\n")
        pattern, desc = self._extract_pattern(chunk, lang)
        q, c, r = self._score(chunk, lang)
        return {
            "language": lang.value,
            "pattern_type": pattern.value,
            "description": desc,
            "file_path": path,
            "line_start": line_start,
            "line_end": line_end,
            "quality_score": q,
            "complexity_score": c,
            "reusability_score": r,
            "chunk_index": idx,
            "chunk_size": len(chunk),
        }

    def _extract_pattern(self, chunk: str, lang: CodeLanguage) -> Tuple[CodePattern, str]:
        # Minimal heuristics; extend per language if needed.
        if lang == CodeLanguage.python and re.search(r"@app\.|FastAPI|flask|router\.\w+", chunk):
            return CodePattern.api_endpoint, "API endpoint"
        if re.search(r"\btry\b|\bexcept\b|\braise\b|\bcatch\b|\bthrow\b", chunk):
            return CodePattern.error_handling, "Error handling"
        if re.search(r"\bclass\s+\w+", chunk):
            return CodePattern.class_definition, "Class"
        if re.search(r"\bdef\s+\w+\(|function\s+\w+\(", chunk):
            return CodePattern.function_definition, "Function"
        if re.search(r"SELECT|INSERT|UPDATE|DELETE", chunk, re.IGNORECASE):
            return CodePattern.database_query, "Database query"
        if re.search(r"auth|login|jwt|oauth|token|session", chunk, re.IGNORECASE):
            return CodePattern.authentication, "Authentication"
        return CodePattern.configuration, "Configuration/Generic"

    def _score(self, chunk: str, lang: CodeLanguage) -> Tuple[float, float, float]:
        lines = [ln for ln in chunk.split("\n") if ln.strip()]
        n = max(1, len(lines))
        comments = len(re.findall(r"#.*|//.*|/\*.*?\*/", chunk))
        quality = min(1.0, 0.5 + min(0.2, (comments / n) * 2))
        branches = len(
            re.findall(r"\bif\b|\belse\b|\belif\b|\bfor\b|\bwhile\b|\btry\b|\bexcept\b|\bcatch\b", chunk)
        )
        complexity = min(1.0, (branches / n) * 2)
        reuse = 0.3
        if re.search(r"\bdef\s+\w+\(|function\s+\w+\(", chunk):
            reuse += 0.3
        if re.search(r"\bclass\s+\w+", chunk):
            reuse += 0.2
        if "(" in chunk and ")" in chunk:
            reuse += 0.1
        return float(min(1.0, quality)), float(min(1.0, complexity)), float(min(1.0, reuse))

    def _make_doc(self, chunk: str, meta: Dict[str, Any]) -> str:
        return (
            f"Language: {meta['language']}\n"
            f"Pattern: {meta['pattern_type']}\n"
            f"Description: {meta['description']}\n"
            f"File: {meta['file_path']}\n\n"
            f"Code:\n{chunk}\n"
        )


