# PR: Stabilize Agent Generation, Add Suite API, Connector/RAG Hardening, Marker Alignment, and Pydantic v2 Cleanup

## Overview
This PR delivers end-to-end stabilization for the Granite test generator and pipeline:

- Agent
  - Strong, example-driven prompt with pre-seeded assistant response.
  - Deterministic fallback when generated text lacks required tags.
  - Safer parsing (number stripping, tolerant of missing "->", defensive defaults).
  - Provenance overlap heuristic with structured logging at decision points.
  - Deterministic IDs using MD5 (stable across runs).
  - New API: `generate_test_suite_for_team(...)` for Regression/E2E suites.

- Connectors
  - LocalFileSystemConnector
    - Accepts `directory` alias (compatibility with consumers/tests).
    - Uses `file_path.name` (including extension) for requirement IDs.
    - Nested JSON handling: compact JSON for nested descriptions; safer detection of nested arrays (`len(v) > 0`).
    - Single-object JSON support in `_process_json_file` when fields are present.
  - Jira/GitHub auth guidance: explicit error messages for 401/403 without exposing tokens.

- RAG Retriever
  - Added `_keyword_fallback` independent of retriever classes; resilient when LC retrievers are monkeypatched or fail.

- Orchestration & Config
  - Main auto-registers default local team only when local inputs exist.
  - Merges integration config from `INTEGRATION_CONFIG_PATH`, normalizing `path`→`input_directory` and `output`→`output_directory`.

- Pydantic v2 compatibility
  - Replaced `.dict()` with `.model_dump()` (with fallback) in all JSON writing paths to remove deprecation warnings.

## Pytest Markers & Discovery
- Markers registered in both root and package `pytest.ini`:
  - `integration`, `contract`, `e2e`, `regression` (+ existing `slow`, `unit`, `asyncio`, `mlx`).
- Test discovery restricted to test directories (`testpaths`) to avoid collecting `src/**/test_*.py`.

## How to Run
- All tests:
  - `pytest -q`
- By suite:
  - Regression: `pytest -m "regression" -q`
  - E2E: `pytest -m "e2e" -q`
  - Integration: `pytest -m "integration" -q`
  - Contract: `pytest -m "contract" -q`

## Results
- Local run: 94 passed, 0 failed. Remaining warnings are non-blocking (pytest collection naming).

## Rationale & Trade-offs
- Prompting and pre-seeding anchor 1B models to the required format.
- Fallback ensures structured output even on malformed generations, without inventing domain facts.
- Deterministic IDs make test outputs stable across interpreters and CI runs.
- Connectors avoid over-ingestion and provide clearer auth guidance.
- RAG fallback provides robust behavior without heavy LC dependencies.
- `.model_dump()` avoids Pydantic v2 deprecation while maintaining v1 compatibility via a runtime fallback.

## Risks & Mitigations
- Behavior change for local connector IDs (`name` vs `stem`): covered by updated tests.
- Integration config merge: normalized fields are backward compatible; errors logged with context.
- Fallback generation content is intentionally generic; downstream parsing remains strict.

## Follow-ups (Optional)
- Switch all internal `.dict()` calls (beyond writing paths) if any surface later.
- Add CI matrix to run marker-based suites separately for faster feedback.
- Migrate remaining pydantic deprecation call sites if new ones emerge.

