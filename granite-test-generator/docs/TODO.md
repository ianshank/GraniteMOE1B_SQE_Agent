# TODO

- Raise CI coverage threshold after additional tests land (target 40–60%).
- Expand orchestrator tests to cover:
  - Parallelism timing assertions (ensure concurrent execution reduces wall time).
  - Logging assertions for key stages (registration, fetch, generate, push, report).
- Add end-to-end test for main pipeline with team configs to validate per-team JSON output routing.
- Consider centralizing logging configuration with env-driven levels and structured format.
- Explore HTML coverage publication (e.g., GitHub Pages) for easier review.

