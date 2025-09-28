"""Pytest configuration and path setup to ensure `src` is importable.

This makes tests robust across environments and Python versions by
explicitly adding the project root (which contains `src/`) to sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Add project root (the parent of the tests directory) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically skip tests marked 'mlx' when MLX is unavailable.

    MLX (mlx_lm) is an Apple Silicon-specific stack and is typically not
    available in Linux CI. Tests that explicitly require MLX should be marked
    with `@pytest.mark.mlx`. This hook adds a skip marker in such environments.
    """
    try:
        from src.models.granite_moe import _MLX_AVAILABLE  # type: ignore
    except Exception:
        _MLX_AVAILABLE = False  # noqa: N806

    if _MLX_AVAILABLE:
        return

    skip_mlx = pytest.mark.skip(reason="MLX stack (mlx_lm) not available in this environment")
    for item in items:
        if "mlx" in item.keywords:
            item.add_marker(skip_mlx)
