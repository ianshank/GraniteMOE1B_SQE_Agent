"""Pytest configuration and path setup to ensure `src` is importable.

This makes tests robust across environments and Python versions by
explicitly adding the project root (which contains `src/`) to sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root (the parent of the tests directory) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

