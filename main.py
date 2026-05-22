"""
Root-level entry point for Render (and any other host that runs from the
repo root rather than the nexus/ subdirectory).

Render's start command is:  uvicorn main:app --host 0.0.0.0 --port $PORT
All nexus/* modules use absolute imports (from core.config import …),
so we must add nexus/ to sys.path before importing them.
"""
import sys
from pathlib import Path

# Insert nexus/ as the first entry so its packages shadow nothing in stdlib.
_NEXUS_DIR = Path(__file__).parent / "nexus"
if str(_NEXUS_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS_DIR))

# Now all nexus-internal imports resolve correctly.
from api.main import app  # noqa: E402

__all__ = ["app"]
