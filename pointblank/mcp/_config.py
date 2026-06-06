"""Shared configuration and state for the Pointblank MCP server."""

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Detect if we're running in a test environment
TESTING_MODE = (
    "pytest" in sys.modules
    or os.environ.get("PYTEST_CURRENT_TEST") is not None
    or os.environ.get("POINTBLANK_TESTING") == "true"
)

# Try to import Pandas, but make it optional
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

# Try to import other DataFrame libraries
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

# Maximum file size for loading (500 MB)
_MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024

# Allowed file extensions for data loading
_ALLOWED_DATA_EXTENSIONS = {".csv", ".xls", ".xlsx", ".parquet", ".json", ".jsonl"}

# Type alias for DataFrame: can be Pandas or Polars or other
if HAS_PANDAS:
    DataFrameType = pd.DataFrame
else:
    DataFrameType = Any


def _get_log_path() -> Path | None:
    """Get a writable log file path, falling back to temp directory."""
    candidates = [
        Path.home() / ".pointblank" / "mcp_server.log",
        Path("/tmp") / "pointblank_mcp_server.log",
    ]
    for p in candidates:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            continue
    return None


def configure_logging() -> logging.Logger:
    """Configure and return the MCP server logger."""
    log_path = _get_log_path()
    handlers: list = [logging.StreamHandler()]
    if log_path:
        handlers.append(logging.FileHandler(str(log_path)))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("pointblank.mcp")


logger = configure_logging()
