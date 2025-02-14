try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError

try:  # pragma: no cover
    __version__ = version("pointblank")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Import objects from the module
from pointblank.tf import TF
from pointblank.column import (
    col,
    starts_with,
    ends_with,
    contains,
    matches,
    everything,
    first_n,
    last_n,
)
from pointblank.validate import (
    Validate,
    load_dataset,
    config,
    preview,
    missing_vals_tbl,
    get_column_count,
    get_row_count,
)
from pointblank.schema import Schema
from pointblank.thresholds import Thresholds
from pointblank.datascan import DataScan
from pointblank.draft import DraftValidation

__all__ = [
    "TF",
    "Validate",
    "Thresholds",
    "Schema",
    "DataScan",
    "DraftValidation",
    "col",
    "starts_with",
    "ends_with",
    "contains",
    "matches",
    "everything",
    "first_n",
    "last_n",
    "load_dataset",
    "config",
    "preview",
    "missing_vals_tbl",
    "get_column_count",
    "get_row_count",
]
