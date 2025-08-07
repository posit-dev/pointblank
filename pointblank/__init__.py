try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:  # pragma: no cover
    __version__ = version("pointblank")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Import objects from the module
from pointblank.actions import send_slack_notification
from pointblank.assistant import assistant
from pointblank.column import (
    col,
    contains,
    ends_with,
    everything,
    expr_col,
    first_n,
    last_n,
    matches,
    starts_with,
)
from pointblank.datascan import DataScan, col_summary_tbl
from pointblank.draft import DraftValidation
from pointblank.schema import Schema
from pointblank.segments import seg_group
from pointblank.tf import TF
from pointblank.thresholds import Actions, FinalActions, Thresholds
from pointblank.validate import (
    Validate,
    config,
    connect_to_table,
    get_action_metadata,
    get_column_count,
    get_data_path,
    get_row_count,
    get_validation_summary,
    load_dataset,
    missing_vals_tbl,
    preview,
)
from pointblank.yaml import (
    validate_yaml,
    yaml_interrogate,
    yaml_to_python,
)

__all__ = [
    "assistant",
    "TF",
    "Validate",
    "Thresholds",
    "Actions",
    "FinalActions",
    "Schema",
    "DataScan",
    "DraftValidation",
    "col",
    "expr_col",
    "col_summary_tbl",
    "starts_with",
    "ends_with",
    "contains",
    "matches",
    "everything",
    "first_n",
    "last_n",
    "load_dataset",
    "get_data_path",
    "config",
    "connect_to_table",
    "preview",
    "missing_vals_tbl",
    "get_action_metadata",
    "get_validation_summary",
    "get_column_count",
    "get_row_count",
    "seg_group",
    "send_slack_notification",
    # YAML functionality
    "yaml_interrogate",
    "validate_yaml",
    "yaml_to_python",
]
