"""Operation implementations for the native conformance engine.

Each operation pre-processes the target dataset, adding computed columns that the condition
evaluator can then reference. Operations are executed sequentially before conditions run.

Computed column naming convention: `_pb_<original_col>_<suffix>` (e.g. `_pb_SEX_valid`).
Rule catalog condition nodes reference these names as `{"name": "_pb_SEX_valid", ...}`.

Registered operations
---------------------
codelist_check          -- _pb_<col>_valid       (True = value in codelist or null)
consistency_check       -- _pb_<col>_consistent  (True = value matches dataset mode, or null)
iso8601_check           -- _pb_<col>_iso8601     (True = valid ISO 8601 partial/full datetime or null)
unique_per_subject      -- _pb_<col>_unique      (True = value is unique within the USUBJID group)
column_presence         -- _pb_<col>_present     (True = column exists in the dataset, scalar broadcast)
has_required_variables  -- _pb_<col>_present     (batch column_presence for a list of columns)
valid_variable_order    -- _pb_variable_order_valid (True = columns appear in expected relative order)
variable_type_check     -- _pb_<col>_type_valid  (True = column dtype matches expected category)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import narwhals as nw

from pointblank.metadata._conformance.ct import ControlledTerminology
from pointblank.metadata._conformance.evaluator import is_iso8601


def _new_bool_series(name: str, values: list[bool], df: nw.DataFrame) -> nw.Series:
    ns = nw.get_native_namespace(df)
    return nw.new_series(name, values, dtype=nw.Boolean, backend=ns)


def apply_operations(
    df: nw.DataFrame,
    operations: list[dict],
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
) -> nw.DataFrame:
    """Apply all operations to `df`, returning an enriched DataFrame."""
    for op in operations:
        operator = op.get("operator", "")
        params = op.get("params", {})
        handler = _REGISTRY.get(operator)
        if handler is None:
            continue
        try:
            df = handler(df, params, ct, datasets)
        except Exception:
            pass  # a failing operation silently skips; conditions that reference its column won't fire
    return df


# ── Operation handlers ────────────────────────────────────────────────────────


def _op_codelist_check(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
) -> nw.DataFrame:
    """Add `_pb_<col>_valid` (True = value in codelist or null)."""
    col: str = params["column"]
    codelist: str = params["codelist"]
    result_col = f"_pb_{col}_valid"
    if col not in df.columns:
        return df.with_columns(nw.lit(True).alias(result_col))
    terms = ct.get_codelist(codelist)
    if terms is None:
        return df.with_columns(nw.lit(True).alias(result_col))
    values = df[col].to_list()
    mask = [True if v is None else (str(v) in terms) for v in values]
    return df.with_columns(_new_bool_series(result_col, mask, df))


def _op_consistency_check(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
) -> nw.DataFrame:
    """Add `_pb_<col>_consistent` (True = value equals the mode, or null)."""
    col: str = params["column"]
    result_col = f"_pb_{col}_consistent"
    if col not in df.columns:
        return df.with_columns(nw.lit(True).alias(result_col))
    values = [v for v in df[col].to_list() if v is not None]
    if not values:
        return df.with_columns(nw.lit(True).alias(result_col))
    expected = Counter(values).most_common(1)[0][0]
    rows = df[col].to_list()
    mask = [True if v is None else (v == expected) for v in rows]
    return df.with_columns(_new_bool_series(result_col, mask, df))


def _op_iso8601_check(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
) -> nw.DataFrame:
    """Add `_pb_<col>_iso8601` (True = valid ISO 8601 partial/complete datetime or null)."""
    col: str = params["column"]
    result_col = f"_pb_{col}_iso8601"
    if col not in df.columns:
        return df.with_columns(nw.lit(True).alias(result_col))
    values = df[col].to_list()
    mask = [True if v is None else is_iso8601(str(v)) for v in values]
    return df.with_columns(_new_bool_series(result_col, mask, df))


def _op_unique_per_subject(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
) -> nw.DataFrame:
    """Add `_pb_<col>_unique` (True = value is unique within the USUBJID group)."""
    col: str = params["column"]
    result_col = f"_pb_{col}_unique"
    if col not in df.columns or "USUBJID" not in df.columns:
        return df.with_columns(nw.lit(True).alias(result_col))
    counts = (
        df.group_by(["USUBJID", col])
        .agg(nw.len().alias("_n"))
        .filter(nw.col("_n") > 1)
        .select(["USUBJID", col])
    )
    dup_pairs: set[tuple] = set(zip(counts["USUBJID"].to_list(), counts[col].to_list()))
    rows_usubjid = df["USUBJID"].to_list()
    rows_col = df[col].to_list()
    mask = [(u, v) not in dup_pairs for u, v in zip(rows_usubjid, rows_col)]
    return df.with_columns(_new_bool_series(result_col, mask, df))


def _op_column_presence(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
) -> nw.DataFrame:
    """Add `_pb_<col>_present` broadcast scalar (True = column exists in the dataset)."""
    col: str = params["column"]
    result_col = f"_pb_{col}_present"
    present = col in df.columns
    return df.with_columns(nw.lit(present).alias(result_col))


def _op_has_required_variables(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
) -> nw.DataFrame:
    """Add `_pb_<col>_present` broadcast columns for each variable in `params["variables"]`.

    Equivalent to running `column_presence` for each variable in the list. Conditions can then check
    individual variables or combine them with `any`/`all`.
    """
    for col in params.get("variables", []):
        result_col = f"_pb_{col}_present"
        df = df.with_columns(nw.lit(col in df.columns).alias(result_col))
    return df


def _op_valid_variable_order(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
) -> nw.DataFrame:
    """Add `_pb_variable_order_valid` broadcast scalar.

    Checks that the variables listed in `params["expected_order"]` appear in the correct relative
    order when both are present.  Variables absent from the dataset are skipped (presence is a
    separate check). Result is `True` if no order violation is found.
    """
    expected: list[str] = params.get("expected_order", [])
    # Filter to variables that are actually in the dataset and record their positions.
    positions = {col: df.columns.index(col) for col in expected if col in df.columns}
    ordered_expected = [col for col in expected if col in positions]
    valid = True
    for i in range(len(ordered_expected) - 1):
        a, b = ordered_expected[i], ordered_expected[i + 1]
        if positions[a] > positions[b]:
            valid = False
            break
    return df.with_columns(nw.lit(valid).alias("_pb_variable_order_valid"))


def _op_variable_type_check(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
) -> nw.DataFrame:
    """Add `_pb_<col>_type_valid` broadcast scalar.

    Checks whether the column's dtype category matches `params["expected_type"]`. Accepted values
    for `expected_type`: `"character"` (string/object) or `"numeric"` (integer/float).  If the
    column is absent, result is `True`.
    """
    col: str = params["column"]
    expected: str = params.get("expected_type", "character").lower()
    result_col = f"_pb_{col}_type_valid"
    if col not in df.columns:
        return df.with_columns(nw.lit(True).alias(result_col))
    dtype = df[col].dtype
    from narwhals import dtypes as _dtypes

    is_numeric = isinstance(dtype, _dtypes.NumericType)
    is_string = isinstance(dtype, _dtypes.String)
    if expected == "numeric":
        valid = is_numeric
    elif expected == "character":
        valid = is_string
    else:
        valid = True  # unknown expected type → don't flag
    return df.with_columns(nw.lit(valid).alias(result_col))


_REGISTRY: dict[str, Any] = {
    "codelist_check": _op_codelist_check,
    "consistency_check": _op_consistency_check,
    "iso8601_check": _op_iso8601_check,
    "unique_per_subject": _op_unique_per_subject,
    "column_presence": _op_column_presence,
    "has_required_variables": _op_has_required_variables,
    "valid_variable_order": _op_valid_variable_order,
    "variable_type_check": _op_variable_type_check,
}
