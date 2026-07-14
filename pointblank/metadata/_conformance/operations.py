"""Operation implementations for the native conformance engine.

Each operation pre-processes the target dataset, adding computed columns that the condition
evaluator can then reference. Operations are executed sequentially before conditions run.

Computed column naming convention: `_pb_<original_col>_<suffix>` (e.g. `_pb_SEX_valid`).
Rule catalog condition nodes reference these names as `{"name": "_pb_SEX_valid", ...}`.

Registered operations
---------------------
codelist_check          -- _pb_<col>_valid         (True = value in codelist or null)
consistency_check       -- _pb_<col>_consistent    (True = value matches dataset mode, or null)
iso8601_check           -- _pb_<col>_iso8601       (True = valid ISO 8601 partial/full datetime or null)
unique_per_subject      -- _pb_<col>_unique        (True = value is unique within the USUBJID group)
column_presence         -- _pb_<col>_present       (True = column exists in the dataset, scalar broadcast)
has_required_variables  -- _pb_<col>_present       (batch column_presence for a list of columns)
valid_variable_order    -- _pb_variable_order_valid (True = columns appear in expected relative order)
variable_type_check     -- _pb_<col>_type_valid    (True = column dtype matches expected category)

Define-XML–aware operations (require define_meta to be non-None; always True otherwise)
----------------------------------------------------------------------------------------
define_var_declared     -- _pb_<col>_in_define     (True = variable is declared in Define-XML)
define_required_check   -- _pb_<col>_mandatory_ok  (True = Mandatory variable has no nulls; row-level)
define_codelist_check   -- _pb_<col>_define_valid  (True = value is in Define-XML codelist; row-level)
define_type_check       -- _pb_<col>_define_type_ok (True = dtype matches Define-XML declared type)
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
    define_meta: Any = None,
) -> nw.DataFrame:
    """Apply all operations to `df`, returning an enriched DataFrame.

    Parameters
    ----------
    define_meta
        Optional `MetadataImport` for the current domain, used by Define-XML-aware operations. Pass
        `None` (the default) when no Define-XML has been provided; those operations will return
        `True` (pass) without flagging anything.
    """
    for op in operations:
        operator = op.get("operator", "")
        params = op.get("params", {})
        handler = _REGISTRY.get(operator)
        if handler is None:
            continue
        try:
            df = handler(df, params, ct, datasets, define_meta)
        except Exception:
            pass  # a failing operation silently skips; conditions that reference its column won't fire
    return df


# ── Operation handlers ────────────────────────────────────────────────────────


def _op_codelist_check(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_<col>_valid` (`True` = value in codelist or null)."""
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
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_<col>_consistent` (`True` = value equals the mode, or null)."""
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
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_<col>_iso8601` (`True` = valid ISO 8601 partial/complete datetime or null)."""
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
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_<col>_unique` (`True` = value is unique within the USUBJID group)."""
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
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_<col>_present` broadcast scalar (`True` = column exists in the dataset)."""
    col: str = params["column"]
    result_col = f"_pb_{col}_present"
    present = col in df.columns
    return df.with_columns(nw.lit(present).alias(result_col))


def _op_has_required_variables(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
    define_meta: Any = None,
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
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_variable_order_valid` broadcast scalar.

    Checks that the variables listed in `params["expected_order"]` appear in the correct relative
    order when both are present. Variables absent from the dataset are skipped (presence is a
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
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_<col>_type_valid` broadcast scalar.

    Checks whether the column's dtype category matches `params["expected_type"]`. Accepted values
    for `expected_type`: `"character"` (string/object) or `"numeric"` (integer/float). If the column
    is absent, result is `True`.
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


def _op_define_var_declared(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_<col>_in_define` broadcast scalar (True = variable is declared in Define-XML).

    If no Define-XML metadata is available, always returns `True` so the rule does not fire.
    """
    col: str = params["column"]
    result_col = f"_pb_{col}_in_define"
    if define_meta is None:
        return df.with_columns(nw.lit(True).alias(result_col))
    declared = {v.name.upper() for v in define_meta.variables}
    return df.with_columns(nw.lit(col.upper() in declared).alias(result_col))


def _op_define_required_check(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_<col>_mandatory_ok` per-row (True = value is non-null when Define-XML says mandatory).

    Rows where the variable is not mandatory, or Define-XML is absent, all get `True`.
    """
    col: str = params["column"]
    result_col = f"_pb_{col}_mandatory_ok"
    if define_meta is None or col not in df.columns:
        return df.with_columns(nw.lit(True).alias(result_col))
    var_meta = next((v for v in define_meta.variables if v.name.upper() == col.upper()), None)
    if var_meta is None or not var_meta.required:
        return df.with_columns(nw.lit(True).alias(result_col))
    # True = ok (value present); False = null where mandatory
    mask = [v is not None for v in df[col].to_list()]
    return df.with_columns(_new_bool_series(result_col, mask, df))


def _op_define_codelist_check(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_<col>_define_valid` per-row (True = value is in the Define-XML codelist or null).

    Uses `allowed_values` from the variable's `VariableMetadata`. If the variable has no codelist
    reference in Define-XML, or Define-XML is absent, always returns `True`.
    """
    col: str = params["column"]
    result_col = f"_pb_{col}_define_valid"
    if define_meta is None or col not in df.columns:
        return df.with_columns(nw.lit(True).alias(result_col))
    var_meta = next((v for v in define_meta.variables if v.name.upper() == col.upper()), None)
    if var_meta is None or not var_meta.allowed_values:
        return df.with_columns(nw.lit(True).alias(result_col))
    allowed = {str(v) for v in var_meta.allowed_values}
    values = df[col].to_list()
    mask = [True if v is None else (str(v) in allowed) for v in values]
    return df.with_columns(_new_bool_series(result_col, mask, df))


def _op_define_type_check(
    df: nw.DataFrame,
    params: dict,
    ct: ControlledTerminology,
    datasets: dict[str, nw.DataFrame],
    define_meta: Any = None,
) -> nw.DataFrame:
    """Add `_pb_<col>_define_type_ok` broadcast scalar (True = dtype matches Define-XML declared type).

    Translates the `display_format` (raw CDISC type) to a character/numeric category and compares it
    against the column's actual narwhals dtype. If Define-XML is absent or the column is not
    declared, returns `True`.
    """
    col: str = params["column"]
    result_col = f"_pb_{col}_define_type_ok"
    if define_meta is None or col not in df.columns:
        return df.with_columns(nw.lit(True).alias(result_col))
    var_meta = next((v for v in define_meta.variables if v.name.upper() == col.upper()), None)
    if var_meta is None:
        return df.with_columns(nw.lit(True).alias(result_col))

    # Map Define-XML type to expected category
    from narwhals import dtypes as _dtypes

    dtype = df[col].dtype
    is_numeric = isinstance(dtype, _dtypes.NumericType)

    define_type = (var_meta.display_format or var_meta.dtype or "").lower()
    numeric_types = {"integer", "float", "numeric", "int64", "float64", "int32", "float32"}
    char_types = {
        "text",
        "string",
        "character",
        "char",
        "datetime",
        "date",
        "time",
        "partialdate",
        "partialdatetime",
        "durationdatetime",
        "incompletedatetime",
    }
    if any(t in define_type for t in numeric_types):
        valid = is_numeric
    elif any(t in define_type for t in char_types):
        valid = not is_numeric
    else:
        valid = True  # unknown type → don't flag

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
    "define_var_declared": _op_define_var_declared,
    "define_required_check": _op_define_required_check,
    "define_codelist_check": _op_define_codelist_check,
    "define_type_check": _op_define_type_check,
}
