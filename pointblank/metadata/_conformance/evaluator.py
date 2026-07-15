"""Narwhals-based condition evaluator for the native conformance engine.

Translates the JSON condition tree used in the rule catalog into narwhals boolean
expressions and evaluates them against a DataFrame, returning a boolean mask (one value
per row: True = rule condition is met = violation, False = no violation).

Condition tree grammar
----------------------
A condition tree is a dict with one of these shapes:

    {"all": [<condition>, ...]}       all sub-conditions must hold (AND)
    {"any": [<condition>, ...]}       at least one sub-condition must hold (OR)
    {"not": <condition>}              invert the sub-condition
    {                                  leaf (single operator check)
      "name":     <column name or "$" prefix for computed columns>,
      "operator": <operator name>,
      "value":    <literal value or list>,
    }

Supported operators (leaf nodes)
---------------------------------
is_null, is_not_null,
equal_to, not_equal_to,
greater_than, greater_than_or_equal_to,
less_than, less_than_or_equal_to,
contains, not_contains,
starts_with, ends_with,
is_in, not_in,
matches_regex,
equal_to_column, not_equal_to_column,
"""

from __future__ import annotations

import re
from functools import reduce
from typing import Any

import narwhals as nw


class EvaluationError(Exception):
    """Raised when a condition references a column that does not exist."""


def evaluate_conditions(df: nw.DataFrame, conditions: dict) -> nw.Series:
    """Evaluate a condition tree against `df`.

    Returns a boolean Series (True = row matches the violation condition).
    Returns an all-False Series when the condition tree is empty.
    """
    if not conditions:
        ns = nw.get_native_namespace(df)
        return nw.new_series("_match", [False] * len(df), dtype=nw.Boolean, backend=ns)
    try:
        expr = _compile(conditions)
        return df.select(expr.alias("_match"))["_match"]
    except Exception as exc:
        raise EvaluationError(str(exc)) from exc


def _compile(cond: dict) -> nw.Expr:
    if "all" in cond:
        sub = [_compile(c) for c in cond["all"]]
        return reduce(lambda a, b: a & b, sub)
    if "any" in cond:
        sub = [_compile(c) for c in cond["any"]]
        return reduce(lambda a, b: a | b, sub)
    if "not" in cond:
        return ~_compile(cond["not"])
    return _compile_leaf(cond)


def _compile_leaf(cond: dict) -> nw.Expr:
    name: str = cond["name"]
    op: str = cond["operator"]
    value: Any = cond.get("value")

    col = nw.col(name)

    if op == "is_null":
        return col.is_null()
    if op == "is_not_null":
        return ~col.is_null()
    if op == "equal_to":
        return col == value
    if op == "not_equal_to":
        return col != value
    if op == "greater_than":
        return col > value
    if op == "greater_than_or_equal_to":
        return col >= value
    if op == "less_than":
        return col < value
    if op == "less_than_or_equal_to":
        return col <= value
    if op == "contains":
        return col.cast(nw.String).str.contains(str(value))
    if op == "not_contains":
        return ~col.cast(nw.String).str.contains(str(value))
    if op == "starts_with":
        return col.cast(nw.String).str.starts_with(str(value))
    if op == "ends_with":
        return col.cast(nw.String).str.ends_with(str(value))
    if op == "is_in":
        terms = list(value) if not isinstance(value, list) else value
        return col.is_in(terms)
    if op == "not_in":
        terms = list(value) if not isinstance(value, list) else value
        return ~col.is_in(terms)
    if op == "matches_regex":
        return col.cast(nw.String).str.contains(str(value))
    if op == "equal_to_column":
        return col == nw.col(str(value))
    if op == "not_equal_to_column":
        return col != nw.col(str(value))

    raise ValueError(f"Unknown operator: {op!r}")


# ── ISO 8601 date helpers (used by date-format operation) ─────────────────────

# Allows YYYY, YYYY-MM, YYYY-MM-DD, YYYY-MM-DDTHH:MM, YYYY-MM-DDTHH:MM:SS, with optional
# timezone offset or Z. Partial dates are permitted by SDTM convention.
_ISO8601_RE = re.compile(
    r"^\d{4}(-\d{2}(-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?)?)?)?$"
)


def is_iso8601(value: str | None) -> bool:
    """Return True if `value` is a non-null, non-empty ISO 8601 partial/complete datetime string."""
    if not value or not isinstance(value, str):
        return False
    return bool(_ISO8601_RE.match(value.strip()))
