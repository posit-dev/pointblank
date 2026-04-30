from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pointblank.field import (
    BoolField,
    DateField,
    DatetimeField,
    DurationField,
    Field,
    FloatField,
    IntField,
    StringField,
)

if TYPE_CHECKING:
    pass

__all__ = ["infer_fields_from_table"]

# ---------------------------------------------------------------------------
# Preset detection: column-name heuristics
# ---------------------------------------------------------------------------

# Normalize a column name to snake_case for matching. Handles:
# - camelCase / PascalCase: "firstName" -> "first_name"
# - dot/space separators: "first.name", "First Name" -> "first_name"
# - multiple separators: "First__Name" -> "first_name"
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _normalize_col_name(name: str) -> str:
    """Normalize column name to lowercase snake_case for heuristic matching."""
    # Insert underscores at camelCase boundaries
    name = _CAMEL_SPLIT_RE.sub("_", name)
    # Replace dots, spaces, dashes with underscores
    name = re.sub(r"[\s.\-]+", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name.lower().strip("_")


# Maps regex patterns (matched against normalized column names) to preset names.
# Order matters: more specific patterns should come first. Matches across camelCase, PascalCase,
# snake_case, dot.case, and abbreviations.
_NAME_TO_PRESET: list[tuple[re.Pattern[str], str]] = [
    # Identifiers
    (re.compile(r"(?:^|_)(?:uu|gu)?id$"), None),  # bare "id" columns → role, not preset
    (re.compile(r"(?:^|_)uuid(?:$|_|\d)"), "uuid4"),
    (re.compile(r"(?:^|_)guid(?:$|_|\d)"), "uuid4"),
    (re.compile(r"(?:^|_)iban(?:$|_|\d)"), "iban"),
    (re.compile(r"(?:^|_)ssn(?:$|_|\d)"), "ssn"),
    # Internet
    (re.compile(r"(?:^|_)ipv6(?:$|_)"), "ipv6"),
    (re.compile(r"(?:^|_)ipv4(?:$|_)"), "ipv4"),
    (re.compile(r"(?:^|_)(?:ip_addr|ip_address|ip)(?:$|_)"), "ipv4"),
    (re.compile(r"(?:^|_)(?:url|website|link|href|web|site|page|www)(?:$|_)"), "url"),
    (re.compile(r"(?:^|_)(?:domain|domain_name)(?:$|_)"), "domain_name"),
    (re.compile(r"(?:^|_)(?:user_?name|username|login)(?:$|_)"), "user_name"),
    # Email (before name patterns)
    (re.compile(r"(?:^|_)e?-?mail(?:$|_)"), "email"),
    (re.compile(r"(?:^|_)email_?addr(?:ess)?(?:$|_)"), "email"),
    # Phone
    (re.compile(r"(?:^|_)(?:phone|tel|telephone|mobile|cell|fax)(?:$|_|_?num)"), "phone_number"),
    # Address components (specific before generic)
    (re.compile(r"(?:^|_)(?:zip|postal|postcode|zip_?code|plz)(?:$|_)"), "postcode"),
    (
        re.compile(r"(?:^|_)(?:state|province|prov|region|division|district|land|pref)(?:$|_)"),
        "state",
    ),
    (re.compile(r"(?:^|_)(?:city|town|municipality|ort)(?:$|_)"), "city"),
    (re.compile(r"(?:^|_)(?:address|street|addr|strasse|rue)(?:$|_)"), "address"),
    (re.compile(r"(?:^|_)(?:country_?code|country_?cd|iso[23]|ctry)(?:$|_)"), "country_code_2"),
    (re.compile(r"(?:^|_)country(?:$|_)"), "country"),
    (re.compile(r"(?:^|_)(?:latitude|lat)(?:$|_)"), "latitude"),
    (re.compile(r"(?:^|_)(?:longitude|lng|lon|long)(?:$|_)"), "longitude"),
    # Person names (specific before generic)
    (re.compile(r"(?:^|_)(?:first_?name|fname|given_?name|vorname)(?:$|_)"), "first_name"),
    (
        re.compile(r"(?:^|_)(?:last_?name|lname|surname|family_?name|nachname)(?:$|_)"),
        "last_name",
    ),
    (re.compile(r"(?:^|_)(?:full_?name|name_full)(?:$|_)"), "name_full"),
    (re.compile(r"(?:^|_)name(?:$|_)"), "name"),
    # Business
    (re.compile(r"(?:^|_)(?:company|org|organization|employer|firma)(?:$|_)"), "company"),
    (re.compile(r"(?:^|_)(?:job|job_?title|position|occupation|beruf)(?:$|_)"), "job"),
    # Financial
    (re.compile(r"(?:^|_)(?:currency|curr|cy)(?:$|_)"), "currency_code"),
]

# Column-name patterns that indicate a role (not a preset) — used for
# inferring uniqueness and ID-like behavior on non-string columns too.
_ID_COLUMN_RE = re.compile(
    r"(?:^|_)(?:uu|gu)?id(?:$|_)|"  # id, uid, guid, uuid
    r"(?:^|_)\w+_id$|"  # user_id, order_id
    r"^id_\w+",  # id_user
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Preset detection: value-based validators
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_IPV4_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
_URL_RE = re.compile(r"^https?://")


def _validate_preset_values(preset: str, sample_values: list[str]) -> bool:
    """Check whether a sample of values is consistent with a preset.

    Uses proportion-based thresholds with different confidence levels for different presets.
    """
    if not sample_values:
        return False

    # Only validate presets where we have good heuristics
    validators: dict[str, tuple[re.Pattern[str], float]] = {
        "email": (_EMAIL_RE, 0.7),
        "uuid4": (_UUID_RE, 0.8),
        "ipv4": (_IPV4_RE, 0.8),
        "url": (_URL_RE, 0.7),
    }

    entry = validators.get(preset)
    if entry is None:
        # No validator for this preset — trust the name heuristic
        return True

    pattern, threshold = entry
    matches = sum(1 for v in sample_values if pattern.search(v))
    return matches / len(sample_values) >= threshold


def _validate_numeric_as_latitude(values: list[Any]) -> bool:
    """Check if numeric values fall within valid latitude bounds [-90, 90]."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return False
    within = sum(1 for v in non_null if -90 <= float(v) <= 90)
    return within / len(non_null) >= 0.9


def _validate_numeric_as_longitude(values: list[Any]) -> bool:
    """Check if numeric values fall within valid longitude bounds [-180, 180]."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return False
    within = sum(1 for v in non_null if -180 <= float(v) <= 180)
    return within / len(non_null) >= 0.9


def _detect_preset(col_name: str, sample_values: list[str]) -> str | None:
    """Detect a preset for a string column from its name and values.

    Uses a two-phase approach:

    1. Name heuristics with camelCase/snake_case/dot.case normalization
    2. Value-based validation with proportion thresholds
    """
    normalized = _normalize_col_name(col_name)

    for pattern, preset in _NAME_TO_PRESET:
        if preset is None:
            continue  # role-only patterns (e.g., bare "id")
        if pattern.search(normalized):
            # Validate against actual values
            if _validate_preset_values(preset, sample_values):
                return preset

    return None


def _detect_numeric_role(col_name: str, values: list[Any]) -> str | None:
    """Detect if a numeric column has a geographic role based on name + value bounds.

    Returns 'latitude' or 'longitude' if detected, None otherwise. Checks both column name patterns
    AND value bounds.
    """
    normalized = _normalize_col_name(col_name)

    if re.search(r"(?:^|_)(?:latitude|lat)(?:$|_)", normalized):
        if _validate_numeric_as_latitude(values):
            return "latitude"

    if re.search(r"(?:^|_)(?:longitude|lng|lon|long)(?:$|_)", normalized):
        if _validate_numeric_as_longitude(values):
            return "longitude"

    return None


def _is_id_column(col_name: str) -> bool:
    """Detect if a column name suggests it's an identifier/primary key.

    Recognizes patterns like "id", "user_id", "ID", "userId", "recordId", etc.
    """
    normalized = _normalize_col_name(col_name)
    return bool(_ID_COLUMN_RE.search(normalized))


# ---------------------------------------------------------------------------
# Per-dtype inference
# ---------------------------------------------------------------------------


def _infer_int_field(
    col_name: str,
    values: list[Any],
    null_count: int,
    total_count: int,
    categorical_threshold: int | float,
) -> IntField:
    """Infer an IntField from observed integer values."""
    non_null = [v for v in values if v is not None]

    kwargs: dict[str, Any] = {"dtype": "Int64"}

    if non_null:
        min_val = min(non_null)
        max_val = max(non_null)
        kwargs["min_val"] = int(min_val)
        kwargs["max_val"] = int(max_val)

        # Check uniqueness — also force unique if name looks like an ID column
        is_unique = len(non_null) > 1 and len(set(non_null)) == len(non_null)
        if is_unique or (_is_id_column(col_name) and is_unique):
            kwargs["unique"] = True

        # Categorical detection
        unique_count = len(set(non_null))
        threshold = (
            categorical_threshold
            if isinstance(categorical_threshold, int)
            else int(categorical_threshold * total_count)
        )
        if unique_count <= threshold and not kwargs.get("unique", False):
            kwargs["allowed"] = sorted(set(int(v) for v in non_null))
            # Remove min/max when using allowed
            kwargs.pop("min_val", None)
            kwargs.pop("max_val", None)

    # Nullability
    if null_count > 0:
        kwargs["nullable"] = True
        kwargs["null_probability"] = round(null_count / total_count, 4)

    return IntField(**kwargs)


def _infer_float_field(
    col_name: str,
    values: list[Any],
    null_count: int,
    total_count: int,
) -> FloatField | StringField:
    """Infer a FloatField from observed float values.

    May return a StringField with a preset if the column is detected as latitude/longitude,
    following an approach that checks both column name AND value bounds for geographic roles.
    """
    non_null = [v for v in values if v is not None]

    # Check for geographic role (name + value bounds validation)
    geo_role = _detect_numeric_role(col_name, values)
    if geo_role in ("latitude", "longitude"):
        kwargs: dict[str, Any] = {"preset": geo_role}
        if null_count > 0:
            kwargs["nullable"] = True
            kwargs["null_probability"] = round(null_count / total_count, 4)
        return StringField(**kwargs)

    kwargs = {"dtype": "Float64"}

    if non_null:
        kwargs["min_val"] = float(min(non_null))
        kwargs["max_val"] = float(max(non_null))

    # Nullability
    if null_count > 0:
        kwargs["nullable"] = True
        kwargs["null_probability"] = round(null_count / total_count, 4)

    return FloatField(**kwargs)


def _infer_string_field(
    col_name: str,
    values: list[Any],
    null_count: int,
    total_count: int,
    categorical_threshold: int | float,
    detect_presets: bool,
) -> StringField:
    """Infer a StringField from observed string values."""
    non_null = [str(v) for v in values if v is not None]

    kwargs: dict[str, Any] = {}

    # Try preset detection first
    if detect_presets and non_null:
        sample = non_null[:200]  # sample for validation
        preset = _detect_preset(col_name, sample)
        if preset is not None:
            kwargs["preset"] = preset
            # When using a preset, skip length/allowed inference
            if null_count > 0:
                kwargs["nullable"] = True
                kwargs["null_probability"] = round(null_count / total_count, 4)
            if len(non_null) > 1 and len(set(non_null)) == len(non_null):
                kwargs["unique"] = True
            return StringField(**kwargs)

    # Categorical detection
    if non_null:
        unique_count = len(set(non_null))
        threshold = (
            categorical_threshold
            if isinstance(categorical_threshold, int)
            else int(categorical_threshold * total_count)
        )
        if unique_count <= threshold:
            kwargs["allowed"] = sorted(set(non_null))
        else:
            # Length constraints
            lengths = [len(s) for s in non_null]
            kwargs["min_length"] = min(lengths)
            kwargs["max_length"] = max(lengths)

            # Uniqueness
            if len(non_null) > 1 and unique_count == len(non_null):
                kwargs["unique"] = True

    # Nullability
    if null_count > 0:
        kwargs["nullable"] = True
        kwargs["null_probability"] = round(null_count / total_count, 4)

    return StringField(**kwargs)


def _infer_bool_field(
    values: list[Any],
    null_count: int,
    total_count: int,
) -> BoolField:
    """Infer a BoolField from observed boolean values."""
    non_null = [v for v in values if v is not None]

    kwargs: dict[str, Any] = {}

    if non_null:
        true_count = sum(1 for v in non_null if v)
        kwargs["p_true"] = round(true_count / len(non_null), 4)

    # Nullability
    if null_count > 0:
        kwargs["nullable"] = True
        kwargs["null_probability"] = round(null_count / total_count, 4)

    return BoolField(**kwargs)


def _infer_date_field(
    values: list[Any],
    null_count: int,
    total_count: int,
) -> DateField:
    """Infer a DateField from observed date values."""
    non_null = [v for v in values if v is not None]

    kwargs: dict[str, Any] = {}

    if non_null:
        kwargs["min_date"] = min(non_null)
        kwargs["max_date"] = max(non_null)

    # Nullability
    if null_count > 0:
        kwargs["nullable"] = True
        kwargs["null_probability"] = round(null_count / total_count, 4)

    return DateField(**kwargs)


def _infer_datetime_field(
    values: list[Any],
    null_count: int,
    total_count: int,
) -> DatetimeField:
    """Infer a DatetimeField from observed datetime values."""
    non_null = [v for v in values if v is not None]

    kwargs: dict[str, Any] = {}

    if non_null:
        kwargs["min_date"] = min(non_null)
        kwargs["max_date"] = max(non_null)

    # Nullability
    if null_count > 0:
        kwargs["nullable"] = True
        kwargs["null_probability"] = round(null_count / total_count, 4)

    return DatetimeField(**kwargs)


def _infer_duration_field(
    values: list[Any],
    null_count: int,
    total_count: int,
) -> DurationField:
    """Infer a DurationField from observed duration values."""
    non_null = [v for v in values if v is not None]

    kwargs: dict[str, Any] = {}

    if non_null:
        kwargs["min_duration"] = min(non_null)
        kwargs["max_duration"] = max(non_null)

    # Nullability
    if null_count > 0:
        kwargs["nullable"] = True
        kwargs["null_probability"] = round(null_count / total_count, 4)

    return DurationField(**kwargs)


# ---------------------------------------------------------------------------
# Dtype classification helpers
# ---------------------------------------------------------------------------

_INT_DTYPES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
_FLOAT_DTYPES = {"float32", "float64"}


def _classify_dtype(dtype_str: str) -> str:
    """Classify a dtype string into a category: int, float, string, bool, date, datetime, duration."""
    d = dtype_str.lower()

    if d in _INT_DTYPES or "int" in d:
        return "int"
    if d in _FLOAT_DTYPES or "float" in d or d == "double":
        return "float"
    if d in ("bool", "boolean"):
        return "bool"
    if "datetime" in d or "timestamp" in d:
        return "datetime"
    if d == "date":
        return "date"
    if d == "time":
        return "time"
    if "duration" in d or "timedelta" in d:
        return "duration"
    # Default to string
    return "string"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def infer_fields_from_table(
    tbl: Any,
    *,
    categorical_threshold: int | float = 20,
    detect_presets: bool = True,
    sample_size: int | None = None,
) -> list[tuple[str, Field]]:
    """
    Inspect a table and infer Field objects for each column.

    Parameters
    ----------
    tbl
        A Polars DataFrame, Pandas DataFrame, or Ibis table (DuckDB, SQLite, etc.).
    categorical_threshold
        If a column has <= this many unique values (int) or this fraction of rows (float in 0..1),
        treat as categorical with `allowed=`.
    detect_presets
        Attempt to match string columns to known generation presets.
    sample_size
        If set, sample this many rows before analysis. None uses all rows.

    Returns
    -------
    list[tuple[str, Field]]
        List of (column_name, Field) tuples ready for Schema construction.
    """
    import narwhals as nw

    from pointblank._constants import IBIS_BACKENDS
    from pointblank._utils import _get_tbl_type

    table_type = _get_tbl_type(tbl)

    # Convert to a common representation via Narwhals or native APIs
    if table_type == "polars":
        col_names, col_dtypes, col_values = _extract_polars(tbl, sample_size)
    elif table_type == "pandas":
        col_names, col_dtypes, col_values = _extract_pandas(tbl, sample_size)
    elif table_type in IBIS_BACKENDS:
        col_names, col_dtypes, col_values = _extract_ibis(tbl, sample_size)
    else:
        # Try via Narwhals for other backends (Ibis/DuckDB, SQLite, etc.)
        try:
            nw_df = nw.from_native(tbl)
            col_names, col_dtypes, col_values = _extract_narwhals(nw_df, sample_size)
        except Exception:
            raise TypeError(
                f"Cannot infer fields from table type '{table_type}'. "
                "Supported types: Polars DataFrame, Pandas DataFrame, Ibis table."
            )

    # Infer a Field for each column
    total_count = len(col_values[0]) if col_values else 0
    result: list[tuple[str, Field]] = []

    for i, (name, dtype_str) in enumerate(zip(col_names, col_dtypes)):
        values = col_values[i]
        null_count = sum(1 for v in values if v is None)
        category = _classify_dtype(dtype_str)

        if category == "int":
            field_obj = _infer_int_field(
                name, values, null_count, total_count, categorical_threshold
            )
        elif category == "float":
            field_obj = _infer_float_field(name, values, null_count, total_count)
        elif category == "string":
            field_obj = _infer_string_field(
                name, values, null_count, total_count, categorical_threshold, detect_presets
            )
        elif category == "bool":
            field_obj = _infer_bool_field(values, null_count, total_count)
        elif category == "date":
            field_obj = _infer_date_field(values, null_count, total_count)
        elif category == "datetime":
            field_obj = _infer_datetime_field(values, null_count, total_count)
        elif category == "duration":
            field_obj = _infer_duration_field(values, null_count, total_count)
        else:
            # Fallback: plain string
            field_obj = _infer_string_field(
                name, values, null_count, total_count, categorical_threshold, detect_presets
            )

        result.append((name, field_obj))

    return result


# ---------------------------------------------------------------------------
# Table extraction helpers
# ---------------------------------------------------------------------------


def _extract_polars(
    tbl: Any, sample_size: int | None
) -> tuple[list[str], list[str], list[list[Any]]]:
    """Extract column info from a Polars DataFrame."""
    if sample_size is not None and len(tbl) > sample_size:
        tbl = tbl.sample(n=sample_size, seed=23)

    col_names = tbl.columns
    col_dtypes = [str(tbl.schema[c]) for c in col_names]
    col_values = [tbl[c].to_list() for c in col_names]
    return col_names, col_dtypes, col_values


def _extract_pandas(
    tbl: Any, sample_size: int | None
) -> tuple[list[str], list[str], list[list[Any]]]:
    """Extract column info from a Pandas DataFrame."""
    if sample_size is not None and len(tbl) > sample_size:
        tbl = tbl.sample(n=sample_size, random_state=42)

    col_names = list(tbl.columns)
    col_dtypes = [str(tbl[c].dtype) for c in col_names]

    col_values = []
    for c in col_names:
        # Convert pandas values, replacing NaN/NaT with None
        series = tbl[c]
        vals = []
        for v in series:
            try:
                import pandas as pd

                if pd.isna(v):
                    vals.append(None)
                else:
                    vals.append(v)
            except (TypeError, ValueError):
                vals.append(v)
        col_values.append(vals)

    return col_names, col_dtypes, col_values


def _extract_ibis(
    tbl: Any, sample_size: int | None
) -> tuple[list[str], list[str], list[list[Any]]]:
    """Extract column info from an Ibis table by executing to Pandas."""
    if sample_size is not None:
        tbl = tbl.head(sample_size)

    # Execute the Ibis expression to a Pandas DataFrame
    pdf = tbl.to_pandas()
    return _extract_pandas(pdf, None)


def _extract_narwhals(
    nw_df: Any, sample_size: int | None
) -> tuple[list[str], list[str], list[list[Any]]]:
    """Extract column info from a Narwhals-wrapped DataFrame."""
    import narwhals as nw

    if sample_size is not None:
        nw_df = nw_df.head(sample_size)

    # Collect if lazy
    if hasattr(nw_df, "collect"):
        nw_df = nw_df.collect()

    schema = dict(nw_df.schema.items())
    col_names = list(schema.keys())
    col_dtypes = [str(v) for v in schema.values()]

    # Convert to native and extract values
    native = nw.to_native(nw_df)
    col_values = []
    for c in col_names:
        if hasattr(native, "to_list"):
            # Polars-like
            col_values.append(native[c].to_list())
        else:
            # Pandas-like
            col_values.append(native[c].tolist())

    return col_names, col_dtypes, col_values
