from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import narwhals as nw

if TYPE_CHECKING:
    from great_tables import GT

__all__ = ["Comparison", "SchemaDiff", "compare"]

# Row statuses, in report order. One row status is assigned per test unit (an aligned row).
ROW_STATUSES = ("match", "changed", "missing", "extra", "dup_key")

# Internal column names used in the joined frame (chosen so they won't collide with user columns)
_TGT_SUFFIX = "__pb_t"
_IN_SOURCE = "__pb_in_s"
_IN_TARGET = "__pb_in_t"
_STATUS = "__pb_status"
_ROW_IDX = "__pb_row"
_N_SOURCE = "__pb_n_s"
_N_TARGET = "__pb_n_t"

# Column name used for the row number in positional mode (matches Pointblank's extracts)
_ROW_NUM_COL = "_row_num_"

# Stand-in for null values when rows are compared as whole (stringified) values in multiset mode
_NULL_SENTINEL = "\x00__pb_null__"

# Name of the status column in `tbl_match()` data extracts
_STATUS_OUT_COL = "_status_"

_NORMALIZERS = ("strip", "lower", "upper", "collapse_whitespace", "round")

_DURATION_UNITS = {
    "us": "microseconds",
    "ms": "milliseconds",
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


@dataclass(frozen=True)
class SchemaDiff:
    """
    Differences between the schemas of the source and target tables.

    Column names refer to the source table except in `only_in_target` (and the second element of
    `renamed` and `case_only` pairs), which refer to the target table.

    Attributes
    ----------
    only_in_source
        Columns present in the source but not the target.
    only_in_target
        Columns present in the target but not the source.
    dtype_changed
        `(column, source_dtype, target_dtype)` for shared columns whose data types differ.
    renamed
        `(source_column, target_column)` pairs paired up through `column_map=`.
    case_only
        `(source_column, target_column)` pairs whose names differ only by letter case. These are
        *not* paired up automatically (they also appear in `only_in_source`/`only_in_target`); use
        `column_map=` to compare them.
    reordered
        `True` if the shared columns appear in a different order in the two tables.
    """

    only_in_source: list[str] = field(default_factory=list)
    only_in_target: list[str] = field(default_factory=list)
    dtype_changed: list[tuple[str, str, str]] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    case_only: list[tuple[str, str]] = field(default_factory=list)
    reordered: bool = False

    @property
    def matches(self) -> bool:
        """`True` if the schemas match (explicit renames through `column_map=` are allowed)."""
        return not (
            self.only_in_source or self.only_in_target or self.dtype_changed or self.reordered
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": self.matches,
            "only_in_source": list(self.only_in_source),
            "only_in_target": list(self.only_in_target),
            "dtype_changed": [list(x) for x in self.dtype_changed],
            "renamed": [list(x) for x in self.renamed],
            "case_only": [list(x) for x in self.case_only],
            "reordered": self.reordered,
        }


@dataclass(frozen=True)
class _ColumnSpec:
    """How one source column is compared against its target counterpart."""

    source: str
    target: str
    source_dtype: Any
    target_dtype: Any
    atol: Any = None
    rtol: float | None = None
    normalize: tuple[tuple[str, Any], ...] = ()


def compare(
    source: Any,
    target: Any,
    keys: str | list[str] | None = None,
    columns: str | list[str] | None = None,
    column_map: dict[str, str] | None = None,
    tolerance: Any = None,
    normalize: dict[str, Any] | None = None,
    null_equal: bool = True,
    schema: Literal["strict", "common"] = "strict",
    dup_keys: Literal["flag", "compare"] = "flag",
    order_by: str | list[str] | None = None,
    partitions: int | None = None,
    source_name: str | None = None,
    target_name: str | None = None,
) -> Comparison:
    """
    Compare two tables row by row and report how they differ.

    The `compare()` function aligns the rows of a *source* table (the reference, or expected data)
    with those of a *target* table (the data being checked) and classifies every aligned row:
    rows that match, rows whose values changed, rows missing from the target, and extra rows in the
    target. It also reports schema differences (columns added, removed, retyped, or reordered). The
    returned `Comparison` object gives you the counts, samples of the differing rows, and a
    tabular report.

    Rows are aligned in one of two ways:

    - **by key** (`keys=` given): rows are matched on the values of one or more key columns, so
      row order doesn't matter. This is the most useful mode for comparing extracts of the same
      data (e.g., the source and target of a pipeline run).
    - **by position** (`keys=None`, the default): the first row of the source is compared with the
      first row of the target, and so on.

    The comparison is backend-agnostic and runs inside the table's own engine: tables can be
    Polars or Pandas DataFrames, Polars LazyFrames, Ibis tables (e.g., DuckDB, PostgreSQL), or
    paths to CSV/Parquet files. File paths are scanned lazily, and only counts and small samples of
    differing rows are ever collected, so tables that don't fit in memory can still be compared.

    Parameters
    ----------
    source
        The reference table. Can be any table type supported by Pointblank, a path to a CSV or
        Parquet file, or a callable that returns a table.
    target
        The table to check against the source. Same input types as `source=`.
    keys
        One or more columns that uniquely identify a row, used to align rows between the tables.
        Key names refer to the source table (use `column_map=` if the target names them
        differently). If `None`, rows are aligned by position, which requires the tables to have
        a defined row order (eager DataFrames, Polars LazyFrames, or any table with `order_by=`).
    columns
        The source columns whose values should be compared. By default, all columns present in
        both tables (other than the keys) are compared. Giving `columns=` also limits the schema
        check to the keys and these columns.
    column_map
        A mapping of source column names to target column names, for columns that were renamed.
    tolerance
        Tolerance for numeric and datetime comparisons. A number is an absolute tolerance applied
        to all numeric columns. A dict with `"atol"` and/or `"rtol"` keys applies to all numeric
        columns, where values are equal if `|source - target| <= atol + rtol * |source|`. A dict
        mapping column names to either of the above sets per-column tolerances. Datetime
        tolerances are given as a `datetime.timedelta` or a string like `"1s"`, `"500ms"`, `"2m"`,
        or `"1d"`, e.g., `tolerance={"updated_at": "1s"}`.
    normalize
        Per-column transformations applied to both sides before comparing, as a dict mapping a
        column name to a list of operations: `"strip"`, `"lower"`, `"upper"`,
        `"collapse_whitespace"`, or `{"round": n}`. Reported values are always the original ones.
    null_equal
        Should a null in the source and a null in the target count as equal? By default, they do.
        NaN values in floating-point columns are treated as null.
    schema
        How schema differences affect the result. With `"strict"` (the default), any schema
        difference (other than renames given in `column_map=`) means the tables don't match, and
        every row counts as failing. With `"common"`, only the shared columns are compared and
        schema differences are reported without failing rows.
    dup_keys
        Rows whose key appears more than once in either table can't be aligned unambiguously, so
        they always get the `"dup_key"` status. With `"flag"` (the default), they are only flagged
        and counted. With `"compare"`, the report also compares the duplicated rows pairwise
        (every source row against every target row with the same key) to show which columns
        differ.
    order_by
        Column(s) defining row order for positional alignment on backends without an inherent
        row order (e.g., database tables). Ignored when `keys=` is given.
    partitions
        Compare the tables in this many passes, each covering a disjoint subset of the key values
        (by key hash). This bounds peak memory for tables too large to compare in one pass (each
        pass joins only about `1/partitions` of the rows), at the cost of reading the tables once
        per pass. Requires `keys=`, and Polars, DuckDB, or Ibis tables (including CSV/Parquet
        paths). The results are identical to a single-pass comparison.
    source_name, target_name
        Optional labels for the two tables, used in the report.

    Returns
    -------
    Comparison
        The result of the comparison.

    Test Units and Row Statuses
    ---------------------------
    Every aligned row is one *test unit* and gets exactly one status:

    - `"match"`: the row is in both tables and all compared values are equal (within tolerance)
    - `"changed"`: the row is in both tables but at least one compared value differs
    - `"missing"`: the row is in the source but not in the target
    - `"extra"`: the row is in the target but not in the source
    - `"dup_key"`: the row's key appears more than once in either table. A duplicated key counts
      as many units as the larger of its two occurrence counts.

    Rows with a null in a key column can't be matched: they count as `"missing"` (source) or
    `"extra"` (target).

    Large Tables
    ------------
    The comparison runs inside the tables' own engine and only counts and small samples of rows
    are collected, so file paths (scanned lazily) and database tables never have to be loaded into
    Python. The engine's join still needs memory for the rows it compares. If that's too much, use
    `partitions=` to compare the tables in several smaller passes. For the lowest memory use with
    very large files, read them as DuckDB relations (e.g., `duckdb.connect().read_csv(path)`) and
    use `partitions=`: in a benchmark with two 10-million-row CSV files (730 MB each), peak memory
    was about 0.8 GB with DuckDB and `partitions=8`, versus about 5.5 GB to read both files with
    Polars and join them.

    Examples
    --------
    ```{python}
    import pointblank as pb
    import polars as pl

    source = pl.DataFrame(
        {"id": [1, 2, 3, 4], "amount": [10.0, 20.0, 30.0, 40.0], "region": ["N", "S", "E", "W"]}
    )
    target = pl.DataFrame(
        {"id": [1, 2, 4, 5], "amount": [10.0, 20.5, 40.0, 50.0], "region": ["N", "S", "W", "N"]}
    )

    cmp = pb.compare(source, target, keys="id")
    cmp
    ```

    The counts are available as attributes, and the differing rows can be retrieved by status:

    ```{python}
    cmp.n_changed, cmp.n_missing, cmp.n_extra
    ```

    ```{python}
    cmp.rows("changed")
    ```

    A tolerance can absorb small numeric differences:

    ```{python}
    pb.compare(source, target, keys="id", tolerance={"amount": 1}).n_changed
    ```
    """
    from pointblank.validate import _process_data

    keys_list = _as_list(keys)
    columns_list = _as_list(columns)
    order_by_list = _as_list(order_by)
    column_map = dict(column_map or {})

    if schema not in ("strict", "common"):
        raise ValueError(f"`schema=` must be 'strict' or 'common', not {schema!r}.")
    if dup_keys not in ("flag", "compare"):
        raise ValueError(f"`dup_keys=` must be 'flag' or 'compare', not {dup_keys!r}.")
    if partitions is not None:
        if isinstance(partitions, bool) or not isinstance(partitions, int) or partitions < 1:
            raise ValueError("`partitions=` must be a positive integer.")
        if not keys_list or keys_list == ["*"]:
            raise ValueError("`partitions=` requires `keys=` (rows are partitioned by key).")

    if source_name is None and isinstance(source, str):
        source_name = source
    if target_name is None and isinstance(target, str):
        target_name = target

    if callable(source) and not hasattr(source, "columns"):
        source = source()
    if callable(target) and not hasattr(target, "columns"):
        target = target()

    source = _process_data(source, lazy=True)
    target = _process_data(target, lazy=True)

    src, tgt = _align_backends(nw.from_native(source), nw.from_native(target))

    settings = dict(
        keys=keys_list or [],
        columns=columns_list,
        column_map=column_map,
        tolerance=tolerance,
        normalize=normalize,
        null_equal=null_equal,
        schema=schema,
        dup_keys=dup_keys,
        order_by=order_by_list,
        source_name=source_name or "source",
        target_name=target_name or "target",
    )
    if partitions is not None and partitions > 1:
        return Comparison._build_partitioned(src, tgt, partitions=partitions, **settings)
    return Comparison._build(src=src, tgt=tgt, **settings)


class Comparison:
    """
    The result of comparing a source table with a target table.

    `Comparison` objects are created by [`compare()`](`pointblank.compare`). They hold the row
    status counts, per-column mismatch counts, and the schema differences, and can retrieve samples
    of differing rows on demand.

    Attributes
    ----------
    source_name, target_name
        Labels for the two tables.
    keys
        The key columns used to align rows (empty for positional alignment).
    mode
        `"keyed"` or `"positional"`.
    schema_diff
        A [`SchemaDiff`](`pointblank.SchemaDiff`) describing schema differences.
    n_source_rows, n_target_rows
        Row counts of the two tables.
    n_match, n_changed, n_missing, n_extra, n_dup_key
        Number of test units (aligned rows) with each status.
    column_mismatches
        A dict mapping each compared source column to the number of matched rows where its values
        differ.
    """

    def __init__(self) -> None:  # pragma: no cover
        raise TypeError("Create a `Comparison` with `pb.compare()`.")

    # ------------------------------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------------------------------

    @classmethod
    def _build(
        cls,
        src: Any,
        tgt: Any,
        keys: list[str],
        columns: list[str] | None,
        column_map: dict[str, str],
        tolerance: Any,
        normalize: dict[str, Any] | None,
        null_equal: bool,
        schema: str,
        dup_keys: str,
        order_by: list[str] | None,
        source_name: str,
        target_name: str,
    ) -> Comparison:
        self = object.__new__(cls)
        self.source_name = source_name
        self.target_name = target_name
        multiset = keys == ["*"]
        if multiset:
            keys = []
        self.keys = list(keys)
        self.mode = "multiset" if multiset else ("keyed" if keys else "positional")
        self._order_by = order_by
        self._tgt_input = tgt
        self._key_casts: dict[str, Any] = {}
        self._parts: list[Comparison] | None = None
        self.partitions = 1
        self.schema_mode = schema
        self.dup_keys_mode = dup_keys
        self.null_equal = null_equal
        self._sample_cache: dict[tuple[str, int], Any] = {}

        src_schema = dict(src.collect_schema().items())
        tgt_schema = dict(tgt.collect_schema().items())
        self._src_columns = list(src_schema)
        self._tgt_columns = list(tgt_schema)

        # Validate the column map, keys, and columns subset
        for s_col, t_col in column_map.items():
            if s_col not in src_schema:
                raise ValueError(f"`column_map=` refers to {s_col!r}, which isn't in the source.")
            if t_col not in tgt_schema:
                raise ValueError(f"`column_map=` maps to {t_col!r}, which isn't in the target.")
        self.column_map = column_map

        def tgt_name(col: str) -> str:
            return column_map.get(col, col)

        for key in keys:
            if key not in src_schema:
                raise ValueError(f"Key column {key!r} isn't in the source table.")
            if tgt_name(key) not in tgt_schema:
                raise ValueError(
                    f"Key column {tgt_name(key)!r} isn't in the target table. If it was renamed, "
                    "map it with `column_map=`."
                )

        if columns is not None:
            for col in columns:
                if col not in src_schema:
                    raise ValueError(f"Column {col!r} given in `columns=` isn't in the source.")

        self.schema_diff = _schema_diff(
            src_schema=src_schema,
            tgt_schema=tgt_schema,
            column_map=column_map,
            scope=None if columns is None else list(keys) + list(columns),
        )

        # Determine the compared columns (source name -> target name)
        candidate = columns if columns is not None else [c for c in src_schema if c not in keys]
        tol_by_col = _parse_tolerance(tolerance)
        norm_by_col = _parse_normalize(normalize)
        for col in list(tol_by_col) + list(norm_by_col):
            if col != "*" and col not in src_schema:
                raise ValueError(
                    f"Column {col!r} given in `tolerance=`/`normalize=` isn't in the source."
                )

        specs: list[_ColumnSpec] = []
        for col in candidate:
            if col in keys or tgt_name(col) not in tgt_schema:
                continue
            s_dtype, t_dtype = src_schema[col], tgt_schema[tgt_name(col)]
            atol, rtol = tol_by_col.get(col, tol_by_col.get("*", (None, None)))
            if col not in tol_by_col and not _is_numeric(s_dtype):
                # A table-wide tolerance only applies to numeric columns
                atol, rtol = None, None
            specs.append(
                _ColumnSpec(
                    source=col,
                    target=tgt_name(col),
                    source_dtype=s_dtype,
                    target_dtype=t_dtype,
                    atol=atol,
                    rtol=rtol,
                    normalize=tuple(norm_by_col.get(col, ())),
                )
            )
        self._specs = specs
        self.compared_columns = [spec.source for spec in specs]

        if self.mode == "multiset":
            if tolerance is not None:
                raise ValueError(
                    "`tolerance=` can't be used with `keys='*'` (multiset comparison matches "
                    "whole rows exactly)."
                )
            self._build_multiset(src, tgt)
            return self

        # Positional alignment: add a row index on both sides and use it as the key
        if self.mode == "positional":
            src = _with_row_index(src, order_by, "source")
            tgt = _with_row_index(tgt, order_by, "target")
            join_keys = [_ROW_IDX]
            tgt_keys = [_ROW_IDX]
        else:
            join_keys = list(keys)
            tgt_keys = [tgt_name(k) for k in keys]
            src, tgt, self._key_casts = _harmonize_key_dtypes(
                src, tgt, join_keys, tgt_keys, src_schema, tgt_schema
            )

        self._join_keys = join_keys
        self._tgt_keys = tgt_keys
        self._src = src
        self._tgt = tgt

        # Rows with null keys can't be matched; keep them out of the join
        n_null_src = n_null_tgt = 0
        if self.mode == "keyed":
            src_null = _any_null(join_keys)
            tgt_null = _any_null(tgt_keys)
            n_null_src, n_null_tgt = _count(src.filter(src_null)), _count(tgt.filter(tgt_null))
            if n_null_src:
                src = src.filter(~src_null)
            if n_null_tgt:
                tgt = tgt.filter(~tgt_null)
        self._n_null_src = n_null_src
        self._n_null_tgt = n_null_tgt

        # Duplicated keys can't be aligned; count them and keep them out of the join
        self._dup_frame = None
        n_dup_keys = n_dup_units = n_dup_src_rows = n_dup_tgt_rows = 0
        if self.mode == "keyed":
            dup = _dup_key_frame(src, tgt, join_keys, tgt_keys)
            stats = _collect_row(
                dup.select(
                    nw.len().alias("n_keys"),
                    nw.max_horizontal(nw.col(_N_SOURCE), nw.col(_N_TARGET)).sum().alias("units"),
                    nw.col(_N_SOURCE).sum().alias("n_s"),
                    nw.col(_N_TARGET).sum().alias("n_t"),
                )
            )
            n_dup_keys = int(stats["n_keys"] or 0)
            if n_dup_keys:
                n_dup_units = int(stats["units"] or 0)
                n_dup_src_rows = int(stats["n_s"] or 0)
                n_dup_tgt_rows = int(stats["n_t"] or 0)
                dup_keys_only = dup.select(join_keys)
                src = src.join(dup_keys_only, on=join_keys, how="anti")
                tgt = tgt.join(
                    dup_keys_only.rename(dict(zip(join_keys, tgt_keys))), on=tgt_keys, how="anti"
                )
                self._dup_frame = dup
        self._src_main = src
        self._tgt_main = tgt
        self.n_dup_key_values = n_dup_keys

        # The main full outer join and per-row status
        joined = _join_frames(src, tgt, join_keys, tgt_keys)
        if isinstance(joined, nw.DataFrame) and len(joined) == 0:
            # Nothing to classify (literal broadcasting also fails on empty Pandas frames), so add
            # empty flag/status columns without literals
            joined = joined.with_columns(
                *[
                    nw.col(_IN_SOURCE).cast(nw.Boolean).alias(_flag_name(i))
                    for i in range(len(specs))
                ],
                nw.col(_IN_SOURCE).cast(nw.String).alias(_STATUS),
            )
            self._joined = joined
            self._set_counts(
                {}, specs, n_null_src, n_null_tgt, n_dup_units, n_dup_src_rows, n_dup_tgt_rows
            )
            return self
        flag_exprs = {
            _flag_name(i): _differs_expr(spec, null_equal) for i, spec in enumerate(specs)
        }
        joined = joined.with_columns(**flag_exprs) if flag_exprs else joined
        flag_names = list(flag_exprs)
        any_differs = (
            nw.any_horizontal(*[nw.col(f) for f in flag_names], ignore_nulls=True)
            if flag_names
            else nw.lit(False)
        )
        status = (
            nw.when(nw.col(_IN_SOURCE).is_null())
            .then(nw.lit("extra"))
            .otherwise(
                nw.when(nw.col(_IN_TARGET).is_null())
                .then(nw.lit("missing"))
                .otherwise(nw.when(any_differs).then(nw.lit("changed")).otherwise(nw.lit("match")))
            )
        )
        joined = joined.with_columns(status.alias(_STATUS))
        self._joined = joined

        # One aggregation pass for all counts
        both = ~nw.col(_IN_SOURCE).is_null() & ~nw.col(_IN_TARGET).is_null()
        aggs = [
            (nw.col(_STATUS) == nw.lit(st)).cast(nw.Int64).sum().alias(st)
            for st in ("match", "changed", "missing", "extra")
        ]
        aggs += [(nw.col(f) & both).cast(nw.Int64).sum().alias(f) for f in flag_names]
        counts = _collect_row(joined.select(*aggs))
        self._set_counts(
            counts, specs, n_null_src, n_null_tgt, n_dup_units, n_dup_src_rows, n_dup_tgt_rows
        )
        return self

    def _set_counts(
        self,
        counts: dict[str, Any],
        specs: list[_ColumnSpec],
        n_null_src: int,
        n_null_tgt: int,
        n_dup_units: int,
        n_dup_src_rows: int,
        n_dup_tgt_rows: int,
    ) -> None:
        """Set the status counts and row counts from the joined frame's aggregation."""

        def get(name: str) -> int:
            return int(counts.get(name) or 0)

        self.n_match = get("match")
        self.n_changed = get("changed")
        self.n_missing = get("missing") + n_null_src
        self.n_extra = get("extra") + n_null_tgt
        self.n_dup_key = n_dup_units
        self.column_mismatches = {spec.source: get(_flag_name(i)) for i, spec in enumerate(specs)}

        n_src_joined = self.n_match + self.n_changed + get("missing")
        n_tgt_joined = self.n_match + self.n_changed + get("extra")
        self.n_source_rows = n_src_joined + n_null_src + n_dup_src_rows
        self.n_target_rows = n_tgt_joined + n_null_tgt + n_dup_tgt_rows

    # ------------------------------------------------------------------------------------------
    # Summary properties
    # ------------------------------------------------------------------------------------------

    @property
    def n(self) -> int:
        """The number of test units (aligned rows)."""
        return self.n_match + self.n_changed + self.n_missing + self.n_extra + self.n_dup_key

    @property
    def schema_ok(self) -> bool:
        """`True` if the schemas match, or if schema differences are tolerated (`"common"`)."""
        return self.schema_mode == "common" or self.schema_diff.matches

    @property
    def n_passed(self) -> int:
        """Test units that pass: matching rows (none, if the schema check fails)."""
        return self.n_match if self.schema_ok else 0

    @property
    def n_failed(self) -> int:
        """Test units that fail: all non-matching rows (all rows, if the schema check fails)."""
        return self.n - self.n_passed

    @property
    def identical(self) -> bool:
        """`True` if the tables match under the comparison settings."""
        return self.schema_ok and self.n_failed == 0

    @property
    def status_counts(self) -> dict[str, int]:
        """The number of test units with each row status."""
        return {
            "match": self.n_match,
            "changed": self.n_changed,
            "missing": self.n_missing,
            "extra": self.n_extra,
            "dup_key": self.n_dup_key,
        }

    def __bool__(self) -> bool:
        return self.identical

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}={v:,}" for k, v in self.status_counts.items())
        schema = "ok" if self.schema_diff.matches else "differs"
        return (
            f"Comparison({self.source_name!r} vs {self.target_name!r}: {counts}; "
            f"schema {schema}; identical={self.identical})"
        )

    def _repr_html_(self) -> str:
        return self.get_tabular_report().as_raw_html()

    # ------------------------------------------------------------------------------------------
    # Row samples
    # ------------------------------------------------------------------------------------------

    def rows(
        self, status: Literal["changed", "missing", "extra", "dup_key", "match"], limit: int = 100
    ) -> Any:
        """
        Get a sample of rows with a given status.

        Only up to `limit=` rows are collected, so this is safe to use on very large tables.

        Parameters
        ----------
        status
            The row status to retrieve:

            - `"changed"`: the keys, then a `<column>_source` and a `<column>_target` column for
              each compared column
            - `"missing"`: source rows not in the target (source column names)
            - `"extra"`: target rows not in the source (target column names)
            - `"dup_key"`: duplicated key values with their occurrence counts in each table
              (`n_source`, `n_target`)
            - `"match"`: the keys of matching rows
        limit
            The maximum number of rows to return.

        Returns
        -------
        Any
            A table of the same type as the (collected) input tables. In positional mode, rows
            are identified by a 1-based `_row_num_` column.
        """
        if status not in ROW_STATUSES:
            raise ValueError(f"`status=` must be one of {ROW_STATUSES}, not {status!r}.")
        cache_key = (status, limit)
        if cache_key not in self._sample_cache:
            self._sample_cache[cache_key] = _to_native_eager(self._fetch(status, limit))
        return self._sample_cache[cache_key]

    def _key_out_names(self) -> list[str]:
        if self.mode == "multiset":
            return []
        return [_ROW_NUM_COL] if self.mode == "positional" else list(self.keys)

    def _key_out_exprs(self, from_target: bool = False) -> list[Any]:
        """Expressions producing the user-facing key column(s) from the joined frame."""
        if self.mode == "positional":
            col = _ROW_IDX + _TGT_SUFFIX if from_target else _ROW_IDX
            return [(nw.col(col) + 1).alias(_ROW_NUM_COL)]
        out = []
        for k, tk in zip(self._join_keys, self._tgt_keys):
            if from_target:
                out.append(nw.col(tk + _TGT_SUFFIX).alias(k))
            else:
                out.append(nw.col(k))
        return out

    def _fetch(self, status: str, limit: int) -> Any:
        """Fetch a sample of rows with a status as an eager narwhals DataFrame."""
        if self._parts is not None:
            return self._collect_from_parts(status, limit, lambda part, n: part._fetch(status, n))
        if self.mode == "multiset":
            return self._fetch_multiset(status, limit)
        joined = self._joined
        filtered = joined.filter(nw.col(_STATUS) == nw.lit(status))

        if status == "dup_key":
            if self._dup_frame is None:
                frame = self._src_main.select(self._join_keys).head(0)
                frame = frame.with_columns(
                    nw.lit(0).cast(nw.Int64).alias("n_source"),
                    nw.lit(0).cast(nw.Int64).alias("n_target"),
                )
                return _collect(frame)
            frame = self._dup_frame.select(
                *self._join_keys,
                nw.col(_N_SOURCE).cast(nw.Int64).alias("n_source"),
                nw.col(_N_TARGET).cast(nw.Int64).alias("n_target"),
            )
            return _collect(frame.head(limit))

        if status == "changed":
            out = self._key_out_exprs()
            for spec in self._specs:
                out.append(nw.col(spec.source).alias(f"{spec.source}_source"))
                out.append(nw.col(spec.target + _TGT_SUFFIX).alias(f"{spec.source}_target"))
            out += [nw.col(_flag_name(i)) for i in range(len(self._specs))]
            return _collect(filtered.select(*out).head(limit))

        if status == "match":
            return _collect(filtered.select(*self._key_out_exprs()).head(limit))

        if status == "missing":
            non_keys = [c for c in self._src_columns if c not in self.keys]
            out = self._key_out_exprs() + [nw.col(c) for c in non_keys]
            frame = _collect(filtered.select(*out).head(limit))
            if self._n_null_src and len(frame) < limit:
                extra = _collect(
                    self._src.filter(_any_null(self._join_keys))
                    .select(*[nw.col(k) for k in self.keys], *[nw.col(c) for c in non_keys])
                    .head(limit - len(frame))
                )
                frame = _concat_eager(frame, extra)
            return frame

        # status == "extra"
        tgt_key_names = set(self._tgt_keys)
        non_keys = [c for c in self._tgt_columns if c not in tgt_key_names]
        out = self._key_out_exprs(from_target=True) + [
            nw.col(c + _TGT_SUFFIX).alias(c) for c in non_keys
        ]
        frame = _collect(filtered.select(*out).head(limit))
        if self._n_null_tgt and len(frame) < limit:
            extra = _collect(
                self._tgt.filter(_any_null(self._tgt_keys))
                .select(
                    *[nw.col(tk).alias(k) for k, tk in zip(self.keys, self._tgt_keys)],
                    *[nw.col(c) for c in non_keys],
                )
                .head(limit - len(frame))
            )
            frame = _concat_eager(frame, extra)
        return frame

    def _dup_pairs(self, limit: int) -> Any:
        """
        Pairwise comparison of rows sharing a duplicated key (for `dup_keys="compare"`).

        Returns an eager narwhals DataFrame with the key columns and one difference flag per
        compared column, for every source/target row pair of up to `limit` duplicated keys.
        """
        if self._parts is not None:
            pairs = None
            for part in self._parts:
                piece = part._dup_pairs(limit)
                if piece is not None:
                    pairs = piece if pairs is None else _concat_eager(pairs, piece)
            return pairs
        if self._dup_frame is None:
            return None
        sample_keys = _collect(self._dup_frame.select(self._join_keys).head(limit))
        # Joining an eager sample back onto a lazy table isn't possible on every backend, so
        # filter with `is_in` per key column instead (bounded by `limit` key values). This can
        # over-select for composite keys; the inner join below restores exact key matching.
        src_f = self._src
        tgt_f = self._tgt
        for k, tk in zip(self._join_keys, self._tgt_keys):
            values = sample_keys[k].to_list()
            src_f = src_f.filter(nw.col(k).is_in(values))
            tgt_f = tgt_f.filter(nw.col(tk).is_in(values))
        pairs = _join_frames(src_f, tgt_f, self._join_keys, self._tgt_keys, how="inner")
        flags = {
            _flag_name(i): _differs_expr(spec, self.null_equal)
            for i, spec in enumerate(self._specs)
        }
        if flags:
            pairs = pairs.with_columns(**flags)
        pairs = _collect(pairs.select(*self._join_keys, *list(flags)))
        if len(self._join_keys) > 1:
            pairs = pairs.join(sample_keys, on=self._join_keys, how="semi")
        return pairs

    # ------------------------------------------------------------------------------------------
    # Partitioned comparison (`partitions=`)
    # ------------------------------------------------------------------------------------------

    @classmethod
    def _build_partitioned(cls, src: Any, tgt: Any, partitions: int, **settings: Any) -> Comparison:
        """
        Compare in `partitions` passes, each restricted (on both sides) to the rows whose key
        hashes into one bucket, then combine the per-partition results. A key value always lands
        in the same bucket on both sides, so every partition is a self-contained comparison.
        """
        keys = settings["keys"]
        column_map = settings["column_map"]
        tgt_keys = [column_map.get(k, k) for k in keys]
        src_schema = dict(src.collect_schema().items())
        tgt_schema = dict(tgt.collect_schema().items())
        for k, tk in zip(keys, tgt_keys):
            if k not in src_schema:
                raise ValueError(f"Key column {k!r} isn't in the source table.")
            if tk not in tgt_schema:
                raise ValueError(
                    f"Key column {tk!r} isn't in the target table. If it was renamed, map it "
                    "with `column_map=`."
                )

        # Keys must hash identically on both sides, so harmonize their types first
        src, tgt, _ = _harmonize_key_dtypes(src, tgt, keys, tgt_keys, src_schema, tgt_schema)

        parts = [
            cls._build(
                src=_bucket_filter(src, keys, partitions, i),
                tgt=_bucket_filter(tgt, tgt_keys, partitions, i),
                **settings,
            )
            for i in range(partitions)
        ]

        self = object.__new__(cls)
        first = parts[0]
        for attr in (
            "source_name",
            "target_name",
            "keys",
            "mode",
            "schema_mode",
            "dup_keys_mode",
            "null_equal",
            "column_map",
            "schema_diff",
            "compared_columns",
            "_specs",
            "_src_columns",
            "_tgt_columns",
            "_join_keys",
            "_tgt_keys",
            "_order_by",
        ):
            setattr(self, attr, getattr(first, attr))
        for attr in (
            "n_match",
            "n_changed",
            "n_missing",
            "n_extra",
            "n_dup_key",
            "n_dup_key_values",
            "n_source_rows",
            "n_target_rows",
        ):
            setattr(self, attr, sum(getattr(part, attr) for part in parts))
        self.column_mismatches = {
            col: sum(part.column_mismatches[col] for part in parts)
            for col in first.column_mismatches
        }
        self._parts = parts
        self.partitions = partitions
        self._tgt_input = None
        self._key_casts = {}
        self._sample_cache = {}
        return self

    def _collect_from_parts(self, status: str, limit: int, fetch: Any) -> Any:
        """Gather up to `limit` rows from the partitions, in partition order."""
        assert self._parts is not None
        count_attr = {"dup_key": "n_dup_key_values", "match": "n_match"}.get(status, f"n_{status}")
        frame = None
        for part in self._parts:
            remaining = limit - (0 if frame is None else len(frame))
            if remaining <= 0:
                break
            if frame is not None and getattr(part, count_attr, 0) == 0:
                continue
            piece = fetch(part, remaining)
            frame = piece if frame is None else _concat_eager(frame, piece)
        return frame

    # ------------------------------------------------------------------------------------------
    # Multiset mode (`keys="*"`)
    # ------------------------------------------------------------------------------------------

    def _multiset_values(self, frame: Any, side: str) -> Any:
        """Each compared column as a normalized string (nulls as a sentinel), for grouping."""
        exprs = []
        for i, spec in enumerate(self._specs):
            name = spec.source if side == "source" else spec.target
            dtype = spec.source_dtype if side == "source" else spec.target_dtype
            expr = nw.col(name)
            if _is_float(dtype):
                expr = expr.fill_nan(None)
            if (
                _is_numeric(spec.source_dtype)
                and _is_numeric(spec.target_dtype)
                and str(spec.source_dtype) != str(spec.target_dtype)
            ):
                # e.g., 1 (integer) and 1.0 (float) should be the same value
                expr = expr.cast(nw.Float64)
            expr = _normalized(expr, spec.normalize)
            exprs.append(expr.cast(nw.String).fill_null(_NULL_SENTINEL).alias(f"__pb_m_{i}"))
        return frame.select(*exprs)

    def _build_multiset(self, src: Any, tgt: Any) -> None:
        """
        Compare the tables as multisets of rows: each distinct row (over the compared columns)
        matches as many times as it occurs in both tables; surplus copies are missing or extra.
        """
        self._join_keys, self._tgt_keys = [], []
        self._src, self._tgt = src, tgt
        self._src_main, self._tgt_main = src, tgt
        self._n_null_src = self._n_null_tgt = 0
        self._dup_frame = None
        self.n_dup_key_values = 0
        self.n_dup_key = 0
        self.n_changed = 0
        self.column_mismatches = {}

        n_s, n_t = nw.col(_N_SOURCE), nw.col(_N_TARGET)
        if not self._specs:
            # No shared columns: rows can only be counted
            n_src, n_tgt = _count(src), _count(tgt)
            self.n_match = min(n_src, n_tgt)
            self.n_missing = max(n_src - n_tgt, 0)
            self.n_extra = max(n_tgt - n_src, 0)
            self.n_source_rows, self.n_target_rows = n_src, n_tgt
            self._joined = None
            return

        m_cols = [f"__pb_m_{i}" for i in range(len(self._specs))]
        gs = self._multiset_values(src, "source").group_by(m_cols).agg(nw.len().alias(_N_SOURCE))
        gt = (
            self._multiset_values(tgt, "target")
            .group_by(m_cols)
            .agg(nw.len().alias(_N_TARGET))
            .rename({c: c + _TGT_SUFFIX for c in m_cols})
        )
        joined = gs.join(gt, left_on=m_cols, right_on=[c + _TGT_SUFFIX for c in m_cols], how="full")
        joined = joined.with_columns(
            *[nw.coalesce(nw.col(c), nw.col(c + _TGT_SUFFIX)).alias(c) for c in m_cols],
            n_s.fill_null(0).cast(nw.Int64),
            n_t.fill_null(0).cast(nw.Int64),
        ).select(*m_cols, _N_SOURCE, _N_TARGET)
        self._joined = joined

        zero = nw.lit(0).cast(nw.Int64)
        counts = _collect_row(
            joined.select(
                nw.min_horizontal(n_s, n_t).sum().alias("match"),
                nw.when(n_s > n_t).then(n_s - n_t).otherwise(zero).sum().alias("missing"),
                nw.when(n_t > n_s).then(n_t - n_s).otherwise(zero).sum().alias("extra"),
                n_s.sum().alias("n_s"),
                n_t.sum().alias("n_t"),
            )
        )
        self.n_match = int(counts["match"] or 0)
        self.n_missing = int(counts["missing"] or 0)
        self.n_extra = int(counts["extra"] or 0)
        self.n_source_rows = int(counts["n_s"] or 0)
        self.n_target_rows = int(counts["n_t"] or 0)

    def _fetch_multiset(self, status: str, limit: int) -> Any:
        """Samples in multiset mode: distinct rows (as strings) with their occurrence counts."""
        names = [spec.source if status != "extra" else spec.target for spec in self._specs]
        n_s, n_t = nw.col(_N_SOURCE), nw.col(_N_TARGET)
        if self._joined is None:
            return _records_frame_like([], names + ["n_source", "n_target"])
        cond = {
            "missing": n_s > n_t,
            "extra": n_t > n_s,
            "match": (n_s > 0) & (n_t > 0),
        }.get(status, nw.lit(False))
        out = [
            nw.when(nw.col(f"__pb_m_{i}") == nw.lit(_NULL_SENTINEL))
            .then(nw.lit(None, dtype=nw.String))
            .otherwise(nw.col(f"__pb_m_{i}"))
            .alias(name)
            for i, name in enumerate(names)
        ]
        out += [n_s.alias("n_source"), n_t.alias("n_target")]
        return _collect(self._joined.filter(cond).select(*out).head(limit))

    # ------------------------------------------------------------------------------------------
    # Support for `Validate.tbl_match()`: data extracts and sundering
    # ------------------------------------------------------------------------------------------

    def _failing_rows(self, limit: int) -> Any:
        """
        All failing test units in one table (the `tbl_match()` data extract): the key column(s),
        a `_status_` column, then `<column>_source`/`<column>_target` pairs for each compared
        column. In multiset mode, the compared columns (as strings) and their occurrence counts.
        """
        if self._parts is not None:
            frame = None
            for part in self._parts:
                remaining = limit - (0 if frame is None else len(frame))
                if remaining <= 0:
                    break
                piece = nw.from_native(part._failing_rows(remaining))
                frame = piece if frame is None else _concat_eager(frame, piece)
            return frame.to_native()
        if self.mode == "multiset":
            parts = []
            for status in ("missing", "extra"):
                frame = self._fetch_multiset(status, limit)
                if status == "extra":
                    frame = frame.rename(
                        {s.target: s.source for s in self._specs if s.target != s.source}
                    )
                parts.append(frame.with_columns(nw.lit(status).alias(_STATUS_OUT_COL)))
            frame = _concat_eager(parts[0], parts[1])
            cols = [_STATUS_OUT_COL] + [c for c in frame.columns if c != _STATUS_OUT_COL]
            return frame.select(cols).head(limit).to_native()

        pair_exprs = []
        for spec in self._specs:
            pair_exprs.append(nw.col(spec.source).alias(f"{spec.source}_source"))
            pair_exprs.append(nw.col(spec.target + _TGT_SUFFIX).alias(f"{spec.source}_target"))

        main = _collect(
            self._joined.filter(nw.col(_STATUS) != nw.lit("match"))
            .select(
                *self._key_out_exprs(),
                nw.col(_STATUS).alias(_STATUS_OUT_COL),
                *pair_exprs,
            )
            .head(limit)
        )
        columns = main.columns
        schema = dict(main.schema.items())

        def null_of(col: str) -> Any:
            return nw.lit(None).cast(schema[col]).alias(col)

        # Rows that never entered the join: null keys (missing/extra) and duplicated keys
        extras = []
        if self._n_null_src:
            extras.append(
                self._src.filter(_any_null(self._join_keys)).select(
                    *[nw.col(k).cast(schema[k]) for k in self.keys],
                    nw.lit("missing").alias(_STATUS_OUT_COL),
                    *[
                        nw.col(spec.source).alias(c).cast(schema[c])
                        if c.endswith("_source")
                        else null_of(c)
                        for spec in self._specs
                        for c in (f"{spec.source}_source", f"{spec.source}_target")
                    ],
                )
            )
        if self._n_null_tgt:
            extras.append(
                self._tgt.filter(_any_null(self._tgt_keys)).select(
                    *[
                        nw.col(tk).cast(schema[k]).alias(k)
                        for k, tk in zip(self.keys, self._tgt_keys)
                    ],
                    nw.lit("extra").alias(_STATUS_OUT_COL),
                    *[
                        nw.col(spec.target).alias(c).cast(schema[c])
                        if c.endswith("_target")
                        else null_of(c)
                        for spec in self._specs
                        for c in (f"{spec.source}_source", f"{spec.source}_target")
                    ],
                )
            )
        if self._dup_frame is not None:
            extras.append(
                self._dup_frame.select(
                    *[nw.col(k).cast(schema[k]) for k in self.keys],
                    nw.lit("dup_key").alias(_STATUS_OUT_COL),
                    *[null_of(c) for c in columns[len(self.keys) + 1 :]],
                )
            )

        frame = main
        for extra in extras:
            if len(frame) >= limit:
                break
            frame = _concat_eager(frame, _collect(extra.select(columns).head(limit - len(frame))))
        return frame.to_native()

    def _target_pass_table(self) -> Any:
        """
        The target table (in its original row order) with a boolean `pb_is_good_` column: `True`
        for rows whose status is `"match"`. Used by `Validate.get_sundered_data()`. Returns `None`
        when this isn't possible (lazy inputs, or multiset mode where rows aren't identified).
        """
        tgt = self._tgt_input
        if (
            self.mode == "multiset"
            or not isinstance(tgt, nw.DataFrame)
            or not isinstance(self._joined, nw.DataFrame)
        ):
            return None

        columns = tgt.columns
        orig = "__pb_tgt_row"
        tgt = tgt.with_row_index(orig)
        temp_keys = [f"__pb_k_{i}" for i in range(len(self._tgt_keys))]
        if self.mode == "positional":
            left = _with_row_index(tgt, self._order_by, "target", name=temp_keys[0])
        else:
            left = tgt.with_columns(
                *[
                    (
                        nw.col(tk).cast(self._key_casts[tk])
                        if tk in self._key_casts
                        else nw.col(tk)
                    ).alias(tmp)
                    for tk, tmp in zip(self._tgt_keys, temp_keys)
                ]
            )
        matched = self._joined.filter(nw.col(_STATUS) == nw.lit("match")).select(
            *[nw.col(tk + _TGT_SUFFIX).alias(tmp) for tk, tmp in zip(self._tgt_keys, temp_keys)]
        )
        if not self.schema_ok or len(matched) == 0:
            # No row passes (and empty frames can't take a literal column on every backend)
            return tgt.select(*columns).with_columns(nw.lit(False).alias("pb_is_good_")).to_native()
        out = (
            left.join(
                matched.with_columns(nw.lit(True).alias("pb_is_good_")), on=temp_keys, how="left"
            )
            .with_columns(nw.col("pb_is_good_").fill_null(False))
            .sort(orig)
            .select(*columns, "pb_is_good_")
        )
        return out.to_native()

    # ------------------------------------------------------------------------------------------
    # Summaries and export
    # ------------------------------------------------------------------------------------------

    def column_summary(self) -> Any:
        """
        Get a per-column summary of value mismatches.

        Returns
        -------
        Any
            A table (Polars if available, else Pandas) with one row per compared column: the
            source and target column names and data types, the number of matched rows whose
            values differ (`n_mismatch`), and that number as a fraction of matched rows
            (`f_mismatch`).
        """
        n_matched = self.n_match + self.n_changed
        records = [
            {
                "column": spec.source,
                "target_column": spec.target,
                "dtype_source": str(spec.source_dtype),
                "dtype_target": str(spec.target_dtype),
                "n_mismatch": self.column_mismatches.get(spec.source),
                "f_mismatch": (
                    self.column_mismatches[spec.source] / n_matched
                    if n_matched and spec.source in self.column_mismatches
                    else None
                ),
            }
            for spec in self._specs
        ]
        return _records_to_frame(
            records,
            columns=[
                "column",
                "target_column",
                "dtype_source",
                "dtype_target",
                "n_mismatch",
                "f_mismatch",
            ],
        )

    def to_dict(self, samples: bool = True, limit: int = 10) -> dict[str, Any]:
        """
        Get the comparison results as a JSON-serializable dict.

        Parameters
        ----------
        samples
            Should samples of differing rows be included? Set to `False` when the data might
            contain sensitive values (e.g., PII) and only counts and schema details may be stored.
        limit
            The maximum number of sample rows per status.
        """
        out: dict[str, Any] = {
            "source_name": self.source_name,
            "target_name": self.target_name,
            "mode": self.mode,
            "keys": list(self.keys),
            "compared_columns": list(self.compared_columns),
            "column_map": dict(self.column_map),
            "schema_mode": self.schema_mode,
            "null_equal": self.null_equal,
            "partitions": self.partitions,
            "n_source_rows": self.n_source_rows,
            "n_target_rows": self.n_target_rows,
            "n": self.n,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "identical": self.identical,
            "status_counts": self.status_counts,
            "column_mismatches": dict(self.column_mismatches),
            "schema_diff": self.schema_diff.to_dict(),
        }
        if samples:
            sample_out: dict[str, list[dict[str, Any]]] = {}
            for status in ("changed", "missing", "extra", "dup_key"):
                if self.status_counts[status]:
                    frame = self._fetch(status, limit)
                    keep = [c for c in frame.columns if not c.startswith("__pb_")]
                    sample_out[status] = [
                        {k: _jsonable(v) for k, v in row.items()}
                        for row in frame.select(keep).rows(named=True)
                    ]
            out["samples"] = sample_out
        return out

    def to_json(self, samples: bool = True, limit: int = 10, indent: int | None = 2) -> str:
        """Get the comparison results as a JSON string (see `to_dict()`)."""
        return json.dumps(self.to_dict(samples=samples, limit=limit), indent=indent)

    def get_tabular_report(self, limit: int = 10, title: str | None = None) -> GT:
        """
        Get a tabular report of the comparison.

        The report has a header with the verdict, the row status counts, and the comparison
        settings, followed by sections (as row groups) for schema differences, per-column
        mismatches, and samples of changed, missing, extra, and duplicate-key rows. Sections with
        nothing to show are omitted.

        Parameters
        ----------
        limit
            The maximum number of sample rows shown per row status.
        title
            An optional title replacing the default.

        Returns
        -------
        GT
            A Great Tables object.
        """
        from pointblank._compare_report import _comparison_report

        return _comparison_report(self, limit=limit, title=title)


# ----------------------------------------------------------------------------------------------
# Engine helpers
# ----------------------------------------------------------------------------------------------


def _as_list(x: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def _flag_name(i: int) -> str:
    return f"__pb_d_{i}"


def _is_numeric(dtype: Any) -> bool:
    return bool(dtype.is_numeric())


def _is_float(dtype: Any) -> bool:
    return isinstance(dtype, (nw.Float32, nw.Float64)) or dtype in (nw.Float32, nw.Float64)


def _is_temporal(dtype: Any) -> bool:
    return bool(dtype.is_temporal())


def _any_null(cols: list[str]) -> Any:
    return nw.any_horizontal(*[nw.col(c).is_null() for c in cols], ignore_nulls=False)


def _collect(frame: Any) -> Any:
    """
    Collect a narwhals frame into an eager narwhals DataFrame (Polars if available, else Pandas,
    for lazy backends like DuckDB and Ibis).
    """
    if not isinstance(frame, nw.LazyFrame):
        return frame
    if frame.implementation == nw.Implementation.POLARS:
        # The streaming engine processes data in batches, keeping memory use bounded for large
        # (e.g., scanned CSV/Parquet) inputs
        return frame.collect(engine="streaming")
    from pointblank._utils import _is_lib_present

    if _is_lib_present("polars"):
        return frame.collect(backend="polars")
    if _is_lib_present("pandas"):  # pragma: no cover
        return frame.collect(backend="pandas")
    return frame.collect()  # pragma: no cover


def _collect_row(frame: Any) -> dict[str, Any]:
    """Collect a single-row aggregation into a dict."""
    rows = _collect(frame).rows(named=True)
    return dict(rows[0]) if rows else {}


def _count(frame: Any) -> int:
    return int(_collect_row(frame.select(nw.len().alias("n"))).get("n") or 0)


def _to_native_eager(frame: Any) -> Any:
    keep = [c for c in frame.columns if not c.startswith("__pb_")]
    return frame.select(keep).to_native()


def _concat_eager(a: Any, b: Any) -> Any:
    if len(b) == 0:
        return a
    if len(a) == 0:
        return b
    try:
        return nw.concat([a, b], how="vertical")
    except Exception:  # pragma: no cover
        # Fall back when dtypes differ (e.g., all-null columns in one part)
        return a


def _records_frame_like(records: list[dict[str, Any]], columns: list[str]) -> Any:
    """An eager narwhals DataFrame (all-string columns when empty) for sample output."""
    return nw.from_native(_records_to_frame(records, columns))


def _records_to_frame(records: list[dict[str, Any]], columns: list[str]) -> Any:
    from pointblank._utils import _is_lib_present

    if _is_lib_present("polars"):
        import polars as pl

        if not records:
            return pl.DataFrame({c: [] for c in columns})
        return pl.DataFrame(records).select(columns)
    import pandas as pd  # pragma: no cover

    return pd.DataFrame(records, columns=columns)  # pragma: no cover


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    return str(value)


def _align_backends(src: Any, tgt: Any) -> tuple[Any, Any]:
    """
    Bring two narwhals frames onto a common backend so they can be joined.

    Frames on the same backend are left as they are, so database tables stay in the database and
    the comparison is pushed down. A Polars DataFrame paired with a Polars LazyFrame is made lazy.
    Otherwise both frames are converted to Polars if available (DataFrames if both inputs are
    in-memory, else LazyFrames), or to Pandas.
    """
    s_impl = src.implementation
    t_impl = tgt.implementation
    s_lazy = isinstance(src, nw.LazyFrame)
    t_lazy = isinstance(tgt, nw.LazyFrame)

    if s_impl == t_impl and _same_connection(src, tgt):
        if s_lazy != t_lazy:
            return src.lazy(), tgt.lazy()
        return src, tgt

    from pointblank._utils import _is_lib_present

    if _is_lib_present("polars"):
        if not s_lazy and not t_lazy:
            # Two in-memory tables stay in memory (as Polars DataFrames)
            return _to_polars_lazy(src).collect(), _to_polars_lazy(tgt).collect()
        return _to_polars_lazy(src), _to_polars_lazy(tgt)
    return _to_pandas(src), _to_pandas(tgt)  # pragma: no cover


def _same_connection(src: Any, tgt: Any) -> bool:
    """
    For Ibis tables, are both tables in the same database connection (so they can be joined in
    place)? Always `True` for other backends.
    """
    if src.implementation != nw.Implementation.IBIS:
        return True
    try:
        return src.to_native().get_backend() is tgt.to_native().get_backend()
    except Exception:  # pragma: no cover
        return False


def _to_polars_lazy(frame: Any) -> Any:
    import polars as pl

    if frame.implementation == nw.Implementation.POLARS:
        return frame.lazy()
    eager = _collect(frame)
    native = eager.to_native()
    if eager.implementation == nw.Implementation.PANDAS:
        return nw.from_native(pl.from_pandas(native)).lazy()
    return nw.from_native(pl.from_arrow(eager.to_arrow())).lazy()


def _to_pandas(frame: Any) -> Any:  # pragma: no cover
    return nw.from_native(_collect(frame).to_pandas())


def _with_row_index(frame: Any, order_by: list[str] | None, side: str, name: str = _ROW_IDX) -> Any:
    """
    Add a 0-based row index for positional alignment. With `order_by=`, the index follows that
    order but the rows themselves keep their original order.
    """
    if isinstance(frame, nw.DataFrame):
        if not order_by:
            return frame.with_row_index(name)
        # (`with_row_index(order_by=)` on eager frames is unreliable, so sort explicitly)
        orig = "__pb_orig_idx"
        return (
            frame.with_row_index(orig)
            .sort([*order_by, orig])
            .with_row_index(name)
            .sort(orig)
            .drop(orig)
        )
    if order_by:
        return frame.with_row_index(name, order_by=order_by)
    if frame.implementation == nw.Implementation.POLARS:
        # Polars LazyFrames (e.g., from `scan_csv()`) preserve row order
        return nw.from_native(frame.to_native().with_row_index(name))
    raise ValueError(
        f"The {side} table has no defined row order, so rows can't be aligned by position. "
        "Provide `keys=` to align rows by key (recommended), or `order_by=` to define the row "
        "order."
    )


def _bucket_filter(frame: Any, keys: list[str], partitions: int, i: int) -> Any:
    """
    Keep the rows whose key hashes into bucket `i` of `partitions`. The bucket is
    `sum(hash(key) mod partitions) mod partitions` over the key columns, computed with the
    backend's own hash function (so equal key values always share a bucket on the same backend).
    Rows with null keys land in a bucket too, so they're still counted.
    """
    impl = frame.implementation
    native = frame.to_native()

    if impl == nw.Implementation.POLARS:
        import polars as pl

        bucket = pl.sum_horizontal([pl.col(k).hash(seed=0) % partitions for k in keys]) % partitions
        return nw.from_native(native.filter(bucket == i))

    if impl == nw.Implementation.IBIS:
        terms = [((native[k].hash() % partitions) + partitions) % partitions for k in keys]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return nw.from_native(native.filter((total % partitions) == i))

    if impl == nw.Implementation.DUCKDB:
        terms = " + ".join(
            '(hash("' + k.replace('"', '""') + '") % ' + str(partitions) + ")" for k in keys
        )
        return nw.from_native(native.filter(f"(({terms}) % {partitions}) = {i}"))

    raise ValueError(
        "`partitions=` is supported for Polars, DuckDB, and Ibis tables (and CSV/Parquet file "
        f"paths), not {impl}."
    )


def _harmonize_key_dtypes(
    src: Any,
    tgt: Any,
    keys: list[str],
    tgt_keys: list[str],
    src_schema: dict[str, Any],
    tgt_schema: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """
    Cast key columns to a common type where the two tables disagree (e.g., integer keys that
    became floats in Pandas because of a null), so that they can be joined. Also returns the
    casts applied (target key name -> dtype).
    """
    src_casts, tgt_casts = [], []
    casts: dict[str, Any] = {}
    for k, tk in zip(keys, tgt_keys):
        s_dtype, t_dtype = src_schema[k], tgt_schema[tk]
        if str(s_dtype) == str(t_dtype):
            continue
        common = nw.Float64 if (_is_numeric(s_dtype) and _is_numeric(t_dtype)) else nw.String
        src_casts.append(nw.col(k).cast(common))
        tgt_casts.append(nw.col(tk).cast(common))
        casts[tk] = common
    if src_casts:
        src = src.with_columns(*src_casts)
        tgt = tgt.with_columns(*tgt_casts)
    return src, tgt, casts


def _dup_key_frame(src: Any, tgt: Any, keys: list[str], tgt_keys: list[str]) -> Any:
    """Key values occurring more than once in either table, with per-table occurrence counts."""
    ks = src.group_by(keys).agg(nw.len().alias(_N_SOURCE))
    kt = (
        tgt.group_by(tgt_keys)
        .agg(nw.len().alias(_N_TARGET))
        .rename({tk: k + _TGT_SUFFIX for k, tk in zip(keys, tgt_keys)})
    )
    joined = ks.join(kt, left_on=keys, right_on=[k + _TGT_SUFFIX for k in keys], how="full")
    joined = joined.with_columns(
        *[nw.coalesce(nw.col(k), nw.col(k + _TGT_SUFFIX)).alias(k) for k in keys],
        nw.col(_N_SOURCE).fill_null(0).cast(nw.Int64),
        nw.col(_N_TARGET).fill_null(0).cast(nw.Int64),
    )
    return joined.filter((nw.col(_N_SOURCE) > 1) | (nw.col(_N_TARGET) > 1)).select(
        *keys, _N_SOURCE, _N_TARGET
    )


def _join_frames(
    src: Any, tgt: Any, keys: list[str], tgt_keys: list[str], how: str = "full"
) -> Any:
    """
    Join source and target; target columns get the `_TGT_SUFFIX` suffix. Presence-flag columns
    show which side each joined row came from. Key columns are coalesced into the source names.
    """
    src = src.with_columns(nw.lit(True).alias(_IN_SOURCE))
    tgt_cols = tgt.collect_schema().names()
    tgt = tgt.rename({c: c + _TGT_SUFFIX for c in tgt_cols}).with_columns(
        nw.lit(True).alias(_IN_TARGET)
    )
    right_on = [tk + _TGT_SUFFIX for tk in tgt_keys]
    joined = src.join(tgt, left_on=keys, right_on=right_on, how=how)  # type: ignore[arg-type]
    if how == "full":
        joined = joined.with_columns(
            *[nw.coalesce(nw.col(k), nw.col(r)).alias(k) for k, r in zip(keys, right_on)]
        )
    return joined


def _normalized(expr: Any, ops: tuple[tuple[str, Any], ...]) -> Any:
    for op, arg in ops:
        if op == "strip":
            expr = expr.str.strip_chars()
        elif op == "lower":
            expr = expr.str.to_lowercase()
        elif op == "upper":
            expr = expr.str.to_uppercase()
        elif op == "collapse_whitespace":
            expr = expr.str.replace_all(r"\s+", " ", literal=False)
        elif op == "round":
            expr = expr.round(int(arg))
    return expr


def _differs_expr(spec: _ColumnSpec, null_equal: bool) -> Any:
    """A boolean expression: do this column's values differ (for rows present on both sides)?"""
    a = nw.col(spec.source)
    b = nw.col(spec.target + _TGT_SUFFIX)

    # NaN is treated as null
    if _is_float(spec.source_dtype):
        a = a.fill_nan(None)
    if _is_float(spec.target_dtype):
        b = b.fill_nan(None)

    s_num, t_num = _is_numeric(spec.source_dtype), _is_numeric(spec.target_dtype)
    s_tmp, t_tmp = _is_temporal(spec.source_dtype), _is_temporal(spec.target_dtype)
    comparable = (s_num and t_num) or (s_tmp and t_tmp) or (spec.source_dtype == spec.target_dtype)
    if not comparable:
        # e.g., a column that's a string in one table and an integer in the other
        a = a.cast(nw.String)
        b = b.cast(nw.String)
        s_num = t_num = s_tmp = t_tmp = False

    a = _normalized(a, spec.normalize)
    b = _normalized(b, spec.normalize)

    equal = (a == b).fill_null(False)
    if spec.atol is not None or spec.rtol is not None:
        if s_num and t_num:
            bound = nw.lit(float(spec.atol or 0.0))
            if spec.rtol:
                bound = bound + nw.lit(float(spec.rtol)) * a.abs()
            equal = equal | ((a - b).abs() <= bound).fill_null(False)
        elif s_tmp and t_tmp and isinstance(spec.atol, datetime.timedelta):
            td = nw.lit(spec.atol)
            equal = equal | (((a - b) <= td) & ((b - a) <= td)).fill_null(False)
    if null_equal:
        equal = equal | (a.is_null() & b.is_null())
    return ~equal


def _schema_diff(
    src_schema: dict[str, Any],
    tgt_schema: dict[str, Any],
    column_map: dict[str, str],
    scope: list[str] | None,
) -> SchemaDiff:
    src_cols = [c for c in src_schema if scope is None or c in scope]
    mapped_tgt = {column_map.get(c, c) for c in src_cols}
    if scope is None:
        tgt_cols = list(tgt_schema)
    else:
        tgt_cols = [c for c in tgt_schema if c in mapped_tgt]

    only_s = [c for c in src_cols if column_map.get(c, c) not in tgt_schema]
    only_t = [c for c in tgt_cols if c not in mapped_tgt]

    dtype_changed = []
    for c in src_cols:
        t = column_map.get(c, c)
        if t in tgt_schema and str(src_schema[c]) != str(tgt_schema[t]):
            dtype_changed.append((c, str(src_schema[c]), str(tgt_schema[t])))

    renamed = [(s, t) for s, t in column_map.items() if s != t and (scope is None or s in scope)]

    case_only = []
    lower_t = {c.lower(): c for c in only_t}
    for c in only_s:
        if c.lower() in lower_t:
            case_only.append((c, lower_t[c.lower()]))

    shared_src_order = [
        column_map.get(c, c) for c in src_cols if column_map.get(c, c) in tgt_schema
    ]
    shared_tgt_order = [c for c in tgt_cols if c in set(shared_src_order)]
    reordered = shared_src_order != shared_tgt_order

    return SchemaDiff(
        only_in_source=only_s,
        only_in_target=only_t,
        dtype_changed=dtype_changed,
        renamed=renamed,
        case_only=case_only,
        reordered=reordered,
    )


def _parse_duration(value: Any) -> datetime.timedelta:
    if isinstance(value, datetime.timedelta):
        return value
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(us|ms|s|m|h|d|w)\s*", str(value))
    if not match:
        raise ValueError(
            f"Can't interpret {value!r} as a duration. Use a `datetime.timedelta` or a string "
            "like '1s', '500ms', '2m', '1h', or '1d'."
        )
    amount, unit = float(match.group(1)), match.group(2)
    return datetime.timedelta(**{_DURATION_UNITS[unit]: amount})


def _parse_tol_value(value: Any) -> tuple[Any, float | None]:
    """Parse one tolerance spec into `(atol, rtol)`; `atol` may be a timedelta."""
    if isinstance(value, dict):
        unknown = set(value) - {"atol", "rtol"}
        if unknown:
            raise ValueError(f"Unknown tolerance option(s): {sorted(unknown)}.")
        atol = value.get("atol")
        rtol = value.get("rtol")
        if atol is not None and not isinstance(atol, (int, float)):
            atol = _parse_duration(atol)
        return atol, (float(rtol) if rtol is not None else None)
    if isinstance(value, bool):
        raise ValueError("A tolerance must be a number, a duration, or a dict.")
    if isinstance(value, (int, float)):
        return float(value), None
    return _parse_duration(value), None


def _parse_tolerance(tolerance: Any) -> dict[str, tuple[Any, float | None]]:
    """
    Parse `tolerance=` into a mapping of column -> `(atol, rtol)`. The key `"*"` holds a
    table-wide tolerance (applied to numeric columns only).
    """
    if tolerance is None:
        return {}
    if isinstance(tolerance, dict) and tolerance and set(tolerance) <= {"atol", "rtol"}:
        return {"*": _parse_tol_value(tolerance)}
    if isinstance(tolerance, dict):
        return {col: _parse_tol_value(v) for col, v in tolerance.items()}
    return {"*": _parse_tol_value(tolerance)}


def _parse_normalize(normalize: dict[str, Any] | None) -> dict[str, list[tuple[str, Any]]]:
    if not normalize:
        return {}
    out: dict[str, list[tuple[str, Any]]] = {}
    for col, ops in normalize.items():
        if isinstance(ops, (str, dict)):
            ops = [ops]
        parsed: list[tuple[str, Any]] = []
        for op in ops:
            if isinstance(op, dict):
                for name, arg in op.items():
                    parsed.append((name, arg))
            elif isinstance(op, (tuple, list)):
                parsed.append((op[0], op[1] if len(op) > 1 else None))
            else:
                parsed.append((op, None))
        for name, arg in parsed:
            if name not in _NORMALIZERS:
                raise ValueError(
                    f"Unknown normalization {name!r} for column {col!r}. Use one of "
                    f"{list(_NORMALIZERS)}."
                )
            if name == "round" and arg is None:
                raise ValueError(
                    "The 'round' normalization needs a number of digits, e.g. {'round': 2}."
                )
        out[col] = parsed
    return out
