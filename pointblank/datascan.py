from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, cast

import narwhals as nw
from great_tables import GT, google_font, html, loc, style
from narwhals.dataframe import LazyFrame

from pointblank._utils_html import _create_table_dims_html, _create_table_type_html, _fmt_frac
from pointblank.scan_profile import ColumnProfile, _as_physical, _DataProfile, _TypeMap
from pointblank.scan_profile_stats import COLUMN_ORDER_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from narwhals.dataframe import DataFrame
    from narwhals.typing import Frame

    from pointblank.scan_profile_stats import StatGroup


__all__ = ["DataScan", "DataScanDiff", "col_summary_tbl"]


class DataScan:
    """
    Get a summary of a dataset.

    The `DataScan` class provides a way to get a summary of a dataset. The summary includes the
    following information:

    - the name of the table (if provided)
    - the type of the table (e.g., `"polars"`, `"pandas"`, etc.)
    - the number of rows and columns in the table
    - column-level information, including:
        - the column name
        - the column type
        - measures of missingness and distinctness
        - measures of negative, zero, and positive values (for numerical columns)
        - a sample of the data (the first 5 values)
        - statistics (if the column contains numbers, strings, or datetimes)

    To obtain a dictionary representation of the summary, you can use the `to_dict()` method. To
    get a JSON representation of the summary, you can use the `to_json()` method. To save the JSON
    text to a file, the `save_to_json()` method could be used.

    :::{.callout-warning}
    The `DataScan()` class is still experimental. Please report any issues you encounter in the
    [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).
    :::

    Parameters
    ----------
    data
        The data to scan and summarize. This could be a DataFrame object, an Ibis table object,
        a CSV file path, a Parquet file path, a GitHub URL pointing to a CSV or Parquet file,
        or a database connection string.
    tbl_name
        Optionally, the name of the table could be provided as `tbl_name`.

    Measures of Missingness and Distinctness
    ----------------------------------------
    For each column, the following measures are provided:

    - `n_missing_values`: the number of missing values in the column
    - `f_missing_values`: the fraction of missing values in the column
    - `n_unique_values`: the number of unique values in the column
    - `f_unique_values`: the fraction of unique values in the column

    The fractions are calculated as the ratio of the measure to the total number of rows in the
    dataset.

    Counts and Fractions of Negative, Zero, and Positive Values
    -----------------------------------------------------------
    For numerical columns, the following measures are provided:

    - `n_negative_values`: the number of negative values in the column
    - `f_negative_values`: the fraction of negative values in the column
    - `n_zero_values`: the number of zero values in the column
    - `f_zero_values`: the fraction of zero values in the column
    - `n_positive_values`: the number of positive values in the column
    - `f_positive_values`: the fraction of positive values in the column

    The fractions are calculated as the ratio of the measure to the total number of rows in the
    dataset.

    Statistics for Numerical and String Columns
    -------------------------------------------
    For numerical and string columns, several statistical measures are provided. Please note that
    for string columms, the statistics are based on the lengths of the strings in the column.

    The following descriptive statistics are provided:

    - `mean`: the mean of the column
    - `std_dev`: the standard deviation of the column

    Additionally, the following quantiles are provided:

    - `min`: the minimum value in the column
    - `p05`: the 5th percentile of the column
    - `q_1`: the first quartile of the column
    - `med`: the median of the column
    - `q_3`: the third quartile of the column
    - `p95`: the 95th percentile of the column
    - `max`: the maximum value in the column
    - `iqr`: the interquartile range of the column

    Statistics for Date and Datetime Columns
    ----------------------------------------
    For date/datetime columns, the following statistics are provided:

    - `min`: the minimum date/datetime in the column
    - `max`: the maximum date/datetime in the column

    Returns
    -------
    DataScan
        A DataScan object.
    """

    # TODO: This needs to be generically typed at the class level, ie. DataScan[T]
    def __init__(self, data: Any, tbl_name: str | None = None) -> None:
        # Import processing functions from validate module
        from pointblank.validate import (
            _process_data,
        )

        # Process input data to handle different data source types
        data = _process_data(data)

        as_native = nw.from_native(data)

        if as_native.implementation.name == "IBIS" and as_native._level == "lazy":
            assert isinstance(as_native, LazyFrame)  # help mypy

            ibis_native = as_native.to_native()

            valid_conversion_methods = ("to_pyarrow", "to_pandas", "to_polars")
            for conv_method in valid_conversion_methods:
                try:
                    valid_native = getattr(ibis_native, conv_method)()
                except (NotImplementedError, ImportError, ModuleNotFoundError):  # pragma: no cover
                    continue  # pragma: no cover
                break
            else:  # pragma: no cover
                msg = (
                    "To use `ibis` as input, you must have one of arrow, pandas, polars or numpy "
                    "available in the process. Until `ibis` is fully supported by Narwhals, this is "
                    "necessary. Additionally, the data must be collected in order to calculate some "
                    "structural statistics, which may be performance detrimental."
                )
                raise ImportError(msg)  # pragma: no cover
            as_native = nw.from_native(valid_native)

        self.nw_data: Frame = nw.from_native(as_native)

        self.tbl_name: str | None = tbl_name
        self.profile: _DataProfile = self._generate_profile_df()

    def _generate_profile_df(self) -> _DataProfile:
        # Get schema and extract all column names from it
        schema: Mapping[str, Any] = self.nw_data.collect_schema()
        columns: list[str] = list(schema.keys())

        profile = _DataProfile(
            table_name=self.tbl_name,
            columns=columns,
            implementation=self.nw_data.implementation,
        )
        for column in columns:
            col_data: Frame = self.nw_data.select(column)

            ## Handle dtyping:
            native_dtype = schema[column]
            if _TypeMap.is_illegal(native_dtype):
                continue
            try:
                prof: type[ColumnProfile] = _TypeMap.fetch_profile(native_dtype)
            except NotImplementedError:
                continue

            col_profile = ColumnProfile(colname=column, coltype=str(native_dtype))

            ## Collect Sample Data:
            ## This is the most consistent way (i think) to get the samples out of the data.
            ## We can avoid writing our own logic to determine operations and rely on narwhals.
            raw_vals: list[Any] = (
                _as_physical(col_data.drop_nulls().head(5)).to_dict()[column].to_list()
            )
            col_profile.sample_data = [str(x) for x in raw_vals]

            col_profile.calc_stats(col_data)

            sub_profile: ColumnProfile = col_profile.spawn_profile(prof)
            sub_profile.calc_stats(col_data)

            profile.column_profiles.append(sub_profile)

        profile.set_row_count(self.nw_data)

        return profile

    @property
    def summary_data(self) -> Any:
        return self.profile.as_dataframe(strict=False).to_native()

    def get_tabular_report(self, *, show_sample_data: bool = False) -> GT:
        # Create the label, table type, and thresholds HTML fragments
        table_type_html = _create_table_type_html(
            tbl_type=str(self.profile.implementation), tbl_name=self.tbl_name, font_size="10px"
        )

        tbl_dims_html = _create_table_dims_html(
            columns=len(self.profile.columns), rows=self.profile.row_count, font_size="10px"
        )

        # Compose the subtitle HTML fragment
        combined_title = (
            "<div>"
            '<div style="padding-top: 0; padding-bottom: 7px;">'
            f"{table_type_html}"
            f"{tbl_dims_html}"
            "</div>"
            "</div>"
        )

        # TODO: Ensure width is 905px in total

        data: DataFrame = self.profile.as_dataframe(strict=False)

        ## Remove all null columns:
        all_null: list[str] = []
        for stat_name in data.iter_columns():
            col_len = len(stat_name.drop_nulls())
            if col_len == 0:
                all_null.append(stat_name.name)
        data = data.drop(all_null)

        if not show_sample_data:
            data = data.drop("sample_data")

        # find what stat cols were used in the analysis
        non_stat_cols = ("icon", "colname")  # TODO: need a better place for this
        present_stat_cols: set[str] = set(data.columns) - set(non_stat_cols)
        present_stat_cols.remove("coltype")
        with contextlib.suppress(KeyError):
            present_stat_cols.remove("freqs")  # TODO: currently used for html but no displayed?

        ## Assemble the target order and find what columns need borders.
        ## Borders should be placed to divide the stat "groups" and create a
        ## generally more aesthetically pleasing experience.
        target_order: list[str] = list(non_stat_cols)
        right_border_cols: list[str] = [non_stat_cols[-1]]

        last_group: StatGroup = COLUMN_ORDER_REGISTRY[0].group
        for col in COLUMN_ORDER_REGISTRY:
            if col.name in present_stat_cols:
                cur_group: StatGroup = col.group
                target_order.append(col.name)

                start_new_group: bool = last_group != cur_group
                if start_new_group:
                    last_group = cur_group
                    last_col_added = target_order[-2]  # -2 since we don't include the current
                    right_border_cols.append(last_col_added)

        right_border_cols.append(target_order[-1])  # add border to last stat col

        label_map: dict[str, Any] = self._build_label_map(target_order)

        ## Final Formatting:
        formatted_data = data.with_columns(
            colname=nw.concat_str(
                nw.lit(
                    "<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>"
                ),
                nw.col("colname"),
                nw.lit("</div><div style='font-size: 11px; color: gray;'>"),
                nw.col("coltype"),
                nw.lit("</div>"),
            ),
            __frac_n_unique=nw.col("n_unique") / nw.lit(self.profile.row_count),
            __frac_n_missing=nw.col("n_missing") / nw.lit(self.profile.row_count),
        )

        ## Pull out type indicies:
        # TODO: The stat types should get an enum? or something?
        # TODO: This all assumes the dates are separated by dashes, is that even true?
        # TODO: This all assumes date_stats are strings already, not ints or anything else.
        any_dates: bool = formatted_data.select(
            __tmp_idx=nw.col("coltype").str.contains("Date", literal=True)
        )["__tmp_idx"].any()
        if any_dates:
            date_stats = [c for c in present_stat_cols if c in ("min", "max")]

            formatted_data = formatted_data.with_columns(
                nw.when(nw.col("coltype").str.contains(r"\bDate\b", literal=False))
                .then(nw.col(c).cast(nw.String).str.replace_all("-", "<br>"))
                .otherwise(nw.col(c).cast(nw.String))
                for c in date_stats
            )

        any_datetimes: bool = formatted_data.select(
            __tmp_idx=nw.col("coltype").str.contains("Datetime", literal=True)
        )["__tmp_idx"].any()
        if any_datetimes:
            datetime_idx = [c for c in present_stat_cols if c in ("min", "max")]
            formatted_data = formatted_data.with_columns(
                nw.when(nw.col("coltype").str.contains(r"\bDatetime\b", literal=False))
                .then(nw.col(c).cast(nw.String).str.replace_all("-", "<br>"))
                .otherwise(nw.col(c).cast(nw.String))
                for c in datetime_idx
            )

        # format fractions:
        # this is an anti-pattern but there's no serious alternative
        _backend = cast(Any, self.profile.implementation)
        for _fmt_col in ("__frac_n_unique", "__frac_n_missing"):
            _formatted: list[str | None] = _fmt_frac(formatted_data[_fmt_col])
            formatted: nw.Series = nw.new_series(_fmt_col, values=_formatted, backend=_backend)
            formatted_data = formatted_data.drop(_fmt_col)
            formatted_data = formatted_data.with_columns(formatted.alias(_fmt_col))

        formatted_data = (
            # TODO: This is a temporary solution?
            # Format the unique and missing pct strings
            formatted_data.with_columns(
                n_unique=nw.concat_str(
                    nw.col("n_unique"),
                    nw.lit("<br>"),
                    nw.col("__frac_n_unique"),
                ),
                n_missing=nw.concat_str(
                    nw.col("n_missing"),
                    nw.lit("<br>"),
                    nw.col("__frac_n_missing"),
                ),
            )
            # TODO: Should be able to use selectors for this
            .drop("__frac_n_unique", "__frac_n_missing", "coltype")
        )

        if "freqs" in formatted_data.columns:  # TODO: don't love this arbitrary check
            # Extract HTML freqs:
            try:
                formatted_data = formatted_data.with_columns(
                    __freq_true=nw.col("freqs").struct.field("True"),
                    __freq_false=nw.col("freqs").struct.field("False"),
                )
            except Exception:  # TODO: should be narrowed if possible
                # if no struct implimentation exists, it must be done manually
                freq_ser: nw.Series = formatted_data["freqs"]
                trues: list[int | None] = []
                falses: list[int | None] = []
                for freq in freq_ser:
                    try:
                        trues.append(freq["True"])
                        falses.append(freq["False"])
                    except (KeyError, TypeError):
                        trues.append(None)
                        falses.append(None)
                true_ser: nw.Series = nw.new_series(
                    name="__freq_true", values=trues, backend=_backend
                )
                false_ser: nw.Series = nw.new_series(
                    name="__freq_false", values=falses, backend=_backend
                )
                formatted_data = formatted_data.with_columns(
                    __freq_true=true_ser, __freq_false=false_ser
                )

            ## format pct true values
            formatted_data = formatted_data.with_columns(
                # for bools, UQs are represented as percentages
                __pct_true=nw.col("__freq_true") / self.profile.row_count,
                __pct_false=nw.col("__freq_false") / self.profile.row_count,
            )
            for _fmt_col in ("__pct_true", "__pct_false"):
                _formatted: list[str | None] = _fmt_frac(formatted_data[_fmt_col])
                formatted = nw.new_series(name=_fmt_col, values=_formatted, backend=_backend)
                formatted_data = formatted_data.drop(_fmt_col)
                formatted_data = formatted_data.with_columns(formatted.alias(_fmt_col))

            formatted_data = (
                formatted_data.with_columns(
                    __bool_unique_html=nw.concat_str(
                        nw.lit("<span style='font-weight: bold;'>T</span>"),
                        nw.col("__pct_true"),
                        nw.lit("<br><span style='font-weight: bold;'>F</span>"),
                        nw.col("__pct_false"),
                    ),
                )
                .with_columns(
                    n_unique=nw.when(~nw.col("__bool_unique_html").is_null())
                    .then(nw.col("__bool_unique_html"))
                    .otherwise(nw.col("n_unique"))
                )
                .drop(
                    "__freq_true",
                    "__freq_false",
                    "__bool_unique_html",
                    "freqs",
                    "__pct_true",
                    "__pct_false",
                )
            )

        ## Determine Value Formatting Selectors:
        fmt_int: list[str] = formatted_data.select(nw.selectors.by_dtype(nw.dtypes.Int64)).columns
        fmt_float: list[str] = formatted_data.select(
            nw.selectors.by_dtype(nw.dtypes.Float64)
        ).columns

        ## GT Table:
        gt_tbl = (
            GT(formatted_data.to_native())
            .tab_header(title=html(combined_title))
            .tab_source_note(source_note="String columns statistics regard the string's length.")
            .cols_align(align="right", columns=list(present_stat_cols))
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(style=style.text(font=google_font("IBM Plex Mono")), locations=loc.body())
            .cols_move_to_start(target_order)
            ## Labeling
            .cols_label(label_map)
            .cols_label(icon="", colname="Column")
            .cols_align("center", columns=list(present_stat_cols))
            .tab_style(
                style=style.text(align="right"), locations=loc.body(columns=list(present_stat_cols))
            )
            ## Value Formatting
            .fmt_integer(columns=fmt_int)
            .fmt_number(
                columns=fmt_float,
                decimals=2,
                drop_trailing_dec_mark=True,
                drop_trailing_zeros=True,
            )
            ## Borders
            .tab_style(
                style=style.borders(sides="right", color="#D3D3D3", style="solid"),
                locations=loc.body(columns=right_border_cols),
            )
            .tab_style(
                style=style.borders(sides="left", color="#E5E5E5", style="dashed"),
                locations=loc.body(columns=list(present_stat_cols)),
            )
            ## Formatting
            .tab_style(
                style=style.text(size="10px"),
                locations=loc.body(columns=list(present_stat_cols)),
            )
            .tab_style(style=style.text(size="12px"), locations=loc.body(columns="colname"))
            .cols_width(
                cases={
                    "icon": "35px",
                    "colname": "200px",
                    **{stat_col: "60px" for stat_col in present_stat_cols},
                }
            )
        )

        if "PYARROW" != formatted_data.implementation.name:
            # TODO: this is more proactive than it should be
            gt_tbl = gt_tbl.sub_missing(missing_text="-")
            # https://github.com/posit-dev/great-tables/issues/667

        # If the version of `great_tables` is `>=0.17.0` then disable Quarto table processing
        if version("great_tables") >= "0.17.0":
            gt_tbl = gt_tbl.tab_options(quarto_disable_processing=True)

        return gt_tbl

    @staticmethod
    def _build_label_map(cols: Sequence[str]) -> dict[str, Any]:
        label_map: dict[str, Any] = {}
        for target_col in cols:
            try:
                matching_stat = next(
                    stat for stat in COLUMN_ORDER_REGISTRY if target_col == stat.name
                )
            except StopIteration:
                continue
            label_map[target_col] = matching_stat.label
        return label_map

    def to_dict(self) -> dict[str, Any]:
        """
        Export the profile as a structured dictionary.

        The returned dictionary contains metadata (table name, row count, column list) plus
        per-column profile entries with their data type, statistics, and sample data. This format is
        designed for round-trip persistence: save it with `to_json()` / `save_to_json()` and restore
        with `from_dict()` / `from_json()` / `load_from_json()`.

        Returns
        -------
        dict[str, Any]
            A dictionary with keys `"metadata"` and `"columns"`.
        """
        columns_out: list[dict[str, Any]] = []
        for prof in self.profile.column_profiles:
            stat_dict: dict[str, Any] = {}
            for stat in prof.statistics:
                stat_dict[stat.name] = stat.val

            columns_out.append(
                {
                    "colname": prof.colname,
                    "coltype": prof.coltype,
                    "sample_data": list(prof.sample_data),
                    "statistics": stat_dict,
                }
            )

        return {
            "metadata": {
                "table_name": self.profile.table_name,
                "row_count": self.profile.row_count,
                "columns": self.profile.columns,
            },
            "columns": columns_out,
        }

    def to_json(self) -> str:
        """
        Export the profile as a JSON string.

        The JSON is structured for round-trip persistence. Use `from_json()` or `load_from_json()`
        to restore a `DataScan` from the output.

        Returns
        -------
        str
            A JSON string representing the profile.
        """
        return json.dumps(self.to_dict(), indent=4, default=str)

    def save_to_json(self, output_file: str) -> None:
        """
        Save the profile to a JSON file.

        Parameters
        ----------
        output_file
            The path to the output JSON file.
        """
        with open(output_file, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataScan:
        """
        Restore a `DataScan` from a dictionary produced by `to_dict()`.

        This reconstructs the profile without needing the original data.

        Parameters
        ----------
        d
            A dictionary with `"metadata"` and `"columns"` keys, as produced by `to_dict()`.

        Returns
        -------
        DataScan
            A restored `DataScan` instance.
        """
        meta = d["metadata"]
        col_entries = d["columns"]

        obj = cls.__new__(cls)
        obj.nw_data = None  # type: ignore[assignment]
        obj.tbl_name = meta.get("table_name")

        profile = _DataProfile.__new__(_DataProfile)
        profile.table_name = meta.get("table_name")
        profile.row_count = meta["row_count"]
        profile.columns = meta["columns"]
        profile.implementation = nw.Implementation.POLARS
        profile.column_profiles = []

        stat_class_map: dict[str, type] = {
            stat_cls.name: stat_cls for stat_cls in COLUMN_ORDER_REGISTRY
        }

        for col_entry in col_entries:
            col_prof = ColumnProfile(
                colname=col_entry["colname"],
                coltype=col_entry["coltype"],
            )
            col_prof.sample_data = col_entry.get("sample_data", [])

            for stat_name, stat_val in col_entry.get("statistics", {}).items():
                stat_cls = stat_class_map.get(stat_name)
                if stat_cls is not None:
                    col_prof.statistics.append(stat_cls(stat_val))

            _assign_type_from_coltype(col_prof)
            profile.column_profiles.append(col_prof)

        obj.profile = profile
        return obj

    @classmethod
    def from_json(cls, json_string: str) -> DataScan:
        """
        Restore a `DataScan` from a JSON string produced by `to_json()`.

        Parameters
        ----------
        json_string
            A JSON string as produced by `to_json()`.

        Returns
        -------
        DataScan
            A restored `DataScan` instance.
        """
        return cls.from_dict(json.loads(json_string))

    @classmethod
    def load_from_json(cls, input_file: str) -> DataScan:
        """
        Load a `DataScan` from a JSON file produced by `save_to_json()`.

        Parameters
        ----------
        input_file
            The path to the JSON file.

        Returns
        -------
        DataScan
            A restored `DataScan` instance.
        """
        with open(input_file) as f:
            return cls.from_json(f.read())

    def compare(self, baseline: DataScan) -> DataScanDiff:
        """
        Compare this scan against a baseline and return the differences.

        The comparison covers schema changes (columns added, removed, or with changed types) and
        statistical drift for columns present in both scans. The returned `DataScanDiff` object
        provides programmatic access to the results and a tabular report via `get_tabular_report()`.

        Parameters
        ----------
        baseline
            The baseline `DataScan` to compare against (typically the older scan).

        Returns
        -------
        DataScanDiff
            An object describing the differences between the two scans.
        """
        return DataScanDiff(current=self, baseline=baseline)


def _assign_type_from_coltype(prof: ColumnProfile) -> None:
    coltype_lower = prof.coltype.lower()
    for type_enum in _TypeMap:
        if any(ind in coltype_lower for ind in type_enum.value):
            prof.__class__ = _TypeMap.fetch_prof_map()[type_enum]
            return


@dataclass
class _ColumnDiff:
    colname: str
    coltype_baseline: str | None
    coltype_current: str | None
    status: str
    stat_diffs: dict[str, tuple[Any, Any]]


class DataScanDiff:
    """
    The result of comparing two `DataScan` profiles.

    Created by calling `DataScan.compare()`. Provides programmatic access to schema changes and
    per-column statistical drift, plus a tabular report via `get_tabular_report()`.

    Attributes
    ----------
    columns_added
        Column names present in the current scan but not the baseline.
    columns_removed
        Column names present in the baseline but not the current scan.
    columns_type_changed
        Column names whose data type changed between baseline and current.
    column_diffs
        Per-column diff details for all columns that appear in either scan.
    """

    def __init__(self, current: DataScan, baseline: DataScan) -> None:
        self._current = current
        self._baseline = baseline

        cur_profiles = {p.colname: p for p in current.profile.column_profiles}
        base_profiles = {p.colname: p for p in baseline.profile.column_profiles}

        cur_names = set(cur_profiles.keys())
        base_names = set(base_profiles.keys())

        self.columns_added: list[str] = sorted(cur_names - base_names)
        self.columns_removed: list[str] = sorted(base_names - cur_names)

        self.columns_type_changed: list[str] = []
        self.column_diffs: list[_ColumnDiff] = []

        all_col_names = list(dict.fromkeys(list(base_profiles.keys()) + list(cur_profiles.keys())))

        for col_name in all_col_names:
            base_prof = base_profiles.get(col_name)
            cur_prof = cur_profiles.get(col_name)

            if base_prof is None:
                self.column_diffs.append(
                    _ColumnDiff(
                        colname=col_name,
                        coltype_baseline=None,
                        coltype_current=cur_prof.coltype if cur_prof else None,
                        status="added",
                        stat_diffs={},
                    )
                )
                continue

            if cur_prof is None:
                self.column_diffs.append(
                    _ColumnDiff(
                        colname=col_name,
                        coltype_baseline=base_prof.coltype,
                        coltype_current=None,
                        status="removed",
                        stat_diffs={},
                    )
                )
                continue

            type_changed = base_prof.coltype != cur_prof.coltype
            if type_changed:
                self.columns_type_changed.append(col_name)

            base_stats = {s.name: s.val for s in base_prof.statistics}
            cur_stats = {s.name: s.val for s in cur_prof.statistics}
            all_stat_names = list(dict.fromkeys(list(base_stats.keys()) + list(cur_stats.keys())))

            stat_diffs: dict[str, tuple[Any, Any]] = {}
            for stat_name in all_stat_names:
                base_val = base_stats.get(stat_name)
                cur_val = cur_stats.get(stat_name)
                if base_val != cur_val:
                    stat_diffs[stat_name] = (base_val, cur_val)

            status = "type_changed" if type_changed else ("changed" if stat_diffs else "unchanged")
            self.column_diffs.append(
                _ColumnDiff(
                    colname=col_name,
                    coltype_baseline=base_prof.coltype,
                    coltype_current=cur_prof.coltype,
                    status=status,
                    stat_diffs=stat_diffs,
                )
            )

    @property
    def has_changes(self) -> bool:
        """Return ``True`` if any schema or statistical changes were detected."""
        return bool(
            self.columns_added
            or self.columns_removed
            or self.columns_type_changed
            or any(d.stat_diffs for d in self.column_diffs)
        )

    @property
    def row_count_diff(self) -> tuple[int, int]:
        """Return `(baseline_row_count, current_row_count)`."""
        return (self._baseline.profile.row_count, self._current.profile.row_count)

    def to_dict(self) -> dict[str, Any]:
        """
        Export the comparison results as a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary with schema changes, row count diff, and per-column stat diffs.
        """
        return {
            "row_count": {
                "baseline": self._baseline.profile.row_count,
                "current": self._current.profile.row_count,
            },
            "columns_added": self.columns_added,
            "columns_removed": self.columns_removed,
            "columns_type_changed": [
                {
                    "column": d.colname,
                    "baseline_type": d.coltype_baseline,
                    "current_type": d.coltype_current,
                }
                for d in self.column_diffs
                if d.status == "type_changed"
            ],
            "stat_diffs": {
                d.colname: {
                    stat_name: {"baseline": bv, "current": cv}
                    for stat_name, (bv, cv) in d.stat_diffs.items()
                }
                for d in self.column_diffs
                if d.stat_diffs
            },
        }

    def get_tabular_report(self) -> GT:
        """
        Generate a GT table summarizing the differences between the two scans.

        Returns
        -------
        GT
            A styled Great Tables report showing schema and statistical drift.
        """
        import polars as pl

        rows: list[dict[str, Any]] = []

        for diff in self.column_diffs:
            if diff.status == "added":
                rows.append(
                    {
                        "column": diff.colname,
                        "status": "Added",
                        "type_baseline": "",
                        "type_current": diff.coltype_current or "",
                        "stat_changes": "",
                    }
                )
            elif diff.status == "removed":
                rows.append(
                    {
                        "column": diff.colname,
                        "status": "Removed",
                        "type_baseline": diff.coltype_baseline or "",
                        "type_current": "",
                        "stat_changes": "",
                    }
                )
            else:
                status = "Type Changed" if diff.status == "type_changed" else "OK"
                if diff.stat_diffs:
                    status = (
                        "Type + Stats Changed" if diff.status == "type_changed" else "Stats Changed"
                    )

                change_parts: list[str] = []
                for stat_name, (bv, cv) in diff.stat_diffs.items():
                    if stat_name == "freqs":
                        continue
                    bv_str = _format_stat_value(bv)
                    cv_str = _format_stat_value(cv)
                    change_parts.append(f"{stat_name}: {bv_str} -> {cv_str}")

                rows.append(
                    {
                        "column": diff.colname,
                        "status": status,
                        "type_baseline": diff.coltype_baseline or "",
                        "type_current": diff.coltype_current or "",
                        "stat_changes": "; ".join(change_parts),
                    }
                )

        if not rows:
            rows.append(
                {
                    "column": "(no columns)",
                    "status": "OK",
                    "type_baseline": "",
                    "type_current": "",
                    "stat_changes": "",
                }
            )

        df = pl.DataFrame(rows)

        base_rc, cur_rc = self.row_count_diff
        rc_note = f"Row count: {base_rc:,} (baseline) vs {cur_rc:,} (current)"

        base_name = self._baseline.tbl_name or "baseline"
        cur_name = self._current.tbl_name or "current"

        gt_tbl = (
            GT(df)
            .tab_header(
                title=html(f"Profile Comparison: {base_name} vs {cur_name}"),
                subtitle=html(rc_note),
            )
            .cols_label(
                column="Column",
                status="Status",
                type_baseline="Type (Baseline)",
                type_current="Type (Current)",
                stat_changes="Changed Statistics",
            )
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(
                style=style.text(font=google_font("IBM Plex Mono")),
                locations=loc.body(),
            )
            .tab_style(
                style=style.text(size="11px"),
                locations=loc.body(columns="stat_changes"),
            )
        )

        return gt_tbl

    def __repr__(self) -> str:
        n_changed = sum(1 for d in self.column_diffs if d.status != "unchanged")
        return (
            f"DataScanDiff("
            f"added={len(self.columns_added)}, "
            f"removed={len(self.columns_removed)}, "
            f"type_changed={len(self.columns_type_changed)}, "
            f"stat_changed={n_changed - len(self.columns_added) - len(self.columns_removed)})"
        )


def _format_stat_value(val: Any) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val)


def col_summary_tbl(data: Any, tbl_name: str | None = None) -> GT:
    """
    Generate a column-level summary table of a dataset.

    The `col_summary_tbl()` function generates a summary table of a dataset, focusing on providing
    column-level information about the dataset. The summary includes the following information:

    - the type of the table (e.g., `"polars"`, `"pandas"`, etc.)
    - the number of rows and columns in the table
    - column-level information, including:
        - the column name
        - the column type
        - measures of missingness and distinctness
        - descriptive stats and quantiles
        - statistics for datetime columns

    The summary table is returned as a GT object, which can be displayed in a notebook or saved to
    an HTML file.

    :::{.callout-warning}
    The `col_summary_tbl()` function is still experimental. Please report any issues you encounter
    in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).
    :::

    Parameters
    ----------
    data
        The table to summarize, which could be a DataFrame object, an Ibis table object, a CSV
        file path, a Parquet file path, or a database connection string. Read the *Supported Input
        Table Types* section for details on the supported table types.
    tbl_name
        Optionally, the name of the table could be provided as `tbl_name=`.

    Returns
    -------
    GT
        A GT object that displays the column-level summaries of the table.

    Supported Input Table Types
    ---------------------------
    The `data=` parameter can be given any of the following table types:

    - Polars DataFrame (`"polars"`)
    - Pandas DataFrame (`"pandas"`)
    - DuckDB table (`"duckdb"`)*
    - MySQL table (`"mysql"`)*
    - PostgreSQL table (`"postgresql"`)*
    - SQLite table (`"sqlite"`)*
    - Parquet table (`"parquet"`)*
    - CSV files (string path or `pathlib.Path` object with `.csv` extension)
    - Parquet files (string path, `pathlib.Path` object, glob pattern, directory with `.parquet`
    extension, or partitioned dataset)
    - GitHub URLs (direct links to CSV or Parquet files on GitHub)
    - Database connection strings (URI format with optional table specification)

    The table types marked with an asterisk need to be prepared as Ibis tables (with type of
    `ibis.expr.types.relations.Table`). Furthermore, using `col_summary_tbl()` with these types of
    tables requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a
    Polars or Pandas DataFrame, the availability of Ibis is not needed.

    Examples
    --------
    It's easy to get a column-level summary of a table using the `col_summary_tbl()` function.
    Here's an example using the `small_table` dataset (itself loaded using the
    [`load_dataset()`](`pointblank.load_dataset`) function):

    ```{python}
    import pointblank as pb

    small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

    pb.col_summary_tbl(data=small_table)
    ```

    This table used above was a Polars DataFrame, but the `col_summary_tbl()` function works with
    any table supported by `pointblank`, including Pandas DataFrames and Ibis backend tables.
    Here's an example using a DuckDB table handled by Ibis:

    ```{python}
    nycflights = pb.load_dataset(dataset="nycflights", tbl_type="duckdb")

    pb.col_summary_tbl(data=nycflights, tbl_name="nycflights")
    ```
    """

    # Import processing functions from validate module
    from pointblank.validate import _process_data

    # Process input data to handle different data source types
    data = _process_data(data)

    scanner = DataScan(data=data, tbl_name=tbl_name)
    return scanner.get_tabular_report()
