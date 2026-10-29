from __future__ import annotations

import contextlib
import json
import numbers
from dataclasses import dataclass
from html import escape as html_escape
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

    Before writing validation rules for a table, it helps to know what's in it: which columns
    have missing values, how many distinct values each column has, and what ranges the numbers
    and dates fall in. A `DataScan` collects these facts for every column in one pass. You can
    look at the result as a report, work with it as a table, or save it to compare against later.

    Once a scan is made, it can be used in several ways:

    - [`get_tabular_report()`](`pointblank.DataScan.get_tabular_report`) shows the summary as a
    table with one row per column (this is also what
    [`col_summary_tbl()`](`pointblank.col_summary_tbl`) returns)
    - [`summary_data`](`pointblank.DataScan.summary_data`) gives the statistics as a DataFrame, for
    filtering, sorting, or building checks on them
    - [`to_dict()`](`pointblank.DataScan.to_dict`), [`to_json()`](`pointblank.DataScan.to_json`),
    and [`save_to_json()`](`pointblank.DataScan.save_to_json`) store the profile, which
    [`from_dict()`](`pointblank.DataScan.from_dict`),
    [`from_json()`](`pointblank.DataScan.from_json`), and
    [`load_from_json()`](`pointblank.DataScan.load_from_json`) restore without the original data
    - [`compare()`](`pointblank.DataScan.compare`) compares two scans and reports schema changes
    and drift in each column's statistics, as a [`DataScanDiff`](`pointblank.DataScanDiff`)

    The scan stores the profile, not the data. To compare the rows of two tables rather than
    their profiles, use [`compare()`](`pointblank.compare`).

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
        Optionally, the name of the table could be provided as `tbl_name`. It's shown in the
        header of the report, stored with the profile, and used to label the two scans in a
        [`DataScanDiff`](`pointblank.DataScanDiff`) report.

    Measures of Missingness and Distinctness
    ----------------------------------------
    For each column, the following measures are provided:

    - `n_missing`: the number of missing values in the column
    - `n_unique`: the number of distinct values in the column (if there are missing values, they
    count as one more distinct value)

    The report also shows each count as a fraction of the number of rows in the table.

    Statistics for Numeric and String Columns
    -----------------------------------------
    For numeric and string columns, the following statistics are provided. For string columns,
    they describe the lengths of the strings (in characters), not the strings themselves, which
    makes it easy to spot truncated values or codes of an unexpected length.

    - `mean`: the mean
    - `std`: the standard deviation
    - `min`: the minimum
    - `p05`: the 5th percentile
    - `q_1`: the first quartile
    - `median`: the median
    - `q_3`: the third quartile
    - `p95`: the 95th percentile
    - `max`: the maximum
    - `iqr`: the interquartile range (`q_3 - q_1`)

    Statistics for Date, Datetime, and Time Columns
    -----------------------------------------------
    For date, datetime, and time columns, the following statistics are provided:

    - `min`: the earliest value in the column
    - `max`: the latest value in the column

    Statistics for Boolean Columns
    ------------------------------
    For Boolean columns, `freqs` gives the number of `True` and `False` values (as a dictionary,
    e.g., `{"True": 9, "False": 4}`). The report shows these as percentages.

    Supported Column Types
    ----------------------
    Numeric columns (integers, floats and decimals), string columns (including categorical and
    enum columns), date, datetime and time columns, and Boolean columns are summarized. Columns of
    other types (e.g., lists, arrays, structs, durations and binary data) can't be summarized, so
    they're listed in the report with no statistics, and in the JSON under `skipped_columns`.
    pandas stores decimals, times, and Booleans with missing values as `object` columns, which
    can't be summarized either.

    Returns
    -------
    DataScan
        A DataScan object.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's scan the `small_table` dataset, which has 13 rows and 8 columns of several types.

    ```{python}
    import pointblank as pb

    small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

    scan = pb.DataScan(small_table, tbl_name="small_table")

    scan.get_tabular_report()
    ```

    The report has one row per column. Column `c` has 2 missing values, `d` spans a wide range
    (from about 108 to 9,999.99), and the string column `b` always has 9 characters, which suggests
    a fixed-format code. Column `e` is Boolean, so it shows the share of `True` and `False` values
    instead of statistics.

    The same statistics are available as a DataFrame through `summary_data`, so they can be
    filtered like any other table. Here are the columns with missing values:

    ```{python}
    import polars as pl

    (
        scan.summary_data
        .filter(pl.col("n_missing") > 0)
        .select("colname", "coltype", "n_missing", "n_unique")
    )
    ```

    A scan can also be kept as a baseline. If its profile is saved now with
    [`save_to_json()`](`pointblank.DataScan.save_to_json`), a later scan of a new version of the
    table can be compared against it with [`compare()`](`pointblank.DataScan.compare`), which
    reports added and removed columns, type changes, and changes in each statistic.

    See Also
    --------
    col_summary_tbl : Get the same report directly from a table.
    missing_vals_tbl : See where in a table the missing values are.
    preview : Show the first and last rows of a table.
    DataScanDiff : The result of comparing two scans.
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
        if not columns:
            raise ValueError("The table has no columns, so there is nothing to summarize.")

        profile = _DataProfile(
            table_name=self.tbl_name,
            columns=columns,
            implementation=self.nw_data.implementation,
        )
        for column in columns:
            col_data: Frame = self.nw_data.select(column)

            ## Handle dtyping:
            native_dtype = schema[column]
            try:
                prof: type[ColumnProfile] = _TypeMap.fetch_profile(native_dtype)
            except NotImplementedError:
                profile.skipped_columns[column] = str(native_dtype)
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
        """The column statistics as a DataFrame.

        The report from [`get_tabular_report()`](`pointblank.DataScan.get_tabular_report`) is for
        reading, while `summary_data` holds the same statistics as data. That's useful for finding
        columns that need attention (the ones with missing values, or with a single distinct
        value), for building validation steps from what was found, or for passing the profile to
        another tool.

        The statistics are described in the [`DataScan`](`pointblank.DataScan`) documentation. A
        statistic that doesn't apply to a column's type is null (for example, `mean` for a date
        column). The order of the statistic columns isn't fixed, so select columns by name.

        Returns
        -------
        Any
            A DataFrame of the same type as the scanned table (Polars for a scan restored with
            [`from_dict()`](`pointblank.DataScan.from_dict`) or a related method), with one row per
            summarized column. The `colname` and `coltype` columns give each column's name and
            data type, `sample_data` holds the first five non-missing values (as one string), and
            `icon` is the type icon used in the report. The other columns are the statistics.
            When a statistic holds values of different types across columns (e.g., `min` for a
            number and a date), its values are converted to strings. Columns of unsupported types
            aren't included.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Here's the summary for the numeric columns of the `small_table` dataset.

        ```{python}
        import pointblank as pb
        import polars as pl

        small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

        summary = pb.DataScan(small_table).summary_data

        (
            summary
            .filter(pl.col("coltype").is_in(["Int64", "Float64"]))
            .select("colname", "n_missing", "n_unique", "mean", "min", "max")
        )
        ```

        Column `c` is the only one with missing values. The `min` and `max` columns are strings,
        because they also hold the dates from the date columns.

        Since this is an ordinary DataFrame, the profile can lead to further checks. This finds
        the columns with one distinct value fewer than the table has rows:

        ```{python}
        summary.filter(pl.col("n_unique") == small_table.height - 1).get_column("colname").to_list()
        ```

        Three columns that look like they should identify each row (a timestamp, a code, and an
        amount) each have exactly one repeated value. That's a sign of a duplicated row, which
        [`Validate.rows_distinct()`](`pointblank.Validate.rows_distinct`) would confirm.
        """
        return self.profile.as_dataframe(strict=False).to_native()

    def get_tabular_report(self, *, show_sample_data: bool = False) -> GT:
        """Get a tabular report of the scan.

        The report presents the profile of each column as one row of a table: the column's name
        and type, the number and fraction of missing and distinct values, and its statistics. It's
        the quickest way to get an overview of a table you haven't seen before, and it's what
        [`col_summary_tbl()`](`pointblank.col_summary_tbl`) returns.

        Each row starts with an icon for the column's type. Statistics that don't apply to any
        column are left out, and those that don't apply to a particular column are shown as `-`.
        Statistics for string columns describe the lengths of the strings (as the note below the
        table says). Boolean columns show the percentages of `True` and `False` values in place of
        the number of distinct values. Columns of unsupported types are listed with an
        *unsupported type* badge and no statistics.

        Parameters
        ----------
        show_sample_data
            Should a column with the first five non-missing values of each column be added? This
            is off by default, since the values can be long, and the table may hold sensitive data.

        Returns
        -------
        GT
            A Great Tables object.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's scan the `game_revenue` dataset and look at the report.

        ```{python}
        import pointblank as pb

        game_revenue = pb.load_dataset(dataset="game_revenue", tbl_type="polars")

        scan = pb.DataScan(game_revenue, tbl_name="game_revenue")

        scan.get_tabular_report()
        ```

        The header shows the table type and its dimensions. In the rows, `item_revenue` has a
        long right tail (its 95th percentile is far below its maximum), and the string columns
        `player_id` and `session_id` have fixed lengths. To see what the values look like, add
        the sample data:

        ```{python}
        scan.get_tabular_report(show_sample_data=True)
        ```

        The samples make it clear that `player_id` and `session_id` are codes, and they show the
        format of each column's values. The report is a Great Tables object, so it can be saved
        to a standalone HTML file with its `write_raw_html()` method.
        """
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
        # (the column name and type are escaped since they go into HTML; e.g., an Ibis type can
        # be `array<int64>`)
        formatted_data = data.with_columns(
            colname=nw.concat_str(
                nw.lit(
                    "<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>"
                ),
                _html_escape_expr(nw.col("colname")),
                nw.lit("</div><div style='font-size: 11px; color: gray;'>"),
                _html_escape_expr(nw.col("coltype")),
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
            # Build the T/F summary for boolean columns straight from each column's profile.
            # The `freqs` dict only has keys for the values present (e.g., `{"True": 3}` for an
            # all-`True` column), so a missing key is a count of zero. The rows of `data` follow
            # the order of `column_profiles`.
            bool_unique_html: list[str | None] = []
            for prof in self.profile.column_profiles:
                freqs = next((stat.val for stat in prof.statistics if stat.name == "freqs"), None)
                if not isinstance(freqs, dict):
                    bool_unique_html.append(None)
                    continue
                pct_true, pct_false = _fmt_frac(
                    [
                        freqs.get("True", 0) / self.profile.row_count,
                        freqs.get("False", 0) / self.profile.row_count,
                    ]
                )
                bool_unique_html.append(
                    f"<span style='font-weight: bold;'>T</span>{pct_true}"
                    f"<br><span style='font-weight: bold;'>F</span>{pct_false}"
                )
            bool_ser: nw.Series = nw.new_series(
                name="__bool_unique_html",
                values=bool_unique_html,
                dtype=nw.String(),
                backend=_backend,
            )

            formatted_data = (
                formatted_data.with_columns(__bool_unique_html=bool_ser)
                .with_columns(
                    n_unique=nw.when(~nw.col("__bool_unique_html").is_null())
                    .then(nw.col("__bool_unique_html"))
                    .otherwise(nw.col("n_unique"))
                )
                .drop("__bool_unique_html", "freqs")
            )

        ## Determine Value Formatting Selectors:
        fmt_int: list[str] = formatted_data.select(nw.selectors.by_dtype(nw.dtypes.Int64)).columns
        fmt_float: list[str] = formatted_data.select(
            nw.selectors.by_dtype(nw.dtypes.Float64)
        ).columns

        # (after the selectors, since the integer columns become floats to hold the empty cells)
        if self.profile.skipped_columns:
            formatted_data = self._with_skipped_column_rows(formatted_data)

        # The columns built as HTML above; GT escapes the cells that no formatter touches (with
        # `great_tables>=1.1.0` that includes cells skipped by `sub_missing()`), so these are given
        # an identity formatter. Other string cells (e.g., `sample_data`) hold the table's own
        # values and stay escaped.
        html_cols: list[str] = [
            c
            for c in ("icon", "colname", "n_unique", "n_missing", "min", "max")
            if c in formatted_data.columns and c not in fmt_int + fmt_float
        ]

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
            .fmt(lambda x: x, columns=html_cols)
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

        # TODO: this is more proactive than it should be
        gt_tbl = gt_tbl.sub_missing(missing_text="-")

        # If the version of `great_tables` is `>=0.17.0` then disable Quarto table processing
        if version("great_tables") >= "0.17.0":
            gt_tbl = gt_tbl.tab_options(quarto_disable_processing=True)

        return gt_tbl

    def _with_skipped_column_rows(self, data: DataFrame) -> DataFrame:
        """
        Add a row for each column that couldn't be summarized, in the table's column order.

        The row has the column's name and type, with an "unsupported type" badge, and no
        statistics. The rows of `data` follow the order of `column_profiles`.
        """
        skipped = self.profile.skipped_columns
        badge = (
            "<span style='display: inline-block; margin-top: 2px; padding: 1px 5px; "
            "border-radius: 3px; background-color: #EEEEEE; color: #666666; font-size: 9px; "
            "font-weight: 600; white-space: nowrap;'>UNSUPPORTED TYPE</span>"
        )
        skipped_rows: dict[str, list[Any]] = {c: [None] * len(skipped) for c in data.columns}
        skipped_rows["icon"] = [_TypeMap.fetch_icon(None)] * len(skipped)  # the `object` icon
        skipped_rows["colname"] = [
            "<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; "
            f"overflow: hidden;'>{html_escape(name, quote=False)}</div>"
            f"<div style='font-size: 11px; color: gray;'>{html_escape(coltype, quote=False)}</div>"
            f"{badge}"
            for name, coltype in skipped.items()
        ]
        backend = cast(Any, self.profile.implementation)
        # A pandas integer column can't hold the empty cells, so integer columns become floats
        # (the caller has already picked the columns to format as integers)
        schema = {
            name: nw.Float64() if dtype.is_integer() else dtype
            for name, dtype in data.schema.items()
        }
        data = data.with_columns(nw.col(name).cast(dtype) for name, dtype in schema.items())
        # On pandas, `from_dict()` (and, before Narwhals 2.27, `lit(None, dtype=...)`) turns `None`
        # in a string column into the text "None", which `sub_missing()` doesn't replace. A `when()`
        # that's never true (every skipped row has a name) makes the empty cells real nulls.
        skipped_tbl = nw.from_dict(skipped_rows, schema=schema, backend=backend).with_columns(
            nw.when(nw.col("colname").is_null()).then(nw.col(name)).alias(name)
            for name in data.columns
            if name not in ("icon", "colname")
        )

        position = {name: i for i, name in enumerate(self.profile.columns)}
        positions = [position[p.colname] for p in self.profile.column_profiles]
        positions += [position[name] for name in skipped]
        return (
            nw.concat([data, skipped_tbl], how="vertical")
            .with_columns(
                nw.new_series("__position", values=positions, dtype=nw.Int64(), backend=backend)
            )
            .sort("__position")
            .drop("__position")
        )

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

        A profile is worth keeping when it serves as a baseline: a record of what a table looked
        like at one point, to compare later scans against, or to store alongside the table in a
        catalog. `to_dict()` returns the whole profile as plain Python data, which
        [`from_dict()`](`pointblank.DataScan.from_dict`) turns back into a `DataScan`. For a JSON
        string or file, use [`to_json()`](`pointblank.DataScan.to_json`) or
        [`save_to_json()`](`pointblank.DataScan.save_to_json`).

        The profile includes the first five non-missing values of each column (`sample_data`). If
        the table holds sensitive values, keep that in mind before storing or sharing the result.

        Returns
        -------
        dict[str, Any]
            A dictionary with two keys. `"metadata"` holds the table name, the row count, the
            list of column names, and `"skipped_columns"` (a dictionary mapping each column of an
            unsupported type to its type). `"columns"` holds a list with one entry per summarized
            column, each a dictionary with `"colname"`, `"coltype"`, `"sample_data"`, and
            `"statistics"` (a dictionary mapping each statistic's name to its value).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's scan a small table of sensor readings, where one reading is missing.

        ```{python}
        import pointblank as pb
        import polars as pl

        readings = pl.DataFrame(
            {"sensor": ["T1", "T2", "T3", "T4"], "temp_c": [21.4, 22.1, None, 20.8]}
        )

        profile = pb.DataScan(readings, tbl_name="readings").to_dict()

        profile["metadata"]
        ```

        Each column has its own entry, with the statistics in a nested dictionary:

        ```{python}
        profile["columns"][1]["statistics"]
        ```

        The missing reading is counted in `n_missing` and left out of the other statistics. The
        statistics keep their Python types (the minimum and maximum of a date column, for example,
        are `datetime.date` values), and passing the dictionary to
        [`from_dict()`](`pointblank.DataScan.from_dict`) gives back the same profile.
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
                # Column name -> type of the columns that couldn't be summarized
                "skipped_columns": dict(self.profile.skipped_columns),
            },
            "columns": columns_out,
        }

    def to_json(self) -> str:
        """
        Export the profile as a JSON string.

        JSON is the format to use when a profile is stored outside of Python or sent to another
        system: in a database column, an object store, or an API response. The structure is the
        same as that of [`to_dict()`](`pointblank.DataScan.to_dict`), and
        [`from_json()`](`pointblank.DataScan.from_json`) restores a `DataScan` from it. Values that
        JSON can't represent directly, such as dates, are written as strings. To write the JSON
        to a file, use [`save_to_json()`](`pointblank.DataScan.save_to_json`).

        Returns
        -------
        str
            A JSON string representing the profile, indented by four spaces.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Here, a table of shipments with a date column is scanned and converted to JSON.

        ```{python}
        import pointblank as pb
        import polars as pl
        from datetime import date

        shipments = pl.DataFrame(
            {"shipped": [date(2025, 3, 1), date(2025, 3, 4), date(2025, 3, 9)]}
        )

        profile_json = pb.DataScan(shipments, tbl_name="shipments").to_json()

        print(profile_json)
        ```

        The earliest and latest dates are stored as strings. Because the result is text, it can
        be stored wherever text can, and [`from_json()`](`pointblank.DataScan.from_json`) turns it
        back into a `DataScan`.
        """
        return json.dumps(self.to_dict(), indent=4, default=_json_default)

    def save_to_json(self, output_file: str) -> None:
        """
        Save the profile to a JSON file.

        Saving a scan to a file is the simplest way to keep a baseline: scan a table once, save
        the profile, and load it later with
        [`load_from_json()`](`pointblank.DataScan.load_from_json`) to compare a new scan against
        it with [`compare()`](`pointblank.DataScan.compare`). The file holds the output of
        [`to_json()`](`pointblank.DataScan.to_json`), so it's small (it stores the statistics, not
        the data) and can be kept in version control next to the code that produces the table.

        Parameters
        ----------
        output_file
            The path to the output JSON file. An existing file is overwritten.

        Returns
        -------
        None
            Nothing is returned. The profile is written to `output_file`.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's save a profile of the `small_table` dataset to a file in a temporary directory.

        ```{python}
        import pointblank as pb
        import tempfile
        from pathlib import Path

        small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

        profile_path = Path(tempfile.mkdtemp()) / "small_table_profile.json"

        pb.DataScan(small_table, tbl_name="small_table").save_to_json(profile_path)

        print(profile_path.read_text()[:180])
        ```

        The file starts with the table's metadata, and the column profiles follow. Loading it with
        [`load_from_json()`](`pointblank.DataScan.load_from_json`) gives back the same profile,
        without needing the table itself.
        """
        with open(output_file, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataScan:
        """
        Restore a `DataScan` from a dictionary.

        The dictionary is the one produced by [`to_dict()`](`pointblank.DataScan.to_dict`). This
        reconstructs the profile without needing the original data, so a profile that was stored
        earlier can be viewed again with
        [`get_tabular_report()`](`pointblank.DataScan.get_tabular_report`), queried through
        [`summary_data`](`pointblank.DataScan.summary_data`), or used as the baseline in
        [`compare()`](`pointblank.DataScan.compare`).

        A restored scan has the statistics but not the data. Its `summary_data` is a Polars
        DataFrame, whatever the type of the original table. And when it's compared with another
        scan, the statistics are compared as usual, but drift scores (which need the values
        themselves) are only computed for Boolean columns.

        Parameters
        ----------
        d
            A dictionary with `"metadata"` and `"columns"` keys, as produced by
            [`to_dict()`](`pointblank.DataScan.to_dict`).

        Returns
        -------
        DataScan
            A restored `DataScan` instance.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's take a profile of a table of sensor readings as a dictionary, then restore it.

        ```{python}
        import pointblank as pb
        import polars as pl

        readings = pl.DataFrame(
            {"sensor": ["T1", "T2", "T3", "T4"], "temp_c": [21.4, 22.1, None, 20.8]}
        )

        profile = pb.DataScan(readings, tbl_name="readings").to_dict()

        restored = pb.DataScan.from_dict(profile)

        restored.get_tabular_report()
        ```

        The report is the same as the one for the original scan, even though `restored` was made
        from the dictionary alone. Converting the restored scan back to a dictionary gives the
        same dictionary:

        ```{python}
        restored.to_dict() == profile
        ```

        So a profile can go through any number of save-and-restore cycles without losing
        information.
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
        profile.skipped_columns = dict(meta.get("skipped_columns", {}))

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
        Restore a `DataScan` from a JSON string.

        The JSON string is the one produced by [`to_json()`](`pointblank.DataScan.to_json`). It's
        the counterpart to that method for profiles stored as text, such as in a database column
        or an API response. To read a JSON file, use
        [`load_from_json()`](`pointblank.DataScan.load_from_json`).

        As with [`from_dict()`](`pointblank.DataScan.from_dict`), the restored scan has the
        statistics but not the data. Values that were written as strings in the JSON (such as
        dates) stay strings.

        Parameters
        ----------
        json_string
            A JSON string as produced by [`to_json()`](`pointblank.DataScan.to_json`).

        Returns
        -------
        DataScan
            A restored `DataScan` instance.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Suppose the profile of the `small_table` dataset was stored as JSON text some time ago.

        ```{python}
        import pointblank as pb
        import polars as pl

        small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

        stored_json = pb.DataScan(small_table, tbl_name="small_table").to_json()
        ```

        Restoring it gives a `DataScan` that can be queried like the original one:

        ```{python}
        restored = pb.DataScan.from_json(stored_json)

        restored.summary_data.select("colname", "n_missing", "n_unique")
        ```

        The table name, row count, and every column's statistics come from the JSON, so the
        original table isn't needed to answer questions about it later.
        """
        return cls.from_dict(json.loads(json_string))

    @classmethod
    def load_from_json(cls, input_file: str) -> DataScan:
        """
        Load a `DataScan` from a JSON file produced by `save_to_json()`.

        This is how a saved baseline is brought back: load the profile from the file, then pass it
        to the [`compare()`](`pointblank.DataScan.compare`) method of a new scan to find out what
        has changed since. The file is the one written by
        [`save_to_json()`](`pointblank.DataScan.save_to_json`) (or any file containing the output
        of [`to_json()`](`pointblank.DataScan.to_json`)).

        As with [`from_dict()`](`pointblank.DataScan.from_dict`), the loaded scan has the
        statistics but not the data, so drift scores are only computed for its Boolean columns.

        Parameters
        ----------
        input_file
            The path to the JSON file.

        Returns
        -------
        DataScan
            A restored `DataScan` instance.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's save a profile of a table of orders as a baseline.

        ```{python}
        import pointblank as pb
        import polars as pl
        import tempfile
        from pathlib import Path

        orders = pl.DataFrame(
            {"order_id": [1, 2, 3, 4], "amount": [25.0, 40.0, 15.5, 60.0]}
        )

        baseline_path = Path(tempfile.mkdtemp()) / "orders_profile.json"

        pb.DataScan(orders, tbl_name="orders").save_to_json(baseline_path)
        ```

        Later, the table has grown and a refund (a negative amount) has appeared. The baseline is
        loaded from the file and compared with a scan of the current table:

        ```{python}
        orders_now = pl.DataFrame(
            {"order_id": [1, 2, 3, 4, 5], "amount": [25.0, 40.0, 15.5, 60.0, -15.5]}
        )

        baseline = pb.DataScan.load_from_json(baseline_path)

        diff = pb.DataScan(orders_now, tbl_name="orders").compare(baseline)

        diff.to_dict()["stat_diffs"]["amount"]["min"]
        ```

        The minimum `amount` went from 15.5 to -15.5. A change like this is easy to miss in the
        data but stands out in a comparison of profiles, and a check such as
        [`Validate.col_vals_ge()`](`pointblank.Validate.col_vals_ge`) can then guard against it.
        """
        with open(input_file) as f:
            return cls.from_json(f.read())

    def compare(self, baseline: DataScan) -> DataScanDiff:
        """
        Compare this scan against a baseline and return the differences.

        Tables that are refreshed regularly (a monthly extract, a nightly load, a model's input
        data) can change in ways that no single row reveals: a column is dropped, a type changes,
        or the values shift to a different range. Comparing a scan of the current table with a
        scan of an earlier version that's known to be good finds those changes. The baseline is
        often a profile saved earlier with [`save_to_json()`](`pointblank.DataScan.save_to_json`)
        and brought back with [`load_from_json()`](`pointblank.DataScan.load_from_json`).

        The scan this method is called on is the *current* one, and `baseline` is the one it's
        compared against. The result is a [`DataScanDiff`](`pointblank.DataScanDiff`), which has
        the differences as attributes, as a dictionary through
        [`to_dict()`](`pointblank.DataScanDiff.to_dict`), and as a report through
        [`get_tabular_report()`](`pointblank.DataScanDiff.get_tabular_report`).

        This compares profiles, not rows. To find out which rows of two tables differ, use
        [`compare()`](`pointblank.compare`) instead.

        What Is Compared
        ----------------
        Columns are matched by name. A column that's only in the current scan is *added*, one
        that's only in the baseline is *removed*, and one whose data type differs is
        *type-changed*. For every column in both scans, each statistic is compared for exact
        equality, so any difference at all (even in the last decimal place of a mean) is
        reported along with both values. When the type has changed, the statistics might no
        longer measure the same thing (a numeric column's mean is of its values, but a string
        column's mean is of its lengths).

        Drift Scores
        ------------
        Statistics show that something changed but not how much the distribution of values moved.
        For that, drift scores are computed for columns whose type hasn't changed:

        - numeric columns get the population stability index (`psi`), with the values divided into
        ten bins at the baseline's deciles (at least ten values are needed on each side), and the
        Kolmogorov–Smirnov statistic (`ks_statistic`, the largest gap between the two cumulative
        distributions) with its p-value (`ks_p_value`). The p-value comes from SciPy if it's
        installed and from an approximation otherwise
        - all other columns get a `psi` computed from how often each distinct value appears

        A common rule of thumb reads a PSI below 0.1 as little or no shift, from 0.1 to 0.25 as a
        moderate shift, and above 0.25 as a large one. A score of zero is left out.

        Drift scores need the values themselves, so they're only computed when both scans were
        made from data. A scan restored with [`from_dict()`](`pointblank.DataScan.from_dict`),
        [`from_json()`](`pointblank.DataScan.from_json`), or
        [`load_from_json()`](`pointblank.DataScan.load_from_json`) only has statistics, and then
        only Boolean columns (whose value counts are part of the profile) get a drift score.

        Parameters
        ----------
        baseline
            The baseline `DataScan` to compare against (typically the older scan).

        Returns
        -------
        DataScanDiff
            An object describing the differences between the two scans.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's compare two monthly extracts of an orders table. Between January and February, the
        `gift` column was dropped, a `coupon` column was added, a new `"app"` channel appeared, and
        order amounts went up.

        ```{python}
        import pointblank as pb
        import polars as pl

        jan = pl.DataFrame(
            {
                "order_id": list(range(1, 21)),
                "amount": [20.0 + i for i in range(20)],
                "channel": ["web"] * 14 + ["store"] * 6,
                "gift": [i % 4 == 0 for i in range(20)],
            }
        )

        feb = pl.DataFrame(
            {
                "order_id": list(range(21, 45)),
                "amount": [35.0 + 1.5 * i for i in range(24)],
                "channel": ["web"] * 10 + ["store"] * 8 + ["app"] * 6,
                "coupon": ["SPRING"] * 5 + [None] * 19,
            }
        )

        scan_jan = pb.DataScan(jan, tbl_name="orders_jan")
        scan_feb = pb.DataScan(feb, tbl_name="orders_feb")

        diff = scan_feb.compare(scan_jan)

        diff
        ```

        The printed summary counts one added column, one removed column, and three columns whose
        statistics changed. The report shows the details:

        ```{python}
        diff.get_tabular_report()
        ```

        The PSI of `amount` (about 6.7) and its KS statistic (0.875) are far beyond the rule of
        thumb, so the amounts really did move. The `channel` column has a PSI of about 2.1 because
        of the new `"app"` value. The shift in `order_id` is expected, since each month has its
        own IDs, which shows that drift scores still need to be read with the meaning of each
        column in mind.

        See Also
        --------
        DataScanDiff : The result of comparing two scans.
        pointblank.compare : Compare two tables row by row.
        """
        return DataScanDiff(current=self, baseline=baseline)


def _json_default(value: Any) -> Any:
    """Make a statistic JSON-ready: NumPy numbers become Python numbers, anything else a string."""
    # pandas gives NumPy integers (e.g., `n_unique`), which `str()` would turn into text
    if isinstance(value, numbers.Number) and hasattr(value, "item"):
        return value.item()
    return str(value)


def _html_escape_expr(expr: nw.Expr) -> nw.Expr:
    """Escape `&`, `<` and `>` in a string expression, so its text can go into HTML."""
    return (
        expr.str.replace_all("&", "&amp;", literal=True)
        .str.replace_all("<", "&lt;", literal=True)
        .str.replace_all(">", "&gt;", literal=True)
    )


def _assign_type_from_coltype(prof: ColumnProfile) -> None:
    type_enum = _TypeMap.from_dtype(prof.coltype)
    if type_enum is not None:
        prof.__class__ = _TypeMap.fetch_prof_map()[type_enum]


def _compute_psi_numeric(
    baseline_vals: list[float],
    current_vals: list[float],
    n_bins: int = 10,
) -> float | None:
    """Compute PSI for numeric columns using quantile-based binning."""
    if len(baseline_vals) < n_bins or len(current_vals) < n_bins:
        return None

    import math

    sorted_base = sorted(baseline_vals)
    n = len(sorted_base)
    edges = [sorted_base[0]]
    for i in range(1, n_bins):
        idx = int(i * n / n_bins)
        edges.append(sorted_base[min(idx, n - 1)])
    edges.append(sorted_base[-1] + 1e-10)

    edges = list(dict.fromkeys(edges))
    if len(edges) < 3:
        return None

    def _bin_counts(vals: list[float], bin_edges: list[float]) -> list[int]:
        counts = [0] * (len(bin_edges) - 1)
        for v in vals:
            for j in range(len(bin_edges) - 1):
                if bin_edges[j] <= v < bin_edges[j + 1]:
                    counts[j] += 1
                    break
            else:
                counts[-1] += 1
        return counts

    base_counts = _bin_counts(baseline_vals, edges)
    cur_counts = _bin_counts(current_vals, edges)

    total_base = sum(base_counts)
    total_cur = sum(cur_counts)
    if total_base == 0 or total_cur == 0:
        return None

    eps = 1e-4
    psi = 0.0
    for bc, cc in zip(base_counts, cur_counts):
        p = max(bc / total_base, eps)
        q = max(cc / total_cur, eps)
        psi += (q - p) * math.log(q / p)

    return psi


def _compute_psi_categorical(
    baseline_freqs: dict[str, int],
    current_freqs: dict[str, int],
) -> float | None:
    """Compute PSI for categorical columns using frequency distributions."""
    import math

    all_keys = set(baseline_freqs) | set(current_freqs)
    if not all_keys:
        return None

    total_base = sum(baseline_freqs.values())
    total_cur = sum(current_freqs.values())
    if total_base == 0 or total_cur == 0:
        return None

    eps = 1e-4
    psi = 0.0
    for key in all_keys:
        p = max(baseline_freqs.get(key, 0) / total_base, eps)
        q = max(current_freqs.get(key, 0) / total_cur, eps)
        psi += (q - p) * math.log(q / p)

    return psi


def _compute_ks_statistic(
    baseline_vals: list[float],
    current_vals: list[float],
) -> dict[str, float] | None:
    """Compute KS statistic and p-value for numeric columns."""
    if len(baseline_vals) < 2 or len(current_vals) < 2:
        return None

    try:
        from scipy.stats import ks_2samp

        stat, p_value = ks_2samp(baseline_vals, current_vals)
        return {"statistic": round(stat, 6), "p_value": round(p_value, 6)}
    except ImportError:
        pass

    sorted_base = sorted(baseline_vals)
    sorted_cur = sorted(current_vals)
    all_vals = sorted(set(sorted_base + sorted_cur))

    n_base = len(sorted_base)
    n_cur = len(sorted_cur)
    max_diff = 0.0

    base_idx = 0
    cur_idx = 0
    for val in all_vals:
        while base_idx < n_base and sorted_base[base_idx] <= val:
            base_idx += 1
        while cur_idx < n_cur and sorted_cur[cur_idx] <= val:
            cur_idx += 1
        diff = abs(base_idx / n_base - cur_idx / n_cur)
        if diff > max_diff:
            max_diff = diff

    import math

    n_eff = (n_base * n_cur) / (n_base + n_cur)
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * max_diff
    p_value = max(0.0, min(1.0, 2.0 * math.exp(-2.0 * lam * lam)))

    return {"statistic": round(max_diff, 6), "p_value": round(p_value, 6)}


def _extract_numeric_values(scan: DataScan, colname: str) -> list[float] | None:
    """Extract non-null numeric values from a scan's raw data."""
    if scan.nw_data is None:
        return None

    try:
        col = scan.nw_data.get_column(colname)
        vals = col.drop_nulls().to_list()
        return [float(v) for v in vals if v is not None]
    except Exception:
        return None


def _extract_categorical_freqs(scan: DataScan, colname: str) -> dict[str, int] | None:
    """Extract value frequency counts from a scan's raw data."""
    if scan.nw_data is None:
        stats = {s.name: s.val for s in _get_col_profile(scan, colname).statistics}
        freqs = stats.get("freqs")
        if isinstance(freqs, dict):
            return freqs
        return None

    try:
        col = scan.nw_data.get_column(colname).drop_nulls()
        vals = col.to_list()
        freqs: dict[str, int] = {}
        for v in vals:
            key = str(v)
            freqs[key] = freqs.get(key, 0) + 1
        return freqs
    except Exception:
        return None


def _get_col_profile(scan: DataScan, colname: str) -> ColumnProfile:
    for p in scan.profile.column_profiles:
        if p.colname == colname:
            return p
    raise KeyError(colname)


def _is_numeric_coltype(coltype: str) -> bool:
    return _TypeMap.from_dtype(coltype) is _TypeMap.NUMERIC


@dataclass
class _ColumnDiff:
    colname: str
    coltype_baseline: str | None
    coltype_current: str | None
    status: str
    stat_diffs: dict[str, tuple[Any, Any]]
    drift_scores: dict[str, Any]


class DataScanDiff:
    """
    The result of comparing two `DataScan` profiles.

    A `DataScanDiff` is what [`DataScan.compare()`](`pointblank.DataScan.compare`) returns, and
    it isn't meant to be created directly. It records how a table's profile changed between a
    baseline scan and a current one: which columns were added, removed, or changed type, which
    statistics changed and by how much, and how far each column's distribution drifted (see
    [`DataScan.compare()`](`pointblank.DataScan.compare`) for how the drift scores are computed).

    The results can be used in three ways. The attributes and the
    [`has_changes`](`pointblank.DataScanDiff.has_changes`) property suit checks in code, such as
    stopping a pipeline when a column disappears.
    [`to_dict()`](`pointblank.DataScanDiff.to_dict`) gives every difference as plain data, for
    logging or alerting. And [`get_tabular_report()`](`pointblank.DataScanDiff.get_tabular_report`)
    shows the differences as a table, for a person to review.

    Parameters
    ----------
    current
        The newer `DataScan` (the one whose `compare()` method was called).
    baseline
        The older `DataScan` that `current` is compared against.

    Attributes
    ----------
    columns_added
        Column names present in the current scan but not the baseline, sorted alphabetically.
    columns_removed
        Column names present in the baseline but not the current scan, sorted alphabetically.
    columns_type_changed
        Column names whose data type changed between baseline and current.
    column_diffs
        Per-column diff details for all columns that appear in either scan, with the baseline's
        columns first. Each entry has the attributes `colname`, `coltype_baseline` and
        `coltype_current` (`None` for a column missing from that scan), `status` (one of
        `"added"`, `"removed"`, `"type_changed"`, `"changed"`, or `"unchanged"`), `stat_diffs` (a
        dictionary mapping each changed statistic to a `(baseline, current)` tuple), and
        `drift_scores` (a dictionary of drift scores, which might be empty).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    In this example, a table gains a row and a `note` column, and its `score` column changes from
    integers to strings.

    ```{python}
    import pointblank as pb
    import polars as pl

    old = pl.DataFrame({"id": [1, 2, 3], "score": [7, 9, 8]})

    new = pl.DataFrame(
        {"id": [1, 2, 3, 4], "score": ["7", "9", "8", "6"], "note": ["a", None, "b", None]}
    )

    diff = pb.DataScan(new).compare(pb.DataScan(old))

    diff.columns_added, diff.columns_removed, diff.columns_type_changed
    ```

    The schema changes are available directly as lists. For a look at each column, go through
    `column_diffs`:

    ```{python}
    for col in diff.column_diffs:
        print(col.colname, col.status, col.coltype_baseline, "->", col.coltype_current)
    ```

    The `id` column kept its type, but its statistics changed because of the new row. A
    pipeline could stop whenever `columns_removed` or `columns_type_changed` isn't empty, since
    those changes tend to break code that reads the table.

    See Also
    --------
    DataScan : Profile a table, column by column.
    pointblank.compare : Compare two tables row by row.
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
                        drift_scores={},
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
                        drift_scores={},
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

            drift_scores: dict[str, Any] = {}
            if not type_changed:
                coltype = cur_prof.coltype
                if _is_numeric_coltype(coltype):
                    base_vals = _extract_numeric_values(baseline, col_name)
                    cur_vals = _extract_numeric_values(current, col_name)
                    if base_vals is not None and cur_vals is not None:
                        psi = _compute_psi_numeric(base_vals, cur_vals)
                        if psi is not None and psi > 0.0:
                            drift_scores["psi"] = round(psi, 6)
                        ks = _compute_ks_statistic(base_vals, cur_vals)
                        if ks is not None and ks["statistic"] > 0.0:
                            drift_scores["ks_statistic"] = ks["statistic"]
                            drift_scores["ks_p_value"] = ks["p_value"]
                else:
                    base_freqs = _extract_categorical_freqs(baseline, col_name)
                    cur_freqs = _extract_categorical_freqs(current, col_name)
                    if base_freqs is not None and cur_freqs is not None:
                        psi = _compute_psi_categorical(base_freqs, cur_freqs)
                        if psi is not None and psi > 0.0:
                            drift_scores["psi"] = round(psi, 6)

            status = "type_changed" if type_changed else ("changed" if stat_diffs else "unchanged")
            if drift_scores and status == "unchanged":
                status = "changed"
            self.column_diffs.append(
                _ColumnDiff(
                    colname=col_name,
                    coltype_baseline=base_prof.coltype,
                    coltype_current=cur_prof.coltype,
                    status=status,
                    stat_diffs=stat_diffs,
                    drift_scores=drift_scores,
                )
            )

    @property
    def has_changes(self) -> bool:
        """
        Return `True` if any schema or statistical changes were detected.

        This is the quickest check of whether a table changed. It's `True` if a column was
        added, removed, or changed type, or if any statistic or drift score differs between the
        two scans. A change in the row count alone doesn't count, since tables often grow without
        changing in character. Check [`row_count_diff`](`pointblank.DataScanDiff.row_count_diff`)
        for that.

        Returns
        -------
        bool
            Whether the comparison found any schema or statistical changes.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Here, a log of job runs grows from three rows to five, but every status is still `"ok"`.

        ```{python}
        import pointblank as pb
        import polars as pl

        runs_before = pl.DataFrame({"status": ["ok"] * 3})
        runs_after = pl.DataFrame({"status": ["ok"] * 5})

        diff = pb.DataScan(runs_after).compare(pb.DataScan(runs_before))

        diff.has_changes
        ```

        The table has more rows, but nothing about its columns changed (they have the same
        values in the same proportions), so `has_changes` is `False`. A single `"failed"` status
        in the new rows would have changed the number of distinct values and made it `True`.
        """
        return bool(
            self.columns_added
            or self.columns_removed
            or self.columns_type_changed
            or any(d.stat_diffs or d.drift_scores for d in self.column_diffs)
        )

    @property
    def row_count_diff(self) -> tuple[int, int]:
        """
        Return `(baseline_row_count, current_row_count)`.

        The row counts are kept apart from the other differences, since a change in size is
        often expected (a table that collects events grows every day). Having both counts lets
        you decide what change is acceptable, such as a minimum growth rate or a maximum drop.

        Returns
        -------
        tuple[int, int]
            The number of rows in the baseline scan and in the current scan.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's compare two versions of a log of job runs, where the newer one has two more rows.

        ```{python}
        import pointblank as pb
        import polars as pl

        runs_before = pl.DataFrame({"status": ["ok"] * 3})
        runs_after = pl.DataFrame({"status": ["ok"] * 5})

        diff = pb.DataScan(runs_after).compare(pb.DataScan(runs_before))

        baseline_rows, current_rows = diff.row_count_diff

        baseline_rows, current_rows
        ```

        The baseline has three rows and the current scan five. The
        [`has_changes`](`pointblank.DataScanDiff.has_changes`) property is `False` here, so code
        that should also react to a table shrinking (for instance, when a load was incomplete)
        needs to check these counts.
        """
        return (self._baseline.profile.row_count, self._current.profile.row_count)

    def to_dict(self) -> dict[str, Any]:
        """
        Export the comparison results as a dictionary.

        The dictionary has every difference the comparison found, in plain Python data. That
        suits logging, alerting (by sending only the parts that matter), or storing the results
        next to the scans that produced them. Columns without changes are left out, so the
        dictionary only grows with the number of differences.

        Returns
        -------
        dict[str, Any]
            A dictionary with these keys: `"row_count"` (a dictionary with the `"baseline"` and
            `"current"` row counts), `"columns_added"` and `"columns_removed"` (lists of column
            names), `"columns_type_changed"` (a list of dictionaries with `"column"`,
            `"baseline_type"`, and `"current_type"`), `"stat_diffs"` (a dictionary that maps each
            column with changed statistics to a dictionary that maps each statistic to its
            `"baseline"` and `"current"` values), and `"drift_scores"` (a dictionary mapping each
            column that has drift scores to those scores).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's compare two versions of a table of orders. The newer version has one more order, a
        refund for -15.5.

        ```{python}
        import pointblank as pb
        import polars as pl

        orders = pl.DataFrame({"order_id": [1, 2, 3, 4], "amount": [25.0, 40.0, 15.5, 60.0]})

        orders_now = pl.DataFrame(
            {"order_id": [1, 2, 3, 4, 5], "amount": [25.0, 40.0, 15.5, 60.0, -15.5]}
        )

        diff_dict = pb.DataScan(orders_now).compare(pb.DataScan(orders)).to_dict()

        diff_dict["row_count"], diff_dict["columns_added"], diff_dict["columns_removed"]
        ```

        There are no schema changes. The changed statistics of `amount` include its minimum:

        ```{python}
        diff_dict["stat_diffs"]["amount"]["min"]
        ```

        Picking out an entry like this one makes it easy to raise an alert on specific changes
        (here, a negative amount) and ignore others, such as a mean that moves a little as new
        orders come in.
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
            "drift_scores": {
                d.colname: d.drift_scores for d in self.column_diffs if d.drift_scores
            },
        }

    def get_tabular_report(self) -> GT:
        """
        Generate a GT table summarizing the differences between the two scans.

        The report is for a person reviewing the changes: it has one row per column of either
        scan, in the order of the baseline's columns followed by the added ones. The title names
        the two tables (baseline first) and the subtitle gives both row counts. The `Status`
        column says what happened to each column:

        - `Added` or `Removed`: the column is in only one of the scans
        - `Type Changed` or `Type + Stats Changed`: the data type changed (and maybe the statistics
        too)
        - `Stats Changed`: the type is the same, but statistics or drift scores differ
        - `OK`: nothing changed

        The types in the baseline and the current scan come next, then every changed statistic
        with its baseline and current values. If any column has drift scores, a final column shows
        them (PSI, and for numeric columns the KS statistic with its p-value).

        This method needs Polars, whatever the type of the scanned tables.

        Returns
        -------
        GT
            A styled Great Tables report showing schema and statistical drift.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Let's compare a scan of the `small_table` dataset with a scan of an older version, where
        the `f` column only had `"low"` and `"high"` values and there was no `e` column.

        ```{python}
        import pointblank as pb
        import polars as pl

        small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

        older = small_table.filter(pl.col("f") != "mid").drop("e")

        diff = pb.DataScan(small_table, tbl_name="current").compare(
            pb.DataScan(older, tbl_name="older")
        )

        diff.get_tabular_report()
        ```

        The `e` column shows as added, and the other columns have changed statistics, since the
        older version had two fewer rows. The PSI of `f` (about 1.15) comes from the `"mid"`
        values that are only in the current version. The `date_time`, `date`, and `b` columns get
        a similar PSI, but for a different reason: nearly all of their values are distinct, so
        removing any rows changes their value frequencies a lot. The PSI of a non-numeric column
        means most for columns with a few distinct values, like `f`. Like other GT tables, the
        report can be saved as HTML with `write_raw_html()`, to share with others who want to
        review the changes.
        """
        import polars as pl

        rows: list[dict[str, Any]] = []
        has_any_drift = any(d.drift_scores for d in self.column_diffs)

        for diff in self.column_diffs:
            drift_str = ""
            if diff.drift_scores:
                drift_parts: list[str] = []
                if "psi" in diff.drift_scores:
                    drift_parts.append(f"PSI: {diff.drift_scores['psi']:.4f}")
                if "ks_statistic" in diff.drift_scores:
                    drift_parts.append(
                        f"KS: {diff.drift_scores['ks_statistic']:.4f} "
                        f"(p={diff.drift_scores['ks_p_value']:.4f})"
                    )
                drift_str = "<br>".join(drift_parts)

            if diff.status == "added":
                row: dict[str, Any] = {
                    "column": diff.colname,
                    "status": "Added",
                    "type_baseline": "",
                    "type_current": diff.coltype_current or "",
                    "stat_changes": "",
                }
                if has_any_drift:
                    row["drift"] = ""
                rows.append(row)
            elif diff.status == "removed":
                row = {
                    "column": diff.colname,
                    "status": "Removed",
                    "type_baseline": diff.coltype_baseline or "",
                    "type_current": "",
                    "stat_changes": "",
                }
                if has_any_drift:
                    row["drift"] = ""
                rows.append(row)
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

                row = {
                    "column": diff.colname,
                    "status": status,
                    "type_baseline": diff.coltype_baseline or "",
                    "type_current": diff.coltype_current or "",
                    "stat_changes": "<br>".join(change_parts),
                }
                if has_any_drift:
                    row["drift"] = drift_str
                rows.append(row)

        if not rows:
            row = {
                "column": "(no columns)",
                "status": "OK",
                "type_baseline": "",
                "type_current": "",
                "stat_changes": "",
            }
            if has_any_drift:
                row["drift"] = ""
            rows.append(row)

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
                **({"drift": "Drift Scores"} if has_any_drift else {}),
            )
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(
                style=style.text(font=google_font("IBM Plex Mono"), size="11px"),
                locations=loc.body(),
            )
            .fmt_markdown(columns=["stat_changes"] + (["drift"] if has_any_drift else []))
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

    Supported Column Types
    ----------------------
    Numeric columns (integers, floats and decimals), string columns (including categorical and
    enum columns), date, datetime and time columns, and Boolean columns are summarized. Columns of
    other types (e.g., lists, arrays, structs, durations and binary data) can't be summarized, so
    they get a row marked *unsupported type* with no statistics. pandas stores decimals, times,
    and Booleans with missing values as `object` columns, which can't be summarized either.

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

    The `tbl_name=` value appears in the report's title, which helps tell reports apart when
    several tables are summarized. An Ibis table is read into memory before its statistics are
    computed, so a very large table might be better summarized from a sample. To keep the
    summary rather than just display it (to query it, save it, or compare it with a later version
    of the table), create a [`DataScan`](`pointblank.DataScan`) instead, which produces this same
    report through its [`get_tabular_report()`](`pointblank.DataScan.get_tabular_report`) method.
    """

    # Import processing functions from validate module
    from pointblank.validate import _process_data

    # Process input data to handle different data source types
    data = _process_data(data)

    scanner = DataScan(data=data, tbl_name=tbl_name)
    return scanner.get_tabular_report()
