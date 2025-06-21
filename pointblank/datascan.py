from __future__ import annotations

import contextlib
import json
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import narwhals as nw
from great_tables import GT, google_font, html, loc, style
from narwhals.dataframe import LazyFrame
from narwhals.typing import FrameT

from pointblank._utils_html import _create_table_dims_html, _create_table_type_html, _fmt_frac
from pointblank.scan_profile import ColumnProfile, _as_physical, _DataProfile, _TypeMap
from pointblank.scan_profile_stats import COLUMN_ORDER_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from narwhals.dataframe import DataFrame
    from narwhals.typing import Frame, IntoFrameT

    from pointblank.scan_profile_stats import StatGroup


__all__ = ["DataScan", "col_summary_tbl"]


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
        a CSV file path, a Parquet file path, or a database connection string.
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
    def __init__(self, data: IntoFrameT, tbl_name: str | None = None) -> None:
        # Import processing functions from validate module
        from pointblank.validate import (
            _process_connection_string,
            _process_csv_input,
            _process_parquet_input,
        )

        # Process input data to handle different data source types
        # Handle connection string input (e.g., "duckdb:///path/to/file.ddb::table_name")
        data = _process_connection_string(data)

        # Handle CSV file input (e.g., "data.csv" or Path("data.csv"))
        data = _process_csv_input(data)

        # Handle Parquet file input (e.g., "data.parquet", "data/*.parquet", "data/")
        data = _process_parquet_input(data)

        as_native = nw.from_native(data)

        if as_native.implementation.name == "IBIS" and as_native._level == "lazy":
            assert isinstance(as_native, LazyFrame)  # help mypy

            ibis_native = as_native.to_native()

            valid_conversion_methods = ("to_pyarrow", "to_pandas", "to_polars")
            for conv_method in valid_conversion_methods:
                try:
                    valid_native = getattr(ibis_native, conv_method)()
                except (NotImplementedError, ImportError, ModuleNotFoundError):
                    continue
                break
            else:
                msg = (
                    "To use `ibis` as input, you must have one of arrow, pandas, polars or numpy "
                    "available in the process. Until `ibis` is fully supported by Narwhals, this is "
                    "necessary. Additionally, the data must be collected in order to calculate some "
                    "structural statistics, which may be performance detrimental."
                )
                raise ImportError(msg)
            as_native = nw.from_native(valid_native)

        self.nw_data: Frame = nw.from_native(as_native)

        self.tbl_name: str | None = tbl_name
        self.profile: _DataProfile = self._generate_profile_df()

    def _generate_profile_df(self) -> _DataProfile:
        columns: list[str] = self.nw_data.columns

        profile = _DataProfile(
            table_name=self.tbl_name,
            columns=columns,
            implementation=self.nw_data.implementation,
        )
        schema: Mapping[str, Any] = self.nw_data.schema
        for column in columns:
            col_data: DataFrame = self.nw_data.select(column)

            ## Handle dtyping:
            native_dtype = schema[column]
            if _TypeMap.is_illegal(native_dtype):
                continue
            try:
                prof: type[ColumnProfile] = _TypeMap.fetch_profile(native_dtype)
            except NotImplementedError:
                continue

            col_profile = ColumnProfile(colname=column, coltype=native_dtype)

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
    def summary_data(self) -> IntoFrameT:
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
        for _fmt_col in ("__frac_n_unique", "__frac_n_missing"):
            _formatted: list[str | None] = _fmt_frac(formatted_data[_fmt_col])
            formatted: nw.Series = nw.new_series(
                _fmt_col, values=_formatted, backend=self.profile.implementation
            )
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
                    name="__freq_true", values=trues, backend=self.profile.implementation
                )
                false_ser: nw.Series = nw.new_series(
                    name="__freq_false", values=falses, backend=self.profile.implementation
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
                formatted = nw.new_series(
                    name=_fmt_col, values=_formatted, backend=self.profile.implementation
                )
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
                icon="35px", colname="200px", **{stat_col: "60px" for stat_col in present_stat_cols}
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

    def to_json(self) -> str:
        prof_dict = self.profile.as_dataframe(strict=False).to_dict(as_series=False)

        return json.dumps(prof_dict, indent=4, default=str)

    def save_to_json(self, output_file: str):
        json_string: str = self.to_json()
        with open(output_file, "w") as f:
            json.dump(json_string, f, indent=4)


def col_summary_tbl(data: FrameT | Any, tbl_name: str | None = None) -> GT:
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
    from pointblank.validate import (
        _process_connection_string,
        _process_csv_input,
        _process_parquet_input,
    )

    # Process input data to handle different data source types
    # Handle connection string input (e.g., "duckdb:///path/to/file.ddb::table_name")
    data = _process_connection_string(data)

    # Handle CSV file input (e.g., "data.csv" or Path("data.csv"))
    data = _process_csv_input(data)

    # Handle Parquet file input (e.g., "data.parquet", "data/*.parquet", "data/")
    data = _process_parquet_input(data)

    scanner = DataScan(data=data, tbl_name=tbl_name)
    return scanner.get_tabular_report()
