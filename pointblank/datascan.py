from __future__ import annotations

import json
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import narwhals as nw
from great_tables import GT, google_font, html, loc, style
from narwhals.typing import FrameT

from pointblank._utils_html import _create_table_dims_html, _create_table_type_html
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
        The data to scan and summarize.
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
        self.nw_data: Frame = nw.from_native(data)
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
        formatted_data = (
            data.with_columns(
                colname=nw.concat_str(
                    nw.lit(
                        "<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>"
                    ),
                    nw.col("colname"),
                    nw.lit("</div><div style='font-size: 11px; color: gray;'>"),
                    nw.col("coltype"),
                    nw.lit("</div>"),
                ),
                # TODO: This is a very temporary solution
                __frac_n_unique=(nw.col("n_unique") / nw.lit(self.profile.row_count)).round(2),
                __frac_n_missing=(nw.col("n_missing") / nw.lit(self.profile.row_count)).round(2),
            )
            .with_columns(
                n_unique=nw.concat_str(
                    nw.col("n_unique"), nw.lit("<br>"), nw.col("__frac_n_unique")
                ),
                n_missing=nw.concat_str(
                    nw.col("n_missing"), nw.lit("<br>"), nw.col("__frac_n_missing")
                ),
            )
            .drop("__frac_n_unique", "__frac_n_missing", "coltype")
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
            .cols_align(align="right", columns=list(present_stat_cols))
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(
                style=style.text(font=google_font("IBM Plex Mono")),
                locations=loc.body(),
            )
            ## Order
            .cols_move_to_start(target_order)
            ## Labeling
            .cols_label(label_map)
            .cols_label(icon="", colname="Column")
            ## Value Formatting
            .fmt_integer(columns=fmt_int)
            .fmt_number(columns=fmt_float, decimals=2)
            .sub_missing(missing_text="-")
            ## Generic Styling
            .tab_style(
                style=style.text(size="10px"),
                locations=loc.body(columns=list(present_stat_cols)),
            )
            .tab_style(
                style=style.text(size="12px"),
                locations=loc.body(columns="colname"),
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
            .cols_width(
                icon="35px", colname="200px", **{stat_col: "50px" for stat_col in present_stat_cols}
            )
        )

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
        The table to summarize, which could be a DataFrame object or an Ibis table object. Read the
        *Supported Input Table Types* section for details on the supported table types.
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

    small_table_polars = pb.load_dataset(dataset="small_table", tbl_type="polars")

    pb.col_summary_tbl(data=small_table_polars)
    ```

    This table used above was a Polars DataFrame, but the `col_summary_tbl()` function works with
    any table supported by `pointblank`, including Pandas DataFrames and Ibis backend tables.
    Here's an example using a DuckDB table handled by Ibis:

    ```{python}
    small_table_duckdb = pb.load_dataset(dataset="nycflights", tbl_type="duckdb")

    pb.col_summary_tbl(data=small_table_duckdb, tbl_name="nycflights")
    ```
    """

    scanner = DataScan(data=data, tbl_name=tbl_name)
    return scanner.get_tabular_report()
