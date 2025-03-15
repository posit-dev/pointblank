from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.metadata import version
from math import floor, log10
from typing import Any

import narwhals as nw
from great_tables import GT, google_font, html, loc, style
from great_tables.vals import fmt_integer, fmt_number, fmt_scientific
from narwhals.typing import FrameT

from pointblank._constants import SVG_ICONS_FOR_DATA_TYPES
from pointblank._utils import _get_tbl_type, _select_df_lib
from pointblank._utils_html import _create_table_dims_html, _create_table_type_html

__all__ = ["DataScan", "col_summary_tbl"]


@dataclass
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

    data: FrameT | Any
    tbl_name: str | None = None
    data_alt: Any | None = field(init=False)
    tbl_category: str = field(init=False)
    tbl_type: str = field(init=False)
    profile: dict = field(init=False)

    def __post_init__(self):
        # Determine if the data is a DataFrame that could be handled by Narwhals,
        # or an Ibis Table
        self.tbl_type = _get_tbl_type(data=self.data)
        ibis_tbl = "ibis.expr.types.relations.Table" in str(type(self.data))
        pl_pd_tbl = "polars" in self.tbl_type or "pandas" in self.tbl_type

        # Set the table category based on the type of table (this will be used to determine
        # how to handle the data)
        if ibis_tbl:
            self.tbl_category = "ibis"
        else:
            self.tbl_category = "dataframe"

        # If the data is DataFrame, convert it to a Narwhals DataFrame
        if pl_pd_tbl:
            self.data_alt = nw.from_native(self.data)
        else:
            self.data_alt = None

        # Generate the profile based on the `tbl_category` value
        if self.tbl_category == "dataframe":
            self.profile = self._generate_profile_df()

        if self.tbl_category == "ibis":
            self.profile = self._generate_profile_ibis()

    def _generate_profile_df(self) -> dict:
        profile = {}

        if self.tbl_name:
            profile["tbl_name"] = self.tbl_name

        row_count = self.data_alt.shape[0]
        column_count = self.data_alt.shape[1]

        profile.update(
            {
                "tbl_type": self.tbl_type,
                "dimensions": {"rows": row_count, "columns": column_count},
                "columns": [],
            }
        )

        for idx, column in enumerate(self.data_alt.columns):
            col_data = self.data_alt[column]
            native_dtype = str(self.data[column].dtype)

            #
            # Collection of sample data
            #
            if "date" in str(col_data.dtype).lower():
                sample_data = col_data.drop_nulls().head(5).cast(nw.String).to_list()
                sample_data = [str(x) for x in sample_data]
            else:
                sample_data = col_data.drop_nulls().head(5).to_list()

            n_missing_vals = int(col_data.is_null().sum())
            n_unique_vals = int(col_data.n_unique())

            # If there are missing values, subtract 1 from the number of unique values
            # to account for the missing value which shouldn't be included in the count
            if (n_missing_vals > 0) and (n_unique_vals > 0):
                n_unique_vals = n_unique_vals - 1

            f_missing_vals = _round_to_sig_figs(n_missing_vals / row_count, 3)
            f_unique_vals = _round_to_sig_figs(n_unique_vals / row_count, 3)

            col_profile = {
                "column_name": column,
                "column_type": native_dtype,
                "column_number": idx + 1,
                "n_missing_values": n_missing_vals,
                "f_missing_values": f_missing_vals,
                "n_unique_values": n_unique_vals,
                "f_unique_values": f_unique_vals,
            }

            #
            # Numerical columns
            #
            if "int" in str(col_data.dtype).lower() or "float" in str(col_data.dtype).lower():
                n_negative_vals = int(col_data.is_between(-1e26, -1e-26).sum())
                f_negative_vals = _round_to_sig_figs(n_negative_vals / row_count, 3)

                n_zero_vals = int(col_data.is_between(0, 0).sum())
                f_zero_vals = _round_to_sig_figs(n_zero_vals / row_count, 3)

                n_positive_vals = row_count - n_missing_vals - n_negative_vals - n_zero_vals
                f_positive_vals = _round_to_sig_figs(n_positive_vals / row_count, 3)

                col_profile_additional = {
                    "n_negative_values": n_negative_vals,
                    "f_negative_values": f_negative_vals,
                    "n_zero_values": n_zero_vals,
                    "f_zero_values": f_zero_vals,
                    "n_positive_values": n_positive_vals,
                    "f_positive_values": f_positive_vals,
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                col_profile_stats = {
                    "statistics": {
                        "numerical": {
                            "descriptive": {
                                "mean": round(float(col_data.mean()), 2),
                                "std_dev": round(float(col_data.std()), 4),
                            },
                            "quantiles": {
                                "min": float(col_data.min()),
                                "p05": round(
                                    float(col_data.quantile(0.05, interpolation="linear")), 2
                                ),
                                "q_1": round(
                                    float(col_data.quantile(0.25, interpolation="linear")), 2
                                ),
                                "med": float(col_data.median()),
                                "q_3": round(
                                    float(col_data.quantile(0.75, interpolation="linear")), 2
                                ),
                                "p95": round(
                                    float(col_data.quantile(0.95, interpolation="linear")), 2
                                ),
                                "max": float(col_data.max()),
                                "iqr": round(
                                    float(col_data.quantile(0.75, interpolation="linear"))
                                    - float(col_data.quantile(0.25, interpolation="linear")),
                                    2,
                                ),
                            },
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            #
            # String columns
            #
            elif (
                "string" in str(col_data.dtype).lower()
                or "categorical" in str(col_data.dtype).lower()
            ):
                col_profile_additional = {
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                # Transform `col_data` to a column of string lengths
                col_str_len_data = col_data.str.len_chars()

                col_profile_stats = {
                    "statistics": {
                        "string_lengths": {
                            "descriptive": {
                                "mean": round(float(col_str_len_data.mean()), 2),
                                "std_dev": round(float(col_str_len_data.std()), 4),
                            },
                            "quantiles": {
                                "min": int(col_str_len_data.min()),
                                "p05": int(col_str_len_data.quantile(0.05, interpolation="linear")),
                                "q_1": int(col_str_len_data.quantile(0.25, interpolation="linear")),
                                "med": int(col_str_len_data.median()),
                                "q_3": int(col_str_len_data.quantile(0.75, interpolation="linear")),
                                "p95": int(col_str_len_data.quantile(0.95, interpolation="linear")),
                                "max": int(col_str_len_data.max()),
                                "iqr": int(col_str_len_data.quantile(0.75, interpolation="linear"))
                                - int(col_str_len_data.quantile(0.25, interpolation="linear")),
                            },
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            #
            # Date and datetime columns
            #
            elif "date" in str(col_data.dtype).lower():
                col_profile_additional = {
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                min_date = str(col_data.min())
                max_date = str(col_data.max())

                col_profile_stats = {
                    "statistics": {
                        "datetime": {
                            "min": min_date,
                            "max": max_date,
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            #
            # Boolean columns
            #
            elif "bool" in str(col_data.dtype).lower():
                col_profile_additional = {
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                n_true_values = int(col_data.sum())
                f_true_values = _round_to_sig_figs(n_true_values / row_count, 3)

                n_false_values = row_count - n_missing_vals - n_true_values
                f_false_values = _round_to_sig_figs(n_false_values / row_count, 3)

                col_profile_stats = {
                    "statistics": {
                        "boolean": {
                            "n_true_values": n_true_values,
                            "f_true_values": f_true_values,
                            "n_false_values": n_false_values,
                            "f_false_values": f_false_values,
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            profile["columns"].append(col_profile)

        return profile

    def _generate_profile_ibis(self) -> dict:
        profile = {}

        if self.tbl_name:
            profile["tbl_name"] = self.tbl_name

        from pointblank.validate import get_row_count

        row_count = get_row_count(data=self.data)
        column_count = len(self.data.columns)

        profile.update(
            {
                "tbl_type": self.tbl_type,
                "dimensions": {"rows": row_count, "columns": column_count},
                "columns": [],
            }
        )

        # Determine which DataFrame library is available
        df_lib = _select_df_lib(preference="polars")
        df_lib_str = str(df_lib)

        if "polars" in df_lib_str:
            df_lib_use = "polars"
        else:
            df_lib_use = "pandas"

        column_dtypes = list(self.data.schema().items())

        for idx, column in enumerate(self.data.columns):
            dtype_str = str(column_dtypes[idx][1])

            col_data = self.data[column]
            col_data_no_null = self.data.drop_null().head(5)[column]

            #
            # Collection of sample data
            #
            if "date" in dtype_str.lower() or "timestamp" in dtype_str.lower():
                if df_lib_use == "polars":
                    import polars as pl

                    sample_data = col_data_no_null.to_polars().cast(pl.String).to_list()
                else:
                    sample_data = col_data_no_null.to_pandas().astype(str).to_list()
            else:
                if df_lib_use == "polars":
                    sample_data = col_data_no_null.to_polars().to_list()
                else:
                    sample_data = col_data_no_null.to_pandas().to_list()

            n_missing_vals = int(_to_df_lib(col_data.isnull().sum(), df_lib=df_lib_use))
            n_unique_vals = int(_to_df_lib(col_data.nunique(), df_lib=df_lib_use))

            # If there are missing values, subtract 1 from the number of unique values
            # to account for the missing value which shouldn't be included in the count
            if (n_missing_vals > 0) and (n_unique_vals > 0):
                n_unique_vals = n_unique_vals - 1

            f_missing_vals = _round_to_sig_figs(n_missing_vals / row_count, 3)
            f_unique_vals = _round_to_sig_figs(n_unique_vals / row_count, 3)

            col_profile = {
                "column_name": column,
                "column_type": dtype_str,
                "column_number": idx + 1,
                "n_missing_values": n_missing_vals,
                "f_missing_values": f_missing_vals,
                "n_unique_values": n_unique_vals,
                "f_unique_values": f_unique_vals,
            }

            #
            # Numerical columns
            #
            if "int" in dtype_str.lower() or "float" in dtype_str.lower():
                n_negative_vals = int(
                    _to_df_lib(col_data.between(-1e26, -1e-26).sum(), df_lib=df_lib_use)
                )
                f_negative_vals = _round_to_sig_figs(n_negative_vals / row_count, 3)

                n_zero_vals = int(_to_df_lib(col_data.between(0, 0).sum(), df_lib=df_lib_use))
                f_zero_vals = _round_to_sig_figs(n_zero_vals / row_count, 3)

                n_positive_vals = row_count - n_missing_vals - n_negative_vals - n_zero_vals
                f_positive_vals = _round_to_sig_figs(n_positive_vals / row_count, 3)

                col_profile_additional = {
                    "n_negative_values": n_negative_vals,
                    "f_negative_values": f_negative_vals,
                    "n_zero_values": n_zero_vals,
                    "f_zero_values": f_zero_vals,
                    "n_positive_values": n_positive_vals,
                    "f_positive_values": f_positive_vals,
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                col_profile_stats = {
                    "statistics": {
                        "numerical": {
                            "descriptive": {
                                "mean": round(_to_df_lib(col_data.mean(), df_lib=df_lib_use), 2),
                                "std_dev": round(_to_df_lib(col_data.std(), df_lib=df_lib_use), 4),
                            },
                            "quantiles": {
                                "min": _to_df_lib(col_data.min(), df_lib=df_lib_use),
                                "p05": round(
                                    _to_df_lib(col_data.approx_quantile(0.05), df_lib=df_lib_use),
                                    2,
                                ),
                                "q_1": round(
                                    _to_df_lib(col_data.approx_quantile(0.25), df_lib=df_lib_use),
                                    2,
                                ),
                                "med": _to_df_lib(col_data.median(), df_lib=df_lib_use),
                                "q_3": round(
                                    _to_df_lib(col_data.approx_quantile(0.75), df_lib=df_lib_use),
                                    2,
                                ),
                                "p95": round(
                                    _to_df_lib(col_data.approx_quantile(0.95), df_lib=df_lib_use),
                                    2,
                                ),
                                "max": _to_df_lib(col_data.max(), df_lib=df_lib_use),
                                "iqr": round(
                                    _to_df_lib(col_data.quantile(0.75), df_lib=df_lib_use)
                                    - _to_df_lib(col_data.quantile(0.25), df_lib=df_lib_use),
                                    2,
                                ),
                            },
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            #
            # String columns
            #
            elif "string" in dtype_str.lower() or "char" in dtype_str.lower():
                col_profile_additional = {
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                # Transform `col_data` to a column of string lengths
                col_str_len_data = col_data.length()

                col_profile_stats = {
                    "statistics": {
                        "string_lengths": {
                            "descriptive": {
                                "mean": round(
                                    float(_to_df_lib(col_str_len_data.mean(), df_lib=df_lib_use)), 2
                                ),
                                "std_dev": round(
                                    float(_to_df_lib(col_str_len_data.std(), df_lib=df_lib_use)), 4
                                ),
                            },
                            "quantiles": {
                                "min": int(_to_df_lib(col_str_len_data.min(), df_lib=df_lib_use)),
                                "p05": int(
                                    _to_df_lib(
                                        col_str_len_data.approx_quantile(0.05),
                                        df_lib=df_lib_use,
                                    )
                                ),
                                "q_1": int(
                                    _to_df_lib(
                                        col_str_len_data.approx_quantile(0.25),
                                        df_lib=df_lib_use,
                                    )
                                ),
                                "med": int(
                                    _to_df_lib(col_str_len_data.median(), df_lib=df_lib_use)
                                ),
                                "q_3": int(
                                    _to_df_lib(
                                        col_str_len_data.approx_quantile(0.75),
                                        df_lib=df_lib_use,
                                    )
                                ),
                                "p95": int(
                                    _to_df_lib(
                                        col_str_len_data.approx_quantile(0.95),
                                        df_lib=df_lib_use,
                                    )
                                ),
                                "max": int(_to_df_lib(col_str_len_data.max(), df_lib=df_lib_use)),
                                "iqr": int(
                                    _to_df_lib(
                                        col_str_len_data.approx_quantile(0.75),
                                        df_lib=df_lib_use,
                                    )
                                )
                                - int(
                                    _to_df_lib(
                                        col_str_len_data.approx_quantile(0.25),
                                        df_lib=df_lib_use,
                                    )
                                ),
                            },
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            #
            # Date and datetime columns
            #
            elif "date" in dtype_str.lower() or "timestamp" in dtype_str.lower():
                col_profile_additional = {
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                min_date = _to_df_lib(col_data.min(), df_lib=df_lib_use)
                max_date = _to_df_lib(col_data.max(), df_lib=df_lib_use)

                col_profile_stats = {
                    "statistics": {
                        "datetime": {
                            "min": str(min_date),
                            "max": str(max_date),
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            #
            # Boolean columns
            #
            elif "bool" in dtype_str.lower():
                col_profile_additional = {
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                n_true_values = _to_df_lib(col_data.cast(int).sum(), df_lib=df_lib)
                f_true_values = _round_to_sig_figs(n_true_values / row_count, 3)

                n_false_values = row_count - n_missing_vals - n_true_values
                f_false_values = _round_to_sig_figs(n_false_values / row_count, 3)

                col_profile_stats = {
                    "statistics": {
                        "boolean": {
                            "n_true_values": n_true_values,
                            "f_true_values": f_true_values,
                            "n_false_values": n_false_values,
                            "f_false_values": f_false_values,
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            profile["columns"].append(col_profile)

        return profile

    def _get_column_data(self, column: str) -> dict | None:
        column_data = self.profile["columns"]

        # Find the column in the column data and return the
        for col in column_data:
            if col["column_name"] == column:
                return col

        # If the column is not found, return None
        return None

    def get_tabular_report(self) -> GT:
        column_data = self.profile["columns"]

        tbl_name = self.tbl_name

        stats_list = []
        datetime_row_list = []

        n_rows = self.profile["dimensions"]["rows"]
        n_columns = self.profile["dimensions"]["columns"]

        # Iterate over each column's data and obtain a dictionary of statistics for each column
        for idx, col in enumerate(column_data):
            if "statistics" in col and "numerical" in col["statistics"]:
                col_dict = _process_numerical_column_data(col)
            elif "statistics" in col and "string_lengths" in col["statistics"]:
                col_dict = _process_string_column_data(col)
            elif "statistics" in col and "datetime" in col["statistics"]:
                col_dict = _process_datetime_column_data(col)
                datetime_row_list.append(idx)
            elif "statistics" in col and "boolean" in col["statistics"]:
                col_dict = _process_boolean_column_data(col)
            else:
                col_dict = _process_other_column_data(col)

            stats_list.append(col_dict)

        # Determine which DataFrame library is available and construct the DataFrame
        # based on the available library
        df_lib = _select_df_lib(preference="polars")
        df_lib_str = str(df_lib)

        if "polars" in df_lib_str:
            import polars as pl

            stats_df = pl.DataFrame(stats_list)
        else:
            import pandas as pd

            stats_df = pd.DataFrame(stats_list)

        stats_df = pl.DataFrame(stats_list)

        stat_columns = [
            "missing_vals",
            "unique_vals",
            "mean",
            "std_dev",
            "min",
            "p05",
            "q_1",
            "med",
            "q_3",
            "p95",
            "max",
            "iqr",
        ]

        # Create the label, table type, and thresholds HTML fragments
        table_type_html = _create_table_type_html(
            tbl_type=self.tbl_type, tbl_name=tbl_name, font_size="10px"
        )

        tbl_dims_html = _create_table_dims_html(columns=n_columns, rows=n_rows, font_size="10px")

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

        gt_tbl = (
            GT(stats_df)
            .tab_header(title=html(combined_title))
            .cols_align(align="right", columns=stat_columns)
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(
                style=style.text(font=google_font("IBM Plex Mono")),
                locations=loc.body(),
            )
            .tab_style(
                style=style.text(size="10px"),
                locations=loc.body(columns=stat_columns),
            )
            .tab_style(
                style=style.text(size="14px"),
                locations=loc.body(columns="column_number"),
            )
            .tab_style(
                style=style.text(size="12px"),
                locations=loc.body(columns="column_name"),
            )
            .tab_style(
                style=style.css("white-space: pre; overflow-x: visible;"),
                locations=loc.body(columns="min"),
            )
            .tab_style(
                style=style.borders(sides="left", color="#D3D3D3", style="solid"),
                locations=loc.body(columns=["missing_vals", "mean", "iqr"]),
            )
            .tab_style(
                style=style.borders(sides="left", color="#E5E5E5", style="dashed"),
                locations=loc.body(
                    columns=["std_dev", "min", "p05", "q_1", "med", "q_3", "p95", "max"]
                ),
            )
            .tab_style(
                style=style.borders(sides="left", style="none"),
                locations=loc.body(
                    columns=["p05", "q_1", "med", "q_3", "p95", "max"],
                    rows=datetime_row_list,
                ),
            )
            .tab_style(
                style=style.fill(color="#FCFCFC"),
                locations=loc.body(columns=["missing_vals", "unique_vals", "iqr"]),
            )
            .tab_style(
                style=style.text(align="center"), locations=loc.column_labels(columns=stat_columns)
            )
            .cols_label(
                column_number="",
                icon="",
                column_name="Column",
                missing_vals="NAs",
                unique_vals="Uniq.",
                mean="Mean",
                std_dev="SD",
                min="Min",
                p05="P05",
                q_1="Q1",
                med="Med",
                q_3="Q3",
                p95="P95",
                max="Max",
                iqr="IQR",
            )
            .cols_width(
                column_number="40px",
                icon="35px",
                column_name="200px",
                missing_vals="50px",
                unique_vals="50px",
                mean="50px",
                std_dev="50px",
                min="50px",
                p05="50px",
                q_1="50px",
                med="50px",
                q_3="50px",
                p95="50px",
                max="50px",
                iqr="50px",  # 875 px total
            )
        )

        # If the version of `great_tables` is `>=0.17.0` then disable Quarto table processing
        if version("great_tables") >= "0.17.0":
            gt_tbl = gt_tbl.tab_options(quarto_disable_processing=True)

        return gt_tbl

    def to_dict(self) -> dict:
        return self.profile

    def to_json(self) -> str:
        return json.dumps(self.profile, indent=4)

    def save_to_json(self, output_file: str):
        with open(output_file, "w") as f:
            json.dump(self.profile, f, indent=4)


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


def _to_df_lib(expr: any, df_lib: str) -> any:
    if df_lib == "polars":
        return expr.to_polars()
    else:
        return expr.to_pandas()


def _round_to_sig_figs(value: float, sig_figs: int) -> float:
    if value == 0:
        return 0
    return round(value, sig_figs - int(floor(log10(abs(value)))) - 1)


def _compact_integer_fmt(value: float | int) -> str:
    if value == 0:
        formatted = "0"
    elif abs(value) >= 1 and abs(value) < 10_000:
        formatted = fmt_integer(value, use_seps=False)[0]
    else:
        formatted = fmt_scientific(value, decimals=1, exp_style="E1")[0]

    return formatted


def _compact_decimal_fmt(value: float | int) -> str:
    if value == 0:
        formatted = "0.00"
    elif abs(value) < 1 and abs(value) >= 0.01:
        formatted = fmt_number(value, decimals=2)[0]
    elif abs(value) < 0.01:
        formatted = fmt_scientific(value, decimals=1, exp_style="E1")[0]
    elif abs(value) >= 1 and abs(value) < 1000:
        formatted = fmt_number(value, n_sigfig=3)[0]
    elif abs(value) >= 1000 and abs(value) < 10_000:
        formatted = fmt_number(value, decimals=0, use_seps=False)[0]
    else:
        formatted = fmt_scientific(value, decimals=1, exp_style="E1")[0]

    return formatted


def _compact_0_1_fmt(value: float | int) -> str:
    if value == 0:
        formatted = " 0.00"
    elif value == 1:
        formatted = " 1.00"
    elif abs(value) < 1 and abs(value) >= 0.01:
        formatted = " " + fmt_number(value, decimals=2)[0]
    elif abs(value) < 0.01:
        formatted = "<0.01"
    elif abs(value) > 0.99:
        formatted = ">0.99"
    else:
        formatted = fmt_number(value, n_sigfig=3)[0]

    return formatted


def _process_numerical_column_data(column_data: dict) -> dict:
    column_number = column_data["column_number"]
    column_name = column_data["column_name"]
    column_type = column_data["column_type"]

    column_name_and_type = (
        f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{column_name}</div>"
        f"<div style='font-size: 11px; color: gray;'>{column_type}</div>"
    )

    # Get the Missing and Unique value counts and fractions
    missing_vals = column_data["n_missing_values"]
    unique_vals = column_data["n_unique_values"]
    missing_vals_frac = _compact_decimal_fmt(column_data["f_missing_values"])
    unique_vals_frac = _compact_decimal_fmt(column_data["f_unique_values"])

    missing_vals_str = f"{missing_vals}<br>{missing_vals_frac}"
    unique_vals_str = f"{unique_vals}<br>{unique_vals_frac}"

    # Get the descriptive and quantile statistics
    descriptive_stats = column_data["statistics"]["numerical"]["descriptive"]
    quantile_stats = column_data["statistics"]["numerical"]["quantiles"]

    # Get all values from the descriptive and quantile stats into a single list
    quantile_stats_vals = [v[1] for v in quantile_stats.items()]

    # Determine if the quantile stats are all integerlike
    integerlike = []

    # Determine if the quantile stats are integerlike
    for val in quantile_stats_vals:
        # Check if a quantile value is a number and then if it is intergerlike
        if not isinstance(val, (int, float)):
            continue
        else:
            integerlike.append(val % 1 == 0)
    quantile_vals_integerlike = all(integerlike)

    # Determine the formatter to use for the quantile values
    if quantile_vals_integerlike:
        q_formatter = _compact_integer_fmt
    else:
        q_formatter = _compact_decimal_fmt

    # Format the descriptive statistics (mean and standard deviation)
    for key, value in descriptive_stats.items():
        descriptive_stats[key] = _compact_decimal_fmt(value=value)

    # Format the quantile statistics
    for key, value in quantile_stats.items():
        quantile_stats[key] = q_formatter(value=value)

    # Create a single dictionary with the statistics for the column
    stats_dict = {
        "column_number": column_number,
        "icon": SVG_ICONS_FOR_DATA_TYPES["numeric"],
        "column_name": column_name_and_type,
        "missing_vals": missing_vals_str,
        "unique_vals": unique_vals_str,
        **descriptive_stats,
        **quantile_stats,
    }

    return stats_dict


def _process_string_column_data(column_data: dict) -> dict:
    column_number = column_data["column_number"]
    column_name = column_data["column_name"]
    column_type = column_data["column_type"]

    column_name_and_type = (
        f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{column_name}</div>"
        f"<div style='font-size: 11px; color: gray;'>{column_type}</div>"
    )

    # Get the Missing and Unique value counts and fractions
    missing_vals = column_data["n_missing_values"]
    unique_vals = column_data["n_unique_values"]
    missing_vals_frac = _compact_decimal_fmt(column_data["f_missing_values"])
    unique_vals_frac = _compact_decimal_fmt(column_data["f_unique_values"])

    missing_vals_str = f"{missing_vals}<br>{missing_vals_frac}"
    unique_vals_str = f"{unique_vals}<br>{unique_vals_frac}"

    # Get the descriptive and quantile statistics
    descriptive_stats = column_data["statistics"]["string_lengths"]["descriptive"]
    quantile_stats = column_data["statistics"]["string_lengths"]["quantiles"]

    # Get all values from the descriptive and quantile stats into a single list
    quantile_stats_vals = [v[1] for v in quantile_stats.items()]

    # Determine if the quantile stats are all integerlike
    integerlike = []

    # Determine if the quantile stats are integerlike
    for val in quantile_stats_vals:
        # Check if a quantile value is a number and then if it is intergerlike
        if not isinstance(val, (int, float)):
            continue
        else:
            integerlike.append(val % 1 == 0)
    quantile_vals_integerlike = all(integerlike)

    # Determine the formatter to use for the quantile values
    if quantile_vals_integerlike:
        q_formatter = _compact_integer_fmt
    else:
        q_formatter = _compact_decimal_fmt

    # Format the descriptive statistics (mean and standard deviation)
    for key, value in descriptive_stats.items():
        formatted_val = _compact_decimal_fmt(value=value)
        descriptive_stats[key] = (
            f'<div><div>{formatted_val}</div><div style="float: left; position: absolute;"><div title="String Lengths" style="font-size: 7px; color: #999; font-style: italic;">SL</div></div></div>'
        )

    # Format the quantile statistics
    for key, value in quantile_stats.items():
        formatted_val = q_formatter(value=value)
        quantile_stats[key] = (
            f'<div><div>{formatted_val}</div><div style="float: left; position: absolute;"><div title="String Lengths" style="font-size: 7px; color: #999; font-style: italic;">SL</div></div></div>'
        )

    # Create a single dictionary with the statistics for the column
    stats_dict = {
        "column_number": column_number,
        "icon": SVG_ICONS_FOR_DATA_TYPES["string"],
        "column_name": column_name_and_type,
        "missing_vals": missing_vals_str,
        "unique_vals": unique_vals_str,
        **descriptive_stats,
        "min": quantile_stats["min"],
        "p05": "&mdash;",
        "q_1": "&mdash;",
        "med": quantile_stats["med"],
        "q_3": "&mdash;",
        "p95": "&mdash;",
        "max": quantile_stats["max"],
        "iqr": quantile_stats["iqr"],
    }

    return stats_dict


def _process_datetime_column_data(column_data: dict) -> dict:
    column_number = column_data["column_number"]
    column_name = column_data["column_name"]
    column_type = column_data["column_type"]

    long_column_type = len(column_type) > 22

    if long_column_type:
        column_type_style = "font-size: 7.5px; color: gray; margin-top: 3px; margin-bottom: 2px;"
    else:
        column_type_style = "font-size: 11px; color: gray;"

    column_name_and_type = (
        f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{column_name}</div>"
        f"<div style='{column_type_style}'>{column_type}</div>"
    )

    # Get the Missing and Unique value counts and fractions
    missing_vals = column_data["n_missing_values"]
    unique_vals = column_data["n_unique_values"]
    missing_vals_frac = _compact_decimal_fmt(column_data["f_missing_values"])
    unique_vals_frac = _compact_decimal_fmt(column_data["f_unique_values"])

    missing_vals_str = f"{missing_vals}<br>{missing_vals_frac}"
    unique_vals_str = f"{unique_vals}<br>{unique_vals_frac}"

    # Get the min and max date
    min_date = column_data["statistics"]["datetime"]["min"]
    max_date = column_data["statistics"]["datetime"]["max"]

    # Format the dates so that they don't break across lines
    min_max_date_str = f"<span style='text-align: left; white-space: nowrap; overflow-x: visible;'>&nbsp;{min_date} &ndash; {max_date}</span>"

    # Create a single dictionary with the statistics for the column
    stats_dict = {
        "column_number": column_number,
        "icon": SVG_ICONS_FOR_DATA_TYPES["date"],
        "column_name": column_name_and_type,
        "missing_vals": missing_vals_str,
        "unique_vals": unique_vals_str,
        "mean": "&mdash;",
        "std_dev": "&mdash;",
        "min": min_max_date_str,
        "p05": "",
        "q_1": "",
        "med": "",
        "q_3": "",
        "p95": "",
        "max": "",
        "iqr": "&mdash;",
    }

    return stats_dict


def _process_boolean_column_data(column_data: dict) -> dict:
    column_number = column_data["column_number"]
    column_name = column_data["column_name"]
    column_type = column_data["column_type"]

    column_name_and_type = (
        f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{column_name}</div>"
        f"<div style='font-size: 11px; color: gray;'>{column_type}</div>"
    )

    # Get the Missing and Unique value counts and fractions
    missing_vals = column_data["n_missing_values"]
    missing_vals_frac = _compact_decimal_fmt(column_data["f_missing_values"])
    missing_vals_str = f"{missing_vals}<br>{missing_vals_frac}"

    # Get the fractions of True and False values
    f_true_values = column_data["statistics"]["boolean"]["f_true_values"]
    f_false_values = column_data["statistics"]["boolean"]["f_false_values"]

    true_vals_frac_fmt = _compact_0_1_fmt(f_true_values)
    false_vals_frac_fmt = _compact_0_1_fmt(f_false_values)

    # Create an HTML string that combines fractions for the True and False values
    true_false_vals_str = f"<span style='font-weight: bold;'>T</span>{true_vals_frac_fmt}<br><span style='font-weight: bold;'>F</span>{false_vals_frac_fmt}"

    # unique_vals_str = f"{unique_vals}<br>{unique_vals_frac}"

    # Create a single dictionary with the statistics for the column
    stats_dict = {
        "column_number": column_number,
        "icon": SVG_ICONS_FOR_DATA_TYPES["boolean"],
        "column_name": column_name_and_type,
        "missing_vals": missing_vals_str,
        "unique_vals": true_false_vals_str,
        "mean": "&mdash;",
        "std_dev": "&mdash;",
        "min": "&mdash;",
        "p05": "&mdash;",
        "q_1": "&mdash;",
        "med": "&mdash;",
        "q_3": "&mdash;",
        "p95": "&mdash;",
        "max": "&mdash;",
        "iqr": "&mdash;",
    }

    return stats_dict


def _process_other_column_data(column_data: dict) -> dict:
    column_number = column_data["column_number"]
    column_name = column_data["column_name"]
    column_type = column_data["column_type"]

    column_name_and_type = (
        f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{column_name}</div>"
        f"<div style='font-size: 11px; color: gray;'>{column_type}</div>"
    )

    # Get the Missing and Unique value counts and fractions
    missing_vals = column_data["n_missing_values"]
    unique_vals = column_data["n_unique_values"]
    missing_vals_frac = _compact_decimal_fmt(column_data["f_missing_values"])
    unique_vals_frac = _compact_decimal_fmt(column_data["f_unique_values"])

    missing_vals_str = f"{missing_vals}<br>{missing_vals_frac}"
    unique_vals_str = f"{unique_vals}<br>{unique_vals_frac}"

    # Create a single dictionary with the statistics for the column
    stats_dict = {
        "column_number": column_number,
        "icon": SVG_ICONS_FOR_DATA_TYPES["object"],
        "column_name": column_name_and_type,
        "missing_vals": missing_vals_str,
        "unique_vals": unique_vals_str,
        "mean": "&mdash;",
        "std_dev": "&mdash;",
        "min": "&mdash;",
        "p05": "&mdash;",
        "q_1": "&mdash;",
        "med": "&mdash;",
        "q_3": "&mdash;",
        "p95": "&mdash;",
        "max": "&mdash;",
        "iqr": "&mdash;",
    }

    return stats_dict
