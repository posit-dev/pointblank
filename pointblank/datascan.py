from __future__ import annotations

import json
from dataclasses import dataclass, field

import narwhals as nw
from narwhals.typing import FrameT
from typing import Any

from pointblank._utils import _get_tbl_type, _select_df_lib


__all__ = [
    "DataScan",
]


@dataclass
class DataScan:
    """
    Get a summary of a DataFrame or Ibis Table through a scan.

    Parameters
    ----------
    data
        The data to scan and summarize.
    tbl_name
        Optionally, the name of the table could be provided as `tbl_name`.

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
            self.profile = self._generate_profile()

        if self.tbl_category == "ibis":
            self.profile = self._generate_profile_ibis()

    def _generate_profile_ibis(self) -> dict:

        profile = {}

        if self.tbl_name:
            profile["tbl_name"] = self.tbl_name

        from pointblank.validate import get_row_count

        profile.update(
            {
                "tbl_type": self.tbl_type,
                "dimensions": {"rows": get_row_count(self.data), "columns": len(self.data.columns)},
                "columns": [],
            }
        )

        # Determine which DataFrame library is available
        df_lib = _select_df_lib(preference="polars")

        column_dtypes = list(self.data.schema().items())

        for idx, column in enumerate(self.data.columns):

            if df_lib == "polars":
                import polars as pl

            dtype_str = str(column_dtypes[idx][1])

            col_data = self.data[column]
            col_data_no_null = self.data.drop_null().head(5)[column]

            if "date" in dtype_str.lower() or "timestamp" in dtype_str.lower():

                if df_lib == "polars":
                    sample_data = col_data_no_null.to_polars().cast(pl.String).to_list()
                else:
                    sample_data = col_data_no_null.to_pandas().astype(str).to_list()
            else:
                if df_lib == "polars":
                    sample_data = col_data_no_null.to_polars().to_list()
                else:
                    sample_data = col_data_no_null.to_pandas().to_list()

            if df_lib == "polars":
                col_profile = {
                    "column_name": column,
                    "column_type": dtype_str,
                    "missing_values": col_data.isnull().sum().to_polars(),
                    "unique_values": col_data.nunique().to_polars(),
                    "sample_data": sample_data,
                    "statistics": {},
                }
            else:
                col_profile = {
                    "column_name": column,
                    "column_type": dtype_str,
                    "missing_values": col_data.isnull().sum().to_pandas(),
                    "unique_values": col_data.nunique().to_pandas(),
                    "sample_data": sample_data,
                    "statistics": {},
                }

            if "int" in dtype_str.lower() or "float" in dtype_str.lower():

                if df_lib == "polars":
                    col_profile["statistics"]["numerical"] = {
                        "mean": col_data.mean().to_polars(),
                        "median": col_data.median().to_polars(),
                        "std_dev": col_data.std().to_polars(),
                        "min": col_data.min().to_polars(),
                        "max": col_data.max().to_polars(),
                        "percentiles": {
                            "25th": col_data.approx_quantile(0.25).to_polars(),
                            "50th": col_data.approx_quantile(0.50).to_polars(),
                            "75th": col_data.approx_quantile(0.75).to_polars(),
                        },
                    }

                else:
                    col_profile["statistics"]["numerical"] = {
                        "mean": col_data.mean().to_pandas(),
                        "median": col_data.median().to_pandas(),
                        "std_dev": col_data.std().to_pandas(),
                        "min": col_data.min().to_pandas(),
                        "max": col_data.max().to_pandas(),
                        "percentiles": {
                            "25th": col_data.approx_quantile(0.25).to_pandas(),
                            "50th": col_data.approx_quantile(0.50).to_pandas(),
                            "75th": col_data.approx_quantile(0.75).to_pandas(),
                        },
                    }

            elif "string" in dtype_str.lower() or "char" in dtype_str.lower():

                if df_lib == "polars":

                    col_profile["statistics"]["string"] = {
                        "mode": col_data.mode().to_polars(),
                    }

                else:

                    col_profile["statistics"]["string"] = {
                        "mode": col_data.mode().to_pandas(),
                    }

            elif "date" in dtype_str.lower() or "timestamp" in dtype_str.lower():

                if df_lib == "polars":
                    min_date = col_data.min().to_polars()
                    max_date = col_data.max().to_polars()
                else:
                    min_date = col_data.min().to_pandas()
                    max_date = col_data.max().to_pandas()

                col_profile["statistics"]["datetime"] = {
                    "min_date": str(min_date),
                    "max_date": str(max_date),
                }

            profile["columns"].append(col_profile)

        return profile

    def _generate_profile(self) -> dict:

        profile = {}

        if self.tbl_name:
            profile["tbl_name"] = self.tbl_name

        profile.update(
            {
                "tbl_type": self.tbl_type,
                "dimensions": {"rows": self.data_alt.shape[0], "columns": self.data_alt.shape[1]},
                "columns": [],
            }
        )

        for column in self.data_alt.columns:

            col_data = self.data_alt[column]
            native_dtype = str(self.data[column].dtype)

            if "date" in str(col_data.dtype).lower():
                sample_data = col_data.drop_nulls().head(5).cast(nw.String).to_list()
                sample_data = [str(x) for x in sample_data]
            else:
                sample_data = col_data.drop_nulls().head(5).to_list()

            col_profile = {
                "column_name": column,
                "column_type": native_dtype,
                "missing_values": int(col_data.is_null().sum()),
                "unique_values": col_data.n_unique(),
                "sample_data": sample_data,
                "statistics": {},
            }

            if "int" in str(col_data.dtype).lower() or "float" in str(col_data.dtype).lower():
                col_profile["statistics"]["numerical"] = {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std_dev": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "percentiles": {
                        "25th": float(col_data.quantile(0.25, interpolation="nearest")),
                        "50th": float(col_data.quantile(0.50, interpolation="nearest")),
                        "75th": float(col_data.quantile(0.75, interpolation="nearest")),
                    },
                }

            elif (
                "string" in str(col_data.dtype).lower()
                or "categorical" in str(col_data.dtype).lower()
            ):
                mode = col_data.mode()
                col_profile["statistics"]["string"] = {
                    "mode": mode[0],
                }

            elif "date" in str(col_data.dtype).lower():

                min_date = str(col_data.min())
                max_date = str(col_data.max())

                col_profile["statistics"]["datetime"] = {
                    "min_date": min_date,
                    "max_date": max_date,
                }

            profile["columns"].append(col_profile)

        return profile

    def get_profile(self) -> dict:
        return self.profile

    def to_json(self) -> str:
        return json.dumps(self.profile, indent=4)

    def save_to_json(self, output_file: str):
        with open(output_file, "w") as f:
            json.dump(self.profile, f, indent=4)
