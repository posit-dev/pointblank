"""
Test preprocessing functions for validation serialization examples.

These functions are used to create validation objects that can be serialized
and stored as reference files for regression testing.
"""

import narwhals as nw
import polars as pl


def double_column_a(df):
    """Double the values in column 'a'."""
    return df.with_columns(pl.col("a") * 2)


def add_computed_column(df):
    """Add a computed column based on existing columns."""
    return df.with_columns((pl.col("a") + pl.col("b")).alias("sum_ab"))


def filter_by_d_gt_100(df):
    """Filter rows where column 'd' is greater than 100."""
    return df.filter(pl.col("d") > 100)


def narwhals_median_transform(df):
    """Use narwhals to compute median - cross-backend compatible."""
    return nw.from_native(df).select(nw.median("a"), nw.median("d"))


def complex_preprocessing(df):
    """Complex preprocessing combining multiple operations."""
    return (
        df.filter(pl.col("a") > 1)
        .with_columns((pl.col("a") * 2).alias("a_doubled"), (pl.col("d") / 10).alias("d_scaled"))
        .filter(pl.col("d_scaled") > 10)
    )


def pandas_compatible_transform(df):
    """Transform that works with pandas DataFrames."""
    if hasattr(df, "assign"):  # pandas
        return df.assign(a_plus_b=df["a"] + df.get("b", 0))
    else:  # polars or other
        return df.with_columns((pl.col("a") + pl.col("b")).alias("a_plus_b"))
