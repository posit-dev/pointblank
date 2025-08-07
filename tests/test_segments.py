from hypothesis import given, strategies as st
import pytest

from pointblank.validate import (
    Validate,
    load_dataset,
    _seg_expr_from_string,
    _seg_expr_from_tuple,
    _apply_segments,
)
from pointblank.segments import (
    Segment,
    seg_group,
)

import pandas as pd
import polars as pl
import ibis


def test_segment_class_fails_not_list():
    with pytest.raises(TypeError, match="Segments must be lists"):
        Segment((1, 2, 3))


def test_segment_class_fails_not_nested_lists():
    with pytest.raises(TypeError, match="Sub-segments must be lists."):
        Segment([1, 2, 3])


def test_segment_class():
    seg1 = Segment([[1, 2, 3]])
    seg2 = Segment([[1, 2, 3], [4, 5]])
    assert seg1.segments == [[1, 2, 3]]
    assert seg2.segments == [[1, 2, 3], [4, 5]]


def test_seg_group():
    seg1 = seg_group([1, 2, 3])
    seg2 = seg_group([[1, 2, 3], [4, 5]])
    assert seg1 == Segment([[1, 2, 3]])
    assert seg2 == Segment([[1, 2, 3], [4, 5]])


# TODO: expand to all tbl_types
@pytest.mark.parametrize("tbl_type", ["pandas", "polars"])
def test_segments_str(tbl_type):
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments="f",
        )
        .interrogate()
    )
    assert validation.n_passed(i=1, scalar=True) == 6
    assert validation.n_passed(i=2, scalar=True) == 5
    assert validation.n_passed(i=3, scalar=True) == 2


# TODO: expand to all tbl_types
@pytest.mark.parametrize("tbl_type", ["pandas", "polars"])
def test_segments_tuple_with_single_value(tbl_type):
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments=("f", "low"),
        )
        .interrogate()
    )
    assert validation.n_passed(i=1, scalar=True) == 5


# TODO: expand to all tbl_types
@pytest.mark.parametrize("tbl_type", ["pandas", "polars"])
def test_segments_tuple_with_multiple_values(tbl_type):
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments=("f", ["low", "high"]),
        )
        .interrogate()
    )
    assert validation.n_passed(i=1, scalar=True) == 5
    assert validation.n_passed(i=2, scalar=True) == 6


# TODO: expand to all tbl_types
@pytest.mark.parametrize("tbl_type", ["pandas", "polars"])
def test_segments_with_multiple_criteria(tbl_type):
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments=["f", ("a", [1, 2])],
        )
        .interrogate()
    )
    assert validation.n_passed(i=1, scalar=True) == 6
    assert validation.n_passed(i=2, scalar=True) == 5
    assert validation.n_passed(i=3, scalar=True) == 2
    assert validation.n_passed(i=4, scalar=True) == 1
    assert validation.n_passed(i=5, scalar=True) == 3


# TODO: expand to all tbl_types
@pytest.mark.parametrize("tbl_type", ["pandas", "polars"])
def test_segments_with_multiple_criteria_2(tbl_type):
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments=[("a", [1, 2]), ("f", ("low", "high"))],
        )
        .interrogate()
    )
    assert validation.n_passed(i=1, scalar=True) == 1
    assert validation.n_passed(i=2, scalar=True) == 3
    assert validation.n_passed(i=3, scalar=True) == 5
    assert validation.n_passed(i=4, scalar=True) == 6


# TODO: use a dataframe agnostic way to handle datetime segments
@pytest.mark.parametrize("tbl_type", ["polars"])
def test_segments_with_dates(tbl_type):
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments=("date", (pl.datetime(2016, 1, 4), pl.datetime(2016, 1, 5))),
        )
        .interrogate()
    )
    assert validation.n_passed(i=1, scalar=True) == 2
    assert validation.n_passed(i=2, scalar=True) == 1


# TODO: need to expand with different lambda functions depending on tbl_type
@pytest.mark.parametrize("tbl_type", ["polars"])
def test_segments_with_preprocessing(tbl_type):
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            pre=lambda df: df.with_columns(
                d_category=pl.when(pl.col("d") > 150).then(pl.lit("high")).otherwise(pl.lit("low"))
            ),
            segments="d_category",
        )
        .interrogate()
    )
    assert validation.n_passed(i=1, scalar=True) == 12
    assert validation.n_passed(i=2, scalar=True) == 1


# TODO: expand to all tbl_types
@pytest.mark.parametrize("tbl_type", ["pandas", "polars"])
def test_segments_with_null_values(tbl_type):
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments=("c", None),
        )
        .interrogate()
    )
    assert validation.n_passed(i=1, scalar=True) == 2


# TODO: expand to all tbl_types
@pytest.mark.parametrize("tbl_type", ["pandas", "polars"])
def test_segments_with_seg_group(tbl_type):
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments=("f", seg_group(["low", "high"])),
        )
        .interrogate()
    )
    assert validation.n_passed(i=1, scalar=True) == 11


# TODO: expand to all tbl_types
@pytest.mark.parametrize("tbl_type", ["pandas", "polars"])
def test_segments_with_multiple_seg_groups(tbl_type):
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments=("a", seg_group([[1, 2, 3], [4], [7, 8]])),
        )
        .interrogate()
    )
    assert validation.n_passed(i=1, scalar=True) == 7
    assert validation.n_passed(i=2, scalar=True) == 3
    assert validation.n_passed(i=3, scalar=True) == 2
