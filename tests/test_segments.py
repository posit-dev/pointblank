from hypothesis import given, strategies as st
import pytest

from pointblank.validate import (
    Validate,
    load_dataset,
)
from pointblank.segments import (
    Segment,
    seg_group,
)

import pandas as pd
import polars as pl


def test_segment_class_fails_not_list():
    with pytest.raises(TypeError, match="Segments must be lists"):
        Segment((1, 2, 3))


def test_segment_class_fails_not_nested_lists():
    with pytest.raises(TypeError, match="Sub-segments must be lists."):
        Segment([1, 2, 3])


def test_segment_class_fails_mixed_types():
    with pytest.raises(TypeError, match="All segment values must have the same type"):
        Segment([[1, 2, 3], ["a", "b"]])


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


def test_seg_group_invalid_input():
    with pytest.raises(ValueError, match="Must input a list of values for a segment"):
        seg_group("not a list")


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
    from datetime import date

    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments=("date", (date(2016, 1, 4), date(2016, 1, 5))),
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


def test_segments_str_lazyframe():
    df = pl.DataFrame({"x": [1.0, -1.0, 2.0], "region": ["US", "US", "EU"]})
    validation_lazy = Validate(data=df.lazy()).col_vals_gt("x", 0, segments="region").interrogate()
    validation_eager = Validate(data=df).col_vals_gt("x", 0, segments="region").interrogate()

    # Both should produce the same number of expanded steps
    assert len(validation_lazy.validation_info) == len(validation_eager.validation_info)

    # EU: 1 row, 1 passes; US: 2 rows, 1 passes
    assert validation_lazy.n_passed(i=1, scalar=True) == validation_eager.n_passed(i=1, scalar=True)
    assert validation_lazy.n_passed(i=2, scalar=True) == validation_eager.n_passed(i=2, scalar=True)


@pytest.mark.parametrize("lazy", [False, True])
def test_seg_group_float_column_int_values(lazy):
    # Polars >= 2.0 no longer coerces between integers and floats in `is_in()`
    df = pl.DataFrame({"x": [1, -1, 2, 3], "g": [1.0, 2.0, 3.0, 4.0]})
    validation = (
        Validate(data=df.lazy() if lazy else df)
        .col_vals_gt("x", 0, segments=("g", seg_group([[1, 2], [3, 4]])))
        .interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 1
    assert validation.n_failed(i=1, scalar=True) == 1
    assert validation.n_passed(i=2, scalar=True) == 2


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_segments_str_with_null_values(tbl_type):
    # Column `c` has missing values; deriving segments from it should give every backend the same
    # sorted set of segments with a single `None` segment at the end
    validation = (
        Validate(data=load_dataset(dataset="small_table", tbl_type=tbl_type))
        .col_vals_gt(
            columns="d",
            value=100,
            segments="c",
        )
        .interrogate()
    )

    assert [v.segments[1] for v in validation.validation_info] == [2, 3, 4, 7, 8, 9, None]
    assert not any(v.eval_error for v in validation.validation_info)
    assert validation.n(i=1, scalar=True) == 1
    assert validation.n(i=2, scalar=True) == 3
    assert validation.n(i=7, scalar=True) == 2
    assert validation.n_passed(i=7, scalar=True) == 2


@pytest.mark.parametrize(
    "dtype", [None, "string", pytest.param("string[pyarrow]", id="string-pyarrow")]
)
def test_segments_str_with_null_values_pandas_string_columns(dtype):
    # Pandas represents a missing string as `nan` or `pd.NA` depending on the dtype; both must be
    # treated as a `None` segment rather than making the step an evaluation error
    df = pd.DataFrame({"x": [1, -1, 2, 3], "region": ["US", "US", None, "EU"]})
    if dtype is not None:
        df = df.astype({"region": dtype})

    validation = Validate(data=df).col_vals_gt("x", 0, segments="region").interrogate()

    assert [v.segments for v in validation.validation_info] == [
        ("region", "EU"),
        ("region", "US"),
        ("region", None),
    ]
    assert not any(v.eval_error for v in validation.validation_info)
    assert validation.n_passed(i=1, scalar=True) == 1
    assert validation.n_passed(i=2, scalar=True) == 1
    assert validation.n_passed(i=3, scalar=True) == 1
    assert validation.n_failed(i=2, scalar=True) == 1


def _empty_segment_tbl(tbl_type):
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64), "s": pl.Series([], dtype=pl.String)})
    if tbl_type == "polars":
        return df
    if tbl_type == "polars-lazy":
        return df.lazy()
    if tbl_type == "pandas":
        return df.to_pandas()
    import ibis

    return ibis.memtable(df.to_arrow())


@pytest.mark.parametrize("tbl_type", ["polars", "polars-lazy", "pandas", "duckdb"])
def test_segments_str_on_empty_table_keeps_step(tbl_type):
    # A table with no rows has no segment values; the segmented step is kept (with no test units,
    # like any step on an empty table) rather than vanishing from the report
    validation = (
        Validate(data=_empty_segment_tbl(tbl_type), thresholds=(1, 2))
        .col_vals_gt("x", 0, segments="s")
        .col_vals_gt("x", 0)
        .interrogate()
    )

    assert len(validation.validation_info) == 2
    assert [v.i for v in validation.validation_info] == [1, 2]

    step = validation.validation_info[0]
    assert step.n == 0
    assert step.eval_error is True
    assert step.segments is None
    assert step.segments_spec == "s"
    assert "no_segments" in step.notes
    assert "No segments were found" in step.notes["no_segments"]["text"]

    # The step after it is unaffected
    assert validation.validation_info[1].n == 0
    assert "no_segments" not in (validation.validation_info[1].notes or {})

    validation.get_tabular_report()


def test_segments_on_table_emptied_by_pre_keeps_step():
    tbl = pl.DataFrame({"x": [1, 2, 3], "s": ["a", "b", "a"]})

    validation = (
        Validate(data=tbl)
        .col_vals_gt("x", 0, pre=lambda df: df.filter(pl.col("x") > 10), segments="s")
        .col_vals_gt("x", 0, segments="s")
        .interrogate()
    )

    assert len(validation.validation_info) == 3
    assert validation.validation_info[0].n == 0
    assert "no_segments" in validation.validation_info[0].notes
    assert [v.segments for v in validation.validation_info[1:]] == [("s", "a"), ("s", "b")]
    assert validation.n_passed(i=2, scalar=True) == 2
    assert validation.n_passed(i=3, scalar=True) == 1


@pytest.mark.parametrize("segments", [("s", []), [("s", [])]])
def test_segments_with_no_values_on_nonempty_table_is_eval_error(segments):
    # With an empty list of segment values there is nothing to check; the step must not fall back
    # to checking every row of the table
    tbl = pl.DataFrame({"x": [1, -2, 3], "s": ["a", "b", "a"]})

    validation = (
        Validate(data=tbl)
        .col_vals_gt("x", 0, segments=segments)
        .col_vals_gt("x", 0, segments="s")
        .interrogate()
    )

    assert len(validation.validation_info) == 3
    step = validation.validation_info[0]
    assert step.eval_error is True
    assert step.n is None
    assert step.n_failed is None
    assert step.segments is None
    assert "no_segments" in step.notes

    assert [v.segments for v in validation.validation_info[1:]] == [("s", "a"), ("s", "b")]
    assert validation.n_failed(i=3, scalar=True) == 1


def test_segments_with_no_segments_found_serializes_segments():
    validation = (
        Validate(data=_empty_segment_tbl("polars"))
        .col_vals_gt("x", 0, segments="s")
        .col_vals_gt("x", 0, segments=("s", []))
        .interrogate()
    )

    code = validation.to_code()
    assert 'segments="s"' in code
    assert 'segments=("s", [])' in code
    assert code.count(".col_vals_gt(") == 2


def test_segments_no_segments_note_is_localized():
    validation = (
        Validate(data=_empty_segment_tbl("polars"), locale="fr")
        .col_vals_gt("x", 0, segments="s")
        .interrogate()
    )

    note_text = validation.validation_info[0].notes["no_segments"]["text"]
    assert note_text.startswith("Aucun segment n'a été trouvé")
