from __future__ import annotations
import pytest
import polars as pl

from pointblank.compare import Compare, MetaSummary


def test_compare_basic() -> None:
    df1 = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    }
    df2 = {
        "a": [1, 2, 3],
        "b": ["4", "5", "7"],
        "c": [8, 9, 10],
    }
    data1 = pl.DataFrame(df1)
    data2 = pl.DataFrame(df2)
    comp = Compare(data1, data2)

    comp.compare()

    ## Pull out the summary data
    summary: MetaSummary = comp.meta_summary

    assert summary.name == ["a", "b"]
    assert summary.n_observations == (3, 3)
    assert summary.n_variables == (2, 3)
    assert summary.in_a_only == set()
    assert summary.in_b_only == {"c"}
    assert summary.in_both == {"a", "b"}
    assert summary.conflicting_types == ["b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
