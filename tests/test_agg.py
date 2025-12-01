import pytest

from pointblank import Validate
import polars as pl


@pytest.fixture
def simple_pl() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 1, 1, None],
            "b": [2, 2, 2, None],
            "c": [3, 3, 3, None],
        }
    )


@pytest.mark.parametrize(
    "tol",
    [
        (0, 0),
        (1, 1),
        (100, 100),
        0,
    ],
)
def test_sums_old(tol, simple_pl) -> None:
    v = Validate(simple_pl).col_sum_eq("a", 3, tol=tol).interrogate()

    v.assert_below_threshold()

    v.get_tabular_report()


@pytest.mark.parametrize(
    ("method", "vals"),
    [
        ("col_sum_eq", (3, 6, 9)),
        ("col_sum_gt", (2, 5, 8)),
        ("col_sum_ge", (3, 6, 9)),
        ("col_sum_lt", (4, 7, 10)),
        ("col_sum_le", (3, 6, 9)),
    ],
)
def test_sums(simple_pl: pl.DataFrame, method: str, vals: tuple[int, int, int]):
    v = Validate(simple_pl)
    for col, val in zip(["a", "b", "c"], vals):
        getattr(v, method)(col, val)
    v = v.interrogate()

    v.assert_below_threshold()
    v.get_tabular_report()


@pytest.mark.parametrize(
    ("method", "vals"),
    [
        ("col_avg_eq", (1, 2, 3)),
        ("col_avg_gt", (0, 1, 2)),
        ("col_avg_ge", (1, 2, 3)),
        ("col_avg_lt", (2, 3, 4)),
        ("col_avg_le", (1, 2, 3)),
    ],
)
def test_avgs(simple_pl: pl.DataFrame, method: str, vals: tuple[int, int, int]):
    v = Validate(simple_pl)
    for col, val in zip(["a", "b", "c"], vals):
        getattr(v, method)(col, val)
    v = v.interrogate()

    v.assert_below_threshold()
    v.get_tabular_report()


if __name__ == "__main__":
    pytest.main([__file__])
