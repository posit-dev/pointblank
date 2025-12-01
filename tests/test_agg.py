import pytest

from pointblank import Validate
import polars as pl


@pytest.mark.parametrize(
    "tol",
    [
        (0, 0),
        (1, 1),
        (100, 100),
        0,
    ],
)
def test_sums(tol) -> None:
    data = pl.DataFrame({"a": [1, 1, 1, None]})
    v = Validate(data).col_sum_eq("a", 3).interrogate()

    v.assert_below_threshold()

    v.get_tabular_report()


if __name__ == "__main__":
    pytest.main([__file__])
