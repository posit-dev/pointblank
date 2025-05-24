from __future__ import annotations

import pointblank as pb
import polars as pl

import sys

import pytest


@pytest.mark.skip(reason="Test does not work yet. Waiting on GT fix.")
def test_no_pandas_interr() -> None:
    sys.modules["pandas"] = None

    with pytest.raises(ModuleNotFoundError):
        import pandas

    df = pl.DataFrame(
        {
            "date": [pl.date(2023, 1, 1), pl.date(2023, 1, 2), pl.date(2023, 1, 3)],
            "number": [1, 2, 3],
        }
    )

    validate = pb.Validate(data=df).col_exists(columns=["date", "number"])

    validate.interrogate().get_tabular_report()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
