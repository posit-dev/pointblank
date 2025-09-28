import polars as pl
import pytest


@pytest.fixture
def half_null_ser() -> pl.Series:
    """A 1k element half null series. Exists to get around rounding issues."""
    data = [None if i % 2 == 0 else i for i in range(1000)]
    return pl.Series("half_null", data)
