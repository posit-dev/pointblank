from __future__ import annotations
import pytest

from pointblank.compare import Compare
import polars.testing.parametric as pt
from hypothesis import given


@pytest.mark.xfail
def test_compare_basic(dfa, dfb) -> None:
    comp = Compare(dfa, dfb)

    comp.compare()

    raise NotImplementedError
