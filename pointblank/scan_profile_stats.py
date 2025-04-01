from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar

import narwhals as nw

from pointblank._utils_html import _make_sublabel

if TYPE_CHECKING:
    from typing import Any


class StatGroup(Enum):
    DESCR = auto()
    STRUCTURE = auto()
    LOGIC = auto()


class Stat(ABC):
    val: Any
    name: ClassVar[str]
    group: ClassVar[StatGroup]
    expr: ClassVar[nw.Expr]
    label: ClassVar[str]

    def __eq__(self, value) -> bool:
        if isinstance(value, str):
            return value == self.name
        if isinstance(value, Stat):
            return value is self
        return NotImplemented

    @classmethod
    def _fetch_priv_name(self) -> str:
        return f"_{self.name}"


@dataclass(frozen=True)
class MeanStat(Stat):
    val: str
    name: ClassVar[str] = "mean"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().mean().cast(nw.Float64)
    label: ClassVar[str] = "Mean"


@dataclass(frozen=True)
class StdStat(Stat):
    val: str
    name: ClassVar[str] = "std"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().std().cast(nw.Float64)
    label: ClassVar[str] = "SD"


@dataclass(frozen=True)
class MinStat(Stat):
    val: str
    name: ClassVar[str] = "min"
    group = StatGroup.STRUCTURE
    expr: ClassVar[nw.Expr] = nw.all().min()  # don't cast as float, can be date
    label: ClassVar[str] = "Min"


@dataclass(frozen=True)
class MaxStat(Stat):
    val: str
    name: ClassVar[str] = "max"
    group = StatGroup.STRUCTURE
    expr: ClassVar[nw.Expr] = nw.all().max()  # don't cast as float, can be date
    label: ClassVar[str] = "Max"


@dataclass(frozen=True)
class P05Stat(Stat):
    val: str
    name: ClassVar[str] = "p05"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().quantile(0.005, interpolation="linear").cast(nw.Float64)
    label: ClassVar[str] = _make_sublabel("P", "5")


@dataclass(frozen=True)
class Q1Stat(Stat):
    val: str
    name: ClassVar[str] = "q_1"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().quantile(0.25, interpolation="linear").cast(nw.Float64)
    label: ClassVar[str] = _make_sublabel("Q", "1")


@dataclass(frozen=True)
class MedianStat(Stat):
    val: str
    name: ClassVar[str] = "median"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().median().cast(nw.Float64)
    label: ClassVar[str] = "Med"


@dataclass(frozen=True)
class Q3Stat(Stat):
    val: str
    name: ClassVar[str] = "q_3"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().quantile(0.75, interpolation="linear").cast(nw.Float64)
    label: ClassVar[str] = _make_sublabel("Q", "3")


@dataclass(frozen=True)
class P95Stat(Stat):
    val: str
    name: ClassVar[str] = "p95"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().quantile(0.95, interpolation="linear").cast(nw.Float64)
    label: ClassVar[str] = _make_sublabel("P", "95")


@dataclass(frozen=True)
class IQRStat(Stat):
    val: str
    name: ClassVar[str] = "iqr"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.col(Q3Stat._fetch_priv_name()) - nw.col(
        Q1Stat._fetch_priv_name()
    ).cast(nw.Float64)
    label: ClassVar[str] = "IQR"


@dataclass(frozen=True)
class NTrue(Stat):
    val: int
    name: ClassVar[str] = "n_true"
    group = StatGroup.LOGIC
    expr: ClassVar[nw.Expr] = nw.all().sum().cast(nw.Int64)
    label: ClassVar[str] = _make_sublabel("True", "N")


@dataclass(frozen=True)
class NFalse(Stat):
    val: int
    name: ClassVar[str] = "n_false"
    group = StatGroup.LOGIC
    ## Indirection to the point of absurdity:
    ## - Reversing the bool results in inconsistent unsigned types, ie. False is max hexidecimal
    ## - Filtering on `nw.all` is generally unavailable, at least for polars
    ## - Replacing T/F w/Null is not interpreted correctly by Narwhals
    ## - Swapping 0 -> 1 and 1 -> 0 is the most consistent, albiet odd
    expr: ClassVar[nw.Expr] = (
        nw.all().cast(nw.Int64).replace_strict([1, 0], new=[0, 1]).sum().cast(nw.Int64)
    )
    label: ClassVar[str] = _make_sublabel("False", "N")


@dataclass(frozen=True)
class NMissing(Stat):
    val: int
    name: ClassVar[str] = "n_missing"
    group = StatGroup.STRUCTURE
    expr: ClassVar[nw.Expr] = nw.all().null_count().cast(nw.Int64)
    label: ClassVar[str] = _make_sublabel("Missing", "N")


@dataclass(frozen=True)
class NUnique(Stat):
    val: int
    name: ClassVar[str] = "n_unique"
    group = StatGroup.STRUCTURE
    expr: ClassVar[nw.Expr] = nw.all().n_unique().cast(nw.Int64)
    label: ClassVar[str] = _make_sublabel("UQ", "N")


COLUMN_ORDER_REGISTRY: tuple[type[Stat], ...] = (
    NUnique,
    NMissing,
    MinStat,
    MaxStat,
    MeanStat,
    StdStat,
    P05Stat,
    Q1Stat,
    MedianStat,
    Q3Stat,
    P95Stat,
    IQRStat,
    NTrue,
    NFalse,
)
