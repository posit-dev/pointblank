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
    SUMMARY = auto()
    STRUCTURE = auto()
    LOGIC = auto()
    IQR = auto()
    FREQ = auto()
    BOUNDS = auto()


# TODO: Make sure all these subclasses are suffixed w/`Stat`
# TODO: Replace all the nw.all w/_col


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
    group = StatGroup.SUMMARY
    expr: ClassVar[nw.Expr] = nw.col("_col").mean()
    label: ClassVar[str] = "Mean"


@dataclass(frozen=True)
class StdStat(Stat):  # TODO: Rename this SD for consistency
    val: str
    name: ClassVar[str] = "std"
    group = StatGroup.SUMMARY
    expr: ClassVar[nw.Expr] = nw.col("_col").std()
    label: ClassVar[str] = "SD"


@dataclass(frozen=True)
class MinStat(Stat):
    val: str
    name: ClassVar[str] = "min"
    group = StatGroup.BOUNDS  # TODO: These should get put back in DESCR once datetime p*
    expr: ClassVar[nw.Expr] = nw.col("_col").min()  # don't cast as float, can be date
    label: ClassVar[str] = "Min"


@dataclass(frozen=True)
class MaxStat(Stat):
    val: str
    name: ClassVar[str] = "max"
    group = StatGroup.BOUNDS  # TODO: These should get put back in DESCR once datetime p*
    expr: ClassVar[nw.Expr] = nw.col("_col").max()  # don't cast as float, can be date
    label: ClassVar[str] = "Max"


@dataclass(frozen=True)
class P05Stat(Stat):
    val: str
    name: ClassVar[str] = "p05"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.col("_col").quantile(0.005, interpolation="linear")
    label: ClassVar[str] = _make_sublabel("P", "5")


@dataclass(frozen=True)
class Q1Stat(Stat):
    val: str
    name: ClassVar[str] = "q_1"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.col("_col").quantile(0.25, interpolation="linear")
    label: ClassVar[str] = _make_sublabel("Q", "1")


@dataclass(frozen=True)
class MedianStat(Stat):
    val: str
    name: ClassVar[str] = "median"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.col("_col").median()
    label: ClassVar[str] = "Med"


@dataclass(frozen=True)
class Q3Stat(Stat):
    val: str
    name: ClassVar[str] = "q_3"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.col("_col").quantile(0.75, interpolation="linear")
    label: ClassVar[str] = _make_sublabel("Q", "3")


@dataclass(frozen=True)
class P95Stat(Stat):
    val: str
    name: ClassVar[str] = "p95"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.col("_col").quantile(0.95, interpolation="linear")
    label: ClassVar[str] = _make_sublabel("P", "95")


@dataclass(frozen=True)
class IQRStat(Stat):
    val: str
    name: ClassVar[str] = "iqr"
    group = StatGroup.IQR
    expr: ClassVar[nw.Expr] = nw.col(Q3Stat._fetch_priv_name()) - nw.col(Q1Stat._fetch_priv_name())
    label: ClassVar[str] = "IQR"


@dataclass(frozen=True)
class FreqStat(Stat):
    val: dict[str, int]  # the key must be stringified
    name: ClassVar[str] = "freqs"
    group = StatGroup.FREQ
    expr: ClassVar[nw.Expr] = nw.len()
    label: ClassVar[str] = "Freq"


@dataclass(frozen=True)
class NMissing(Stat):
    val: int
    name: ClassVar[str] = "n_missing"
    group = StatGroup.STRUCTURE
    expr: ClassVar[nw.Expr] = nw.col("_col").null_count().cast(nw.Int64)
    label: ClassVar[str] = "NA"


@dataclass(frozen=True)
class NUnique(Stat):
    val: int
    name: ClassVar[str] = "n_unique"
    group = StatGroup.STRUCTURE
    expr: ClassVar[nw.Expr] = nw.col("_col").n_unique().cast(nw.Int64)
    label: ClassVar[str] = "UQ"


COLUMN_ORDER_REGISTRY: tuple[type[Stat], ...] = (
    NMissing,
    NUnique,
    MeanStat,
    StdStat,
    MinStat,
    P05Stat,
    Q1Stat,
    MedianStat,
    Q3Stat,
    P95Stat,
    MaxStat,
    FreqStat,
    IQRStat,
)
