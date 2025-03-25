from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar

import narwhals as nw

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
    py_type: ClassVar[type] = float


@dataclass(frozen=True)
class MeanStat(Stat):
    val: str
    name: ClassVar[str] = "mean"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().mean().cast(nw.Float64)
    py_type: ClassVar[type] = float


@dataclass(frozen=True)
class StdStat(Stat):
    val: str
    name: ClassVar[str] = "std"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().std().cast(nw.Float64)
    py_type: ClassVar[type] = float


@dataclass(frozen=True)
class MinStat(Stat):
    val: str
    name: ClassVar[str] = "min"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().min().cast(nw.Float64)
    py_type: ClassVar[type] = float


@dataclass(frozen=True)
class MaxStat(Stat):
    val: str
    name: ClassVar[str] = "max"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().max().cast(nw.Float64)
    py_type: ClassVar[type] = float


@dataclass(frozen=True)
class P05Stat(Stat):
    val: str
    name: ClassVar[str] = "p05"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().quantile(0.005, interpolation="linear").cast(nw.Float64)
    py_type: ClassVar[type] = float


@dataclass(frozen=True)
class Q1Stat(Stat):
    val: str
    name: ClassVar[str] = "q_1"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().quantile(0.25, interpolation="linear").cast(nw.Float64)
    py_type: ClassVar[type] = float


@dataclass(frozen=True)
class MedianStat(Stat):
    val: str
    name: ClassVar[str] = "median"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().median().cast(nw.Float64)
    py_type: ClassVar[type] = float


@dataclass(frozen=True)
class Q3Stat(Stat):
    val: str
    name: ClassVar[str] = "q_3"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().quantile(0.75, interpolation="linear").cast(nw.Float64)
    py_type: ClassVar[type] = float


@dataclass(frozen=True)
class P95Stat(Stat):
    val: str
    name: ClassVar[str] = "p95"
    group = StatGroup.DESCR
    expr: ClassVar[nw.Expr] = nw.all().quantile(0.95, interpolation="linear").cast(nw.Float64)
    py_type: ClassVar[type] = float


@dataclass(frozen=True)
class NTrue(Stat):
    val: int
    name: ClassVar[str] = "n_true"
    group = StatGroup.LOGIC
    expr: ClassVar[nw.Expr] = nw.all().sum().cast(nw.Boolean)
    py_type: ClassVar[type] = int


@dataclass(frozen=True)
class NFalse(Stat):
    val: int
    name: ClassVar[str] = "n_false"
    group = StatGroup.LOGIC
    # TODO: I don't think this is valid
    expr: ClassVar[nw.Expr] = ~nw.all().sum().cast(nw.Boolean)
    py_type: ClassVar[type] = int


@dataclass(frozen=True)
class NMissing(Stat):
    val: int
    name: ClassVar[str] = "n_missing"
    group = StatGroup.STRUCTURE
    expr: ClassVar[nw.Expr] = nw.all().null_count()
    py_type: ClassVar[type] = int


@dataclass(frozen=True)
class NUnique(Stat):
    val: int
    name: ClassVar[str] = "n_unique"
    group = StatGroup.STRUCTURE
    expr: ClassVar[nw.Expr] = nw.all().n_unique()
    py_type: ClassVar[type] = int


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
    NTrue,
    NFalse,
)
