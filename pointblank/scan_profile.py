from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict

import narwhals as nw

from pointblank._constants import SVG_ICONS_FOR_DATA_TYPES
from pointblank.scan_profile_stats import (
    MaxStat,
    MeanStat,
    MedianStat,
    MinStat,
    NFalse,
    NMissing,
    NTrue,
    NUnique,
    P05Stat,
    P95Stat,
    Q1Stat,
    Q3Stat,
    Stat,
    StdStat,
)

if TYPE_CHECKING:
    from collections.abc import MutableSequence

    from narwhals.dataframe import DataFrame


## Columns that pose risks for exceptions or don't have handlers
ILLEGAL_TYPES = ("struct",)


class _Metadata(TypedDict):  # TODO: Need a better name
    row_count: int
    implementation: nw.Implementation


class _TypeMap(Enum):  # ! ordered;
    # TODO: consolidate w/other stats?
    NUMERIC = ("int", "float")
    STRING = ("string", "categorical")
    DATE = ("date",)
    BOOL = ("bool",)

    @classmethod
    def is_illegal(cls, dtype: Any) -> bool:
        return any(ind for ind in ILLEGAL_TYPES if ind in str(dtype).lower())

    @classmethod
    def fetch_prof_map(cls) -> dict[_TypeMap, type[ColumnProfile]]:
        default = defaultdict(lambda: ColumnProfile)
        implemented_dict: dict[_TypeMap, type[ColumnProfile]] = {
            cls.BOOL: _BoolProfile,
            cls.NUMERIC: _NumericProfile,
            cls.STRING: _StringProfile,
            cls.DATE: _DateProfile,
        }
        return default | implemented_dict

    @classmethod
    def fetch_profile(cls, dtype: Any) -> type[ColumnProfile]:
        stringified: str = str(dtype).lower()
        for _type in cls:
            inds: tuple[str, ...] = _type.value
            is_match: bool = any(ind for ind in inds if ind in stringified)
            if is_match:
                return cls.fetch_prof_map()[_type]
        raise NotImplementedError  # pragma: no-cover

    @classmethod
    def fetch_icon(cls, _type: _TypeMap) -> str:
        icon_map = {
            cls.NUMERIC: "numeric",
            cls.STRING: "string",
            cls.DATE: "date",
            cls.BOOL: "boolean",
        }
        try:
            icon_key = icon_map[_type]
        except KeyError:
            icon_key = "object"
        return SVG_ICONS_FOR_DATA_TYPES[icon_key]


class _ColumnProfileABC(ABC):
    @abstractmethod
    def calc_stats(self, data: DataFrame) -> None: ...


@dataclass
class ColumnProfile(_ColumnProfileABC):
    colname: str
    coltype: str
    statistics: MutableSequence[Stat] = field(default_factory=lambda: [])

    @property
    def sample_data(self) -> Sequence[Any]:
        return self._sample_data

    @sample_data.setter
    def sample_data(self, value: object) -> None:
        # TODO: type guard this
        if isinstance(value, Sequence):
            self._sample_data = value
            return
        raise NotImplementedError  # pragma: no cover

    def spawn_profile(self, _subprofile: type[ColumnProfile]) -> ColumnProfile:
        inst = _subprofile(coltype=self.coltype, colname=self.colname, statistics=self.statistics)
        # instantiate non-initializing properties
        inst.sample_data = self.sample_data
        return inst

    def calc_stats(self, data: DataFrame) -> None:
        summarized = data.select(_nmissing=NMissing.expr, _nunique=NUnique.expr)

        self.statistics = [
            NMissing(summarized["_nmissing"].item()),
            NUnique(summarized["_nunique"].item()),
        ]


class _DateProfile(ColumnProfile):
    _type: _TypeMap = _TypeMap.DATE

    def calc_stats(self, data: DataFrame):
        res = data.select(_min=MinStat.expr, _max=MaxStat.expr).to_dict()

        self.statistics: list[Stat] = [MinStat(res["_min"].item()), MaxStat(res["_max"].item())]


class _BoolProfile(ColumnProfile):
    _type: _TypeMap = _TypeMap.BOOL

    def calc_stats(self, data: DataFrame) -> None:
        res = data.select(_ntrue=NTrue.expr, _nfalse=NFalse.expr).to_dict()

        self.statistics: list[Stat] = [NTrue(res["_ntrue"].item()), NFalse(res["_nfalse"].item())]


class _StringProfile(ColumnProfile):
    _type: _TypeMap = _TypeMap.STRING

    def calc_stats(self, data: DataFrame):
        str_data = data.select(nw.all().cast(nw.String).str.len_chars())

        summarized = str_data.select(
            _mean=MeanStat.expr,
            _median=MedianStat.expr,
            _std=StdStat.expr,
            _min=MinStat.expr,
            _max=MaxStat.expr,
            _p05=P05Stat.expr,
            _q1=Q1Stat.expr,
            _q3=Q3Stat.expr,
            _p95=P95Stat.expr,
        )

        res = summarized.to_dict()

        stats: list[Stat] = []

        stats.append(MeanStat(res["_mean"].item()))
        stats.append(MedianStat(res["_median"].item()))
        stats.append(StdStat(res["_std"].item()))
        stats.append(MinStat(res["_min"].item()))
        stats.append(MaxStat(res["_max"].item()))
        stats.append(P05Stat(res["_p05"].item()))
        stats.append(Q1Stat(res["_q1"].item()))
        stats.append(Q3Stat(res["_q3"].item()))
        stats.append(P95Stat(res["_p95"].item()))

        self.statistics.extend(stats)


class _NumericProfile(ColumnProfile):
    _type: _TypeMap = _TypeMap.NUMERIC

    def calc_stats(self, data: DataFrame):
        summarized = data.select(
            _mean=MeanStat.expr,
            _median=MedianStat.expr,
            _std=StdStat.expr,
            _min=MinStat.expr,
            _max=MaxStat.expr,
            _p05=P05Stat.expr,
            _q1=Q1Stat.expr,
            _q3=Q3Stat.expr,
            _p95=P95Stat.expr,
        )

        res = summarized.to_dict()

        stats: list[Stat] = []

        stats.append(MeanStat(res["_mean"].item()))
        stats.append(MedianStat(res["_median"].item()))
        stats.append(StdStat(res["_std"].item()))
        stats.append(MinStat(res["_min"].item()))
        stats.append(MaxStat(res["_max"].item()))
        stats.append(P05Stat(res["_p05"].item()))
        stats.append(Q1Stat(res["_q1"].item()))
        stats.append(Q3Stat(res["_q3"].item()))
        stats.append(P95Stat(res["_p95"].item()))

        self.statistics.extend(stats)


class _DataProfile:  # TODO: feels redundant and weird
    def __init__(
        self,
        table_name: str | None,
        row_count: int,
        columns: list[str],
        implementation: nw.Implementation,
    ):
        self.table_name: str | None = table_name
        self.row_count: int = row_count
        self.columns: list[str] = columns
        self.implementation = implementation
        self.column_profiles: list[ColumnProfile] = []

    def as_dataframe(self, *, format_html: bool = False) -> DataFrame:
        if format_html:
            formatter = str
        else:
            formatter = lambda val: val  # noqa: E731

        cols: list[dict] = []
        for prof in self.column_profiles:
            stat_vals = {}
            for stat in prof.statistics:
                if isinstance(stat.val, float):
                    val = round(stat.val, 5)
                else:
                    val = stat.val

                stat_vals[stat.name] = formatter(val)

            stat_vals |= {"colname": prof.colname}
            stat_vals |= {"coltype": str(prof.coltype)}
            stat_vals |= {"sample_data": prof.sample_data}
            stat_vals |= {"icon": _TypeMap.fetch_icon(prof._type)}
            cols.append(stat_vals)

        # TODO: This is a bad way to do this
        df_native = self.implementation.to_native_namespace().DataFrame(cols)

        return nw.from_native(df_native)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<_DataProfile(table_name={self.table_name}, row_count={self.row_count}, columns={self.columns})>"
