from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.dataframe import DataFrame

from pointblank._constants import SVG_ICONS_FOR_DATA_TYPES
from pointblank._utils import transpose_dicts
from pointblank.scan_profile_stats import (
    FreqStat,
    IQRStat,
    MaxStat,
    MeanStat,
    MedianStat,
    MinStat,
    NMissing,
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

    from narwhals.typing import Frame


## Types that may cause unrecoverable errors and don't pose any value
ILLEGAL_TYPES = ("struct",)


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
    def calc_stats(self, data: Frame) -> None: ...


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
        if isinstance(value, Sequence):
            self._sample_data = value
            return
        raise NotImplementedError  # pragma: no cover

    def spawn_profile(self, _subprofile: type[ColumnProfile]) -> ColumnProfile:
        inst = _subprofile(coltype=self.coltype, colname=self.colname, statistics=self.statistics)
        # instantiate non-initializing properties
        inst.sample_data = self.sample_data
        return inst

    def calc_stats(self, data: Frame) -> None:
        summarized = _as_physical(
            data.select(_col=self.colname).select(_nmissing=NMissing.expr, _nunique=NUnique.expr)
        ).to_dict()

        self.statistics.extend(
            [
                NMissing(summarized["_nmissing"].item()),
                NUnique(summarized["_nunique"].item()),
            ]
        )


class _DateProfile(ColumnProfile):
    _type: _TypeMap = _TypeMap.DATE

    def calc_stats(self, data: Frame):
        res = data.rename({self.colname: "_col"}).select(_min=MinStat.expr, _max=MaxStat.expr)

        physical = _as_physical(res).to_dict()

        self.statistics.extend(
            [
                MinStat(physical["_min"].item()),
                MaxStat(physical["_max"].item()),
            ]
        )


class _BoolProfile(ColumnProfile):
    _type: _TypeMap = _TypeMap.BOOL

    def calc_stats(self, data: Frame) -> None:
        group_by_contexts = (
            data.rename({self.colname: "_col"}).group_by("_col").agg(_freq=FreqStat.expr)
        )

        summarized_groupby = _as_physical(group_by_contexts).to_dict()

        # TODO: Need a real way to do this
        col_vals: list[Any] = summarized_groupby["_col"].to_list()
        freqs: list[int] = summarized_groupby["_freq"].to_list()

        freq_dict: dict[str, int] = {
            str(colval): freq for colval, freq in zip(col_vals, freqs, strict=True)
        }

        self.statistics.extend([FreqStat(freq_dict)])


class _StringProfile(ColumnProfile):
    _type: _TypeMap = _TypeMap.STRING

    def calc_stats(self, data: Frame):
        str_data = data.select(nw.all().cast(nw.String).str.len_chars())

        # TODO: We should get an FreqStat here; estimate cardinality first

        summarized = (
            str_data.rename({self.colname: "_col"})
            .select(
                _mean=MeanStat.expr,
                _median=MedianStat.expr,
                _std=StdStat.expr,
                _min=MinStat.expr,
                _max=MaxStat.expr,
                _p_05=P05Stat.expr,
                _q_1=Q1Stat.expr,
                _q_3=Q3Stat.expr,
                _p_95=P95Stat.expr,
            )
            .with_columns(
                _iqr=IQRStat.expr,
            )
        )

        physical = _as_physical(summarized).to_dict()
        self.statistics.extend(
            [
                MeanStat(physical["_mean"].item()),
                MedianStat(physical["_median"].item()),
                StdStat(physical["_std"].item()),
                MinStat(physical["_min"].item()),
                MaxStat(physical["_max"].item()),
                P05Stat(physical["_p_05"].item()),
                Q1Stat(physical["_q_1"].item()),
                Q3Stat(physical["_q_3"].item()),
                P95Stat(physical["_p_95"].item()),
                IQRStat(physical["_iqr"].item()),
            ]
        )


class _NumericProfile(ColumnProfile):
    _type: _TypeMap = _TypeMap.NUMERIC

    def calc_stats(self, data: Frame):
        res = (
            data.rename({self.colname: "_col"})
            .select(
                _mean=MeanStat.expr,
                _median=MedianStat.expr,
                _std=StdStat.expr,
                _min=MinStat.expr,
                _max=MaxStat.expr,
                _p_05=P05Stat.expr,
                _q_1=Q1Stat.expr,
                _q_3=Q3Stat.expr,
                _p_95=P95Stat.expr,
            )
            # TODO: need a consistent way to indicate this
            .with_columns(_iqr=IQRStat.expr)
        )

        summarized = _as_physical(res).to_dict()
        self.statistics.extend(
            [
                MeanStat(summarized["_mean"].item()),
                MedianStat(summarized["_median"].item()),
                StdStat(summarized["_std"].item()),
                MinStat(summarized["_min"].item()),
                MaxStat(summarized["_max"].item()),
                P05Stat(summarized["_p_05"].item()),
                Q1Stat(summarized["_q_1"].item()),
                Q3Stat(summarized["_q_3"].item()),
                P95Stat(summarized["_p_95"].item()),
                IQRStat(summarized["_iqr"].item()),
            ]
        )


class _DataProfile:  # TODO: feels redundant and weird
    def __init__(
        self,
        table_name: str | None,
        columns: list[str],
        implementation: nw.Implementation,
    ):
        self.table_name: str | None = table_name
        self.columns: list[str] = columns
        self.implementation = implementation
        self.column_profiles: list[ColumnProfile] = []

    def set_row_count(self, data: Frame) -> None:
        assert self.columns  # internal: cols should already be set

        slim = data.select(nw.col(self.columns[0]))

        physical = _as_physical(slim)

        self.row_count = len(physical)

    def as_dataframe(self, *, strict: bool = True) -> DataFrame:
        assert self.column_profiles

        cols: list[dict[str, Any]] = []
        for prof in self.column_profiles:
            stat_vals = {}
            for stat in prof.statistics:
                stat_vals[stat.name] = stat.val

            stat_vals |= {"colname": prof.colname}
            stat_vals |= {"coltype": str(prof.coltype)}
            stat_vals |= {"sample_data": str(prof.sample_data)}  # TODO: not a good way to do this
            stat_vals |= {"icon": _TypeMap.fetch_icon(prof._type)}
            cols.append(stat_vals)

        # Stringify if type mismatch
        # Get all unique keys across all dictionaries
        all_keys = set().union(*(d.keys() for d in cols))

        for key in all_keys:
            # Get all values for this key across all dictionaries
            values = [d.get(key) for d in cols if key in d]

            # Check if all values are of the same type
            if len(values) > 1:
                first_type = type(values[0])

                # use `type` instead of instance check because some types are sub
                # classes of supers, ie. date is a subclass of datetime, so it's
                # technically an instance. This however would fail most dataframe
                # instantiations that require consistent types.
                all_same_type: bool = all(type(v) is first_type for v in values[1:])
                if not all_same_type:
                    if strict:
                        msg = f"Some types in {key!s} stat are different. Turn off `strict` to bypass."
                        raise TypeError(msg)
                    for d in cols:
                        if key in d:
                            d[key] = str(d[key])

        return nw.from_dict(transpose_dicts(cols), backend=self.implementation)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<_DataProfile(table_name={self.table_name}, row_count={self.row_count}, columns={self.columns})>"


def _as_physical(data: Frame) -> DataFrame:
    try:
        # TODO: might be a built in way to do this
        return data.collect()  # type: ignore[union-attr]
    except AttributeError:
        assert isinstance(data, DataFrame)  # help mypy
        return data
