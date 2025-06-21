from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import narwhals as nw

from pointblank import DataScan

if TYPE_CHECKING:
    from typing import Any

    from narwhals.typing import IntoFrameT


class Compare:
    def __init__(self, a: IntoFrameT, b: IntoFrameT, backend: Any = None) -> None:
        self.a: IntoFrameT = a
        self.b: IntoFrameT = b

    def compare(self) -> None:
        self._scana = DataScan(self.a)
        self._asummary: nw.DataFrame = nw.from_native(self._scana.summary_data)
        self._scanb = DataScan(self.b)
        self._bsummary: nw.DataFrame = nw.from_native(self._scanb.summary_data)

    @property
    def meta_summary(self) -> MetaSummary:
        """Return metadata summary."""
        # TODO: elegant error if compare is not called first

        ## Number of rows:
        arows: int = self._scana.profile.row_count
        brows: int = self._scanb.profile.row_count

        ## Number of variables:
        avars: int = len(self._scana.profile.columns)
        bvars: int = len(self._scanb.profile.columns)

        ## Cols only in `a`:
        acols: set[str] = set(self._scana.profile.columns)
        bcols: set[str] = set(self._scanb.profile.columns)
        aonly: set[str] = acols - bcols
        bonly: set[str] = bcols - acols
        bothcols: set[str] = acols & bcols

        ## Conflicting types:
        conflicting: list[str] = []
        for col in bothcols:
            atype = self._scana.profile[col].coltype
            btype = self._scanb.profile[col].coltype
            if atype != btype:
                conflicting.append(col)

        ## Create the Summary Frame:
        aname: str = self._scana.profile.table_name or "a"
        bname: str = self._scanb.profile.table_name or "b"

        return MetaSummary(
            name=[aname, bname],
            n_observations=(arows, brows),
            n_variables=(avars, bvars),
            in_a_only=aonly,
            in_b_only=bonly,
            in_both=bothcols,
            conflicting_types=conflicting,
        )


class MetaSummary(NamedTuple):
    name: list[str]
    n_observations: tuple[int, int]
    n_variables: tuple[int, int]
    in_a_only: set[str]
    in_b_only: set[str]
    in_both: set[str]
    conflicting_types: list[str]
