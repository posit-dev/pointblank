from __future__ import annotations

from typing import TYPE_CHECKING

from pointblank import DataScan

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame


class Compare:
    def __init__(self, a: IntoFrame, b: IntoFrame) -> None:
        self.a: IntoFrame = a
        self.b: IntoFrame = b

    def compare(self) -> None:
        ## Scan both frames
        self._scana = DataScan(self.a)
        self._scanb = DataScan(self.b)

        ## Get summary outs
        summarya = self._scana.summary_data
        summaryb = self._scana.summary_data

        summarya.columns

        self._scana.profile
