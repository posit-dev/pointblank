from __future__ import annotations

from typing import TYPE_CHECKING

from pointblank import DataScan

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame


class Compare:
    def __init__(self, a: IntoFrame, b: IntoFrame) -> None:
        # Import processing functions from validate module
        from pointblank.validate import (
            _process_connection_string,
            _process_csv_input,
            _process_parquet_input,
        )

        # Process input data for table a
        a = _process_connection_string(a)
        a = _process_csv_input(a)
        a = _process_parquet_input(a)

        # Process input data for table b
        b = _process_connection_string(b)
        b = _process_csv_input(b)
        b = _process_parquet_input(b)

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
