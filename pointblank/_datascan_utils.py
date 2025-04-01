from __future__ import annotations

from math import floor, log10
from typing import TYPE_CHECKING

from great_tables.vals import fmt_integer, fmt_number, fmt_scientific

if TYPE_CHECKING:
    pass


def _round_to_sig_figs(value: float, sig_figs: int) -> float:
    if value == 0:
        return 0
    return round(value, sig_figs - int(floor(log10(abs(value)))) - 1)


def _compact_integer_fmt(value: float | int) -> str:
    if value == 0:
        formatted = "0"
    elif abs(value) >= 1 and abs(value) < 10_000:
        formatted = fmt_integer(value, use_seps=False)[0]
    else:
        formatted = fmt_scientific(value, decimals=1, exp_style="E1")[0]

    return formatted


def _compact_decimal_fmt(value: float | int) -> str:
    if value == 0:
        formatted = "0.00"
    elif abs(value) < 1 and abs(value) >= 0.01:
        formatted = fmt_number(value, decimals=2)[0]
    elif abs(value) < 0.01:
        formatted = fmt_scientific(value, decimals=1, exp_style="E1")[0]
    elif abs(value) >= 1 and abs(value) < 1000:
        formatted = fmt_number(value, n_sigfig=3)[0]
    elif abs(value) >= 1000 and abs(value) < 10_000:
        formatted = fmt_number(value, decimals=0, use_seps=False)[0]
    else:
        formatted = fmt_scientific(value, decimals=1, exp_style="E1")[0]

    return formatted


def _compact_0_1_fmt(value: float | int) -> str:
    if value == 0:
        formatted = " 0.00"
    elif value == 1:
        formatted = " 1.00"
    elif abs(value) < 1 and abs(value) >= 0.01:
        formatted = " " + fmt_number(value, decimals=2)[0]
    elif abs(value) < 0.01:
        formatted = "<0.01"
    elif abs(value) > 0.99:
        formatted = ">0.99"
    else:
        formatted = fmt_number(value, n_sigfig=3)[0]

    return formatted
