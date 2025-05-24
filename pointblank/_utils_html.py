from __future__ import annotations

from typing import Any

from great_tables import html

from pointblank._constants import TABLE_TYPE_STYLES
from pointblank._utils import _format_to_integer_value


def _fmt_frac(vec) -> list[str | None]:
    res: list[str | None] = []
    for x in vec:
        if x is None:
            res.append(x)
            continue

        if x == 0:
            res.append("0")
            continue

        if x < 0.01:
            res.append("<.01")
            continue

        try:
            intx: int = int(x)
        except ValueError:  # generic object, ie. NaN
            res.append(str(x))
            continue

        if intx == x:  # can remove trailing 0s w/o loss
            res.append(str(intx))
            continue

        res.append(str(round(x, 2)))

    return res


def _make_sublabel(major: str, minor: str) -> Any:
    return html(
        f'{major!s}<span style="font-size: 0.75em; vertical-align: sub; position: relative; line-height: 0.5em;">{minor!s}</span>'
    )


def _create_table_type_html(
    tbl_type: str | None, tbl_name: str | None, font_size: str = "10px"
) -> str:
    if tbl_type is None:
        return ""

    style = TABLE_TYPE_STYLES.get(tbl_type)

    if style is None:
        return ""

    if tbl_name is None:
        return (
            f"<span style='background-color: {style['background']}; color: {style['text']}; padding: 0.5em 0.5em; "
            f"position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px {style['background']}; "
            f"font-weight: bold; padding: 2px 10px 2px 10px; font-size: {font_size};'>{style['label']}</span>"
        )

    return (
        f"<span style='background-color: {style['background']}; color: {style['text']}; padding: 0.5em 0.5em; "
        f"position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px {style['background']}; "
        f"font-weight: bold; padding: 2px 15px 2px 15px; font-size: {font_size};'>{style['label']}</span>"
        f"<span style='background-color: none; color: #222222; padding: 0.5em 0.5em; "
        f"position: inherit; margin: 5px 10px 5px -4px; border: solid 1px {style['background']}; "
        f"font-weight: bold; padding: 2px 15px 2px 15px; font-size: {font_size};'>{tbl_name}</span>"
    )


def _create_table_dims_html(columns: int, rows: int, font_size: str = "10px") -> str:
    rows_fmt = _format_to_integer_value(int(rows))
    columns_fmt = _format_to_integer_value(int(columns))

    return (
        f"<span style='background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; "
        f"position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; "
        f"font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; "
        f"font-size: {font_size};'>Rows</span>"
        f"<span style='background-color: none; color: #333333; padding: 0.5em 0.5em; "
        f"position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; "
        f"border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: {font_size};'>"
        f"{rows_fmt}</span>"
        f"<span style='background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; "
        f"position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; "
        f"font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; "
        f"font-size: {font_size};'>Columns</span>"
        f"<span style='background-color: none; color: #333333; padding: 0.5em 0.5em; "
        f"position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; "
        f"border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: {font_size};'>"
        f"{columns_fmt}</span>"
    )
