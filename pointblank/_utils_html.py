from __future__ import annotations

from typing import TYPE_CHECKING, Any

from great_tables import html

from pointblank._constants import TABLE_TYPE_STYLES
from pointblank._utils import _format_to_integer_value

if TYPE_CHECKING:
    from pointblank.steps import Steps


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


def _fmt_raw_html(gt_tbl: Any, columns: Any = None) -> Any:
    """
    Render pre-built HTML in body cells without escaping.

    Great Tables v1.0.0 escapes the content of unformatted cells, so columns holding HTML that
    Pointblank generates must be marked for passthrough with `fmt_passthrough(escape=False)`.
    Earlier versions of Great Tables lack that method and never escaped cells, so there the table
    is returned unchanged. Because the last-applied formatter wins, call this before any other
    `fmt_*()` method that targets the same columns.
    """
    if hasattr(gt_tbl, "fmt_passthrough"):
        return gt_tbl.fmt_passthrough(columns=columns, escape=False)
    return gt_tbl


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


def _create_steps_html(steps_obj: Steps) -> str:
    import html as html_module

    n = len(steps_obj)
    header = (
        f"<div style='font-family: system-ui, -apple-system, sans-serif; max-width: 600px;'>"
        f"<div style='padding: 8px 12px; background-color: #f8f8f8; border: 1px solid #e0e0e0; "
        f"border-radius: 4px 4px 0 0; font-weight: 600; font-size: 13px; color: #333;'>"
        f"Steps &mdash; {n} step{'s' if n != 1 else ''}"
        f"</div>"
    )

    if n == 0:
        body = (
            "<div style='padding: 16px 12px; border: 1px solid #e0e0e0; border-top: none; "
            "border-radius: 0 0 4px 4px; color: #888; font-size: 12px; font-style: italic;'>"
            "No steps defined."
            "</div>"
        )
        return header + body + "</div>"

    rows = []
    for i, step in enumerate(steps_obj._steps, start=1):
        kwargs_parts = []
        for k, v in step.kwargs.items():
            if steps_obj._is_default(k, v):
                continue
            v_repr = html_module.escape(repr(v))
            kwargs_parts.append(
                f"<span style='color: #666;'>{html_module.escape(k)}</span>="
                f"<span style='color: #0550ae;'>{v_repr}</span>"
            )
        kwargs_str = ", ".join(kwargs_parts)

        bg = "#ffffff" if i % 2 == 1 else "#fafafa"
        radius = "0 0 4px 4px" if i == n else "0"
        rows.append(
            f"<div style='padding: 6px 12px; border: 1px solid #e0e0e0; border-top: none; "
            f"background-color: {bg}; font-size: 12px; border-radius: {radius};'>"
            f"<span style='color: #888; margin-right: 8px;'>{i}.</span>"
            f"<span style='font-weight: 600; color: #1a7f37;'>"
            f"{html_module.escape(step.method)}</span>"
            f"(<span style='font-size: 11px;'>{kwargs_str}</span>)"
            f"</div>"
        )

    return header + "".join(rows) + "</div>"
