from __future__ import annotations

import datetime
import decimal
import html as html_lib
import math
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from great_tables import GT, google_font, html, loc, style

from pointblank._constants import RTL_LANGUAGES
from pointblank._constants_translations import COMPARISON_REPORT_TEXT

if TYPE_CHECKING:
    from pointblank._compare import Comparison

# Colors for row statuses (shared by the status bar, legend, and section labels)
STATUS_COLORS = {
    "match": "#4CA64C",
    "changed": "#E8A33D",
    "missing": "#CF142B",
    "extra": "#3D7CC9",
    "dup_key": "#8E5BB5",
}

_MAX_VALUE_CHARS = 40
_CHANGED_VALUE_CHARS = 30  # (changed values, before and after)
_GRID_VALUE_CHARS = 24  # (values of missing and extra rows)
_MAX_PREVIEW_COLS = 8
# The width of the key column and grid of the missing and extra rows (in characters), so that
# they fit in a step report, and the gap between grid columns (14px is about 2 characters)
_GRID_CHARS = 108
_GAP_CHARS = 2
# (unquoted, since the styles are in single-quoted attributes)
_MONO = "font-family:IBM Plex Mono, ui-monospace, monospace;"
_SANS = "font-family:IBM Plex Sans, system-ui, sans-serif;"
_SCHEMA_COLOR = "#7A7A7A"


# ----------------------------------------------------------------------------------------------
# Localization
# ----------------------------------------------------------------------------------------------


def _lookup(entry: dict[str, str], lang: str) -> str | None:
    """Find a language's text in a translation entry (language codes match case-insensitively)."""
    if lang in entry:
        return entry[lang]
    lang_lower = lang.lower()
    for code, text in entry.items():
        if code.lower() == lang_lower:
            return text
    return None


def _locale_marks(locale: str) -> tuple[str, str]:
    """The thousands separator and decimal mark for a locale (falling back to `,` and `.`)."""
    try:
        from great_tables._formats import _get_locale_dec_mark, _get_locale_sep_mark

        return (
            _get_locale_sep_mark(default=",", use_seps=True, locale=locale),
            _get_locale_dec_mark(default=".", locale=locale),
        )
    except Exception:  # pragma: no cover
        return ",", "."


@dataclass(frozen=True)
class _Ctx:
    """Language and number-formatting settings for one rendering of the report."""

    lang: str = "en"
    locale: str = "en"
    unit: str = "rows"  # or "groups" (for `tbl_reconcile()`)

    def u(self, key: str) -> str:
        """The label key for the comparison's unit (e.g. `n_rows` -> `n_groups`)."""
        if self.unit == "groups":
            return key.replace("rows", "groups")
        return key

    @property
    def rtl(self) -> bool:
        return self.lang in RTL_LANGUAGES

    @property
    def arrow(self) -> str:
        """An arrow from source to target, in the direction of the text."""
        return "&larr;" if self.rtl else "&rarr;"

    def t(self, key: str, **fields: Any) -> str:
        """Translated text for `key` (English if there's no translation), with fields filled."""
        entry = COMPARISON_REPORT_TEXT.get(key, {})
        text = _lookup(entry, self.lang) or entry.get("en", key)
        return text.format(**fields) if fields else text

    def t_count(self, key: str, n: int, **fields: Any) -> str:
        """Like `t()`, but uses the singular form (`<key>_one`) for a count of 1 if available."""
        if n == 1:
            singular = _lookup(COMPARISON_REPORT_TEXT.get(f"{key}_one", {}), self.lang)
            if singular is not None:
                return singular.format(n=self.int(n), **fields)
        return self.t(key, n=self.int(n), **fields)

    def int(self, n: int) -> str:
        sep, _ = _locale_marks(self.locale)
        return f"{n:,}".replace(",", sep)

    def pct(self, pct: float) -> str:
        _, dec = _locale_marks(self.locale)
        if pct == 0:
            return "0%"
        if pct < 0.1:
            text = "<0.1%"
        elif 99.95 <= pct < 100:
            text = ">99.9%"
        else:
            text = f"{pct:.1f}%"
        return text.replace(".", dec)


# ----------------------------------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------------------------------


def _comparison_report(
    cmp: Comparison,
    limit: int = 10,
    title: str | None = None,
    lang: str = "en",
    locale: str | None = None,
) -> GT:
    """Build the tabular (GT) report for a `Comparison`."""
    from pointblank._utils import _is_lib_present

    ctx = _Ctx(lang=lang, locale=locale or lang, unit=cmp.unit)

    # Each section, with the color of its label
    sections = [
        (_schema_rows(cmp, ctx), _SCHEMA_COLOR),
        (_column_rows(cmp, ctx), STATUS_COLORS["changed"]),
        (_changed_rows(cmp, limit, ctx), STATUS_COLORS["changed"]),
        (_missing_extra_rows(cmp, "missing", limit, ctx), STATUS_COLORS["missing"]),
        (_missing_extra_rows(cmp, "extra", limit, ctx), STATUS_COLORS["extra"]),
        (_dup_rows(cmp, limit, ctx), STATUS_COLORS["dup_key"]),
    ]
    if not any(section for section, _ in sections):
        if cmp.n_source_rows == 0 and cmp.n_target_rows == 0:
            detail = ctx.t("summary_both_empty")
        else:
            detail = ctx.t_count(ctx.u("all_rows_match"), cmp.n)
        result = {
            "group": ctx.t("group_result"),
            "item": _badge("✓", STATUS_COLORS["match"]),
            "detail": detail,
        }
        sections = [([result], STATUS_COLORS["match"])]

    rows: list[dict[str, Any]] = []
    group_colors: dict[str, str] = {}
    for section, color in sections:
        for row in section:
            group_colors.setdefault(row["group"], color)
        rows += section
    # (header rows label the columns of a section's grid)
    header_rows = [i for i, row in enumerate(rows) if row.pop("header", False)]

    if _is_lib_present("polars"):
        import polars as pl

        df: Any = pl.DataFrame(rows, schema=["group", "item", "detail"], orient="row")
    else:  # pragma: no cover
        import pandas as pd

        df = pd.DataFrame(rows, columns=["group", "item", "detail"])

    from pointblank._utils_html import _fmt_raw_html

    # The cells hold HTML built here (Great Tables v1.0.0+ escapes cell content by default)
    gt_tbl = (
        _fmt_raw_html(
            GT(df, groupname_col="group", id="pb_comparison_tbl"), columns=["item", "detail"]
        )
        .tab_header(
            title=html(title or ctx.t("title")),
            subtitle=html(_header_html(cmp, lang=ctx.lang, locale=ctx.locale)),
        )
        .cols_label(item="", detail="")
        .opt_table_font(font=google_font("IBM Plex Sans"))
        .opt_align_table_header(align="right" if ctx.rtl else "left")
        .tab_style(
            # The key column is as narrow as its contents
            style=[
                style.text(font=google_font("IBM Plex Mono"), size="12px"),
                style.css("white-space: nowrap; padding-inline-end: 16px; vertical-align: top;"),
            ],
            locations=loc.body(columns="item"),
        )
        .tab_style(
            style=[style.text(size="12px"), style.css("width: 100%; vertical-align: top;")],
            locations=loc.body(columns="detail"),
        )
        .tab_style(
            style=[style.text(weight="bold", size="12px"), style.fill(color="#F6F7F8")],
            locations=loc.row_groups(),
        )
        .tab_options(column_labels_hidden=True, table_width="100%")
    )
    for group, color in group_colors.items():
        gt_tbl = gt_tbl.tab_style(
            style=style.css(f"border-inline-start: 4px solid {color};"),
            locations=loc.row_groups(rows=[group]),
        )
    if header_rows:
        gt_tbl = gt_tbl.tab_style(
            style=[
                style.text(color="#8A8A8A"),
                style.css("padding-top: 4px; padding-bottom: 2px; border-bottom-style: none;"),
            ],
            locations=loc.body(rows=header_rows),
        )
    if ctx.rtl:
        # (so that the columns run from right to left, too)
        gt_tbl = gt_tbl.opt_css("#pb_comparison_tbl .gt_table { direction: rtl; }")
        gt_tbl = gt_tbl.tab_style(
            style=style.css("direction: rtl; text-align: right;"),
            locations=[loc.body(), loc.row_groups()],
        )
    return gt_tbl


# ----------------------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------------------


def _header_html(cmp: Comparison, lang: str = "en", locale: str | None = None) -> str:
    ctx = _Ctx(lang=lang, locale=locale or lang, unit=cmp.unit)
    esc = html_lib.escape
    if cmp.mode == "keyed":
        keys = ", ".join(f"<code>{esc(k)}</code>" for k in cmp.keys)
        alignment = ctx.t("aligned_by_key", keys=keys)
    elif cmp.mode == "multiset":
        alignment = ctx.t("aligned_multiset")
    else:
        alignment = ctx.t("aligned_by_position")

    tables_line = (
        f"<strong>{esc(cmp.source_name)}</strong> ({ctx.t_count(ctx.u('n_rows'), cmp.n_source_rows)}) "
        f"{ctx.arrow} <strong>{esc(cmp.target_name)}</strong> "
        f"({ctx.t_count(ctx.u('n_rows'), cmp.n_target_rows)}), {alignment}"
    )

    if cmp.identical:
        verdict = _badge(ctx.t("identical"), STATUS_COLORS["match"])
    else:
        pct = (cmp.n_failed / cmp.n * 100) if cmp.n else 0.0
        verdict = (
            _badge(ctx.t("different"), STATUS_COLORS["missing"])
            + " "
            + ctx.t(
                ctx.u("rows_fail"),
                n_failed=ctx.int(cmp.n_failed),
                n=ctx.int(cmp.n),
                pct=ctx.pct(pct),
            )
        )
        if not cmp.schema_ok:
            verdict += " &middot; " + _badge(ctx.t("schema_differs"), STATUS_COLORS["missing"])
        if cmp._nothing_compared:
            verdict += " &middot; " + _badge(ctx.t("no_columns_compared"), STATUS_COLORS["missing"])
            verdict += f" <span style='color:#666;'>{ctx.t('no_columns_note')}</span>"
        elif not cmp.schema_ok:
            verdict += f" <span style='color:#666;'>{ctx.t('strict_schema_note')}</span>"

    lines = [tables_line, verdict]
    summary = _summary_text(cmp, ctx)
    if summary is not None:
        lines.append(f"<span style='color:#444;'>{esc(summary)}</span>")
    if not cmp.identical and cmp.n > 0:
        # (an identical comparison is all matches, which the verdict already says, and with no
        # rows there's nothing to break down)
        lines += [_status_bar(cmp, ctx), _legend(cmp, ctx)]
    spec = _spec_badges(cmp, ctx)
    if spec:
        lines.append(spec)

    direction = " direction:rtl; text-align:right;" if ctx.rtl else ""
    return (
        f"<div style='font-size:13px; line-height:1.9;{direction}'>"
        + "".join(f"<div>{line}</div>" for line in lines)
        + "</div>"
    )


def _summary_text(cmp: Comparison, ctx: _Ctx) -> str | None:
    """
    One plain-text sentence on how different the tables are, from a single changed value to
    tables with nothing in common (`None` if the verdict and the counts say it all).
    """
    if cmp.identical:
        return None
    if cmp.n_source_rows == 0 and cmp.n_target_rows == 0:
        return ctx.t("summary_both_empty")
    if cmp.n_target_rows == 0:
        return ctx.t("summary_target_empty")
    if cmp.n_source_rows == 0:
        return ctx.t("summary_source_empty")
    if cmp._nothing_compared:
        # (the verdict says so)
        return None
    if _nothing_in_common(cmp):
        return ctx.t(
            "summary_no_keys_in_common" if cmp.mode == "keyed" else "summary_no_rows_in_common"
        )
    if cmp.mode == "keyed" and not cmp.compared_columns:
        return ctx.t("summary_only_keys_in_common")
    if cmp.n_changed > 0:
        n_values = sum(cmp.column_mismatches.values())
        total = (cmp.n_match + cmp.n_changed) * len(cmp.compared_columns)
        if n_values >= total:
            return ctx.t("summary_every_value_differs")
        return ctx.t_count("summary_values_differ", n_values, total=ctx.int(total))
    return None


def _nothing_in_common(cmp: Comparison) -> bool:
    """Whether no row of either (non-empty) table aligns with a row of the other."""
    return (
        cmp.n_source_rows > 0
        and cmp.n_target_rows > 0
        and cmp.n_match + cmp.n_changed == 0
        and cmp.n_dup_key == 0
    )


def _status_bar(cmp: Comparison, ctx: _Ctx) -> str:
    total = cmp.n
    if total == 0:
        return f"<span style='color:#666;'>{ctx.t('no_rows')}</span>"
    segments = []
    for status, count in cmp.status_counts.items():
        if count == 0:
            continue
        width = max(count / total * 100, 0.5)
        label = html_lib.escape(ctx.t(f"status_{status}"), quote=True)
        segments.append(
            f"<div title='{label}: {ctx.int(count)}' style='width:{width:.3f}%; "
            f"background:{STATUS_COLORS[status]}; height:100%;'></div>"
        )
    return (
        "<div style='display:flex; width:100%; max-width:600px; height:8px; border-radius:2px; "
        "overflow:hidden; background:#EEE; margin:6px 0;'>" + "".join(segments) + "</div>"
    )


def _legend(cmp: Comparison, ctx: _Ctx) -> str:
    parts = []
    for status, count in cmp.status_counts.items():
        if count == 0 and status == "dup_key":
            continue
        dot = (
            f"<span style='display:inline-block; width:9px; height:9px; border-radius:2px; "
            f"background:{STATUS_COLORS[status]}; margin-right:4px; margin-left:4px;'></span>"
        )
        # Statuses with no rows are dimmed, so the ones that occur stand out
        dim = " color:#999; opacity:0.6;" if count == 0 else ""
        parts.append(
            f"<span style='white-space:nowrap; margin-right:12px;{dim}'>{dot}"
            f"{ctx.t(f'status_{status}')} <strong>{ctx.int(count)}</strong></span>"
        )
    return "".join(parts)


def _spec_badges(cmp: Comparison, ctx: _Ctx) -> str:
    esc = html_lib.escape
    badges = []
    if cmp.aggregation is not None:
        aggs = "/".join(f"<code>{esc(a)}</code>" for a in cmp.aggregation["aggs"])
        if cmp.aggregation["keys"]:
            keys = ", ".join(f"<code>{esc(k)}</code>" for k in cmp.aggregation["keys"])
            badges.append(_chip(ctx.t("spec_aggregated_by", aggs=aggs, keys=keys)))
        else:
            badges.append(_chip(ctx.t("spec_aggregated_totals", aggs=aggs)))
    tolerant = [s.source for s in cmp._specs if s.atol is not None or s.rtol is not None]
    if tolerant:
        badges.append(_chip(ctx.t("spec_tolerance", columns=esc(", ".join(tolerant)))))
    normalized = [s.source for s in cmp._specs if s.normalize]
    if normalized:
        badges.append(_chip(ctx.t("spec_normalized", columns=esc(", ".join(normalized)))))
    if not cmp.null_equal:
        badges.append(_chip("null &ne; null"))
    if cmp.schema_mode == "common" and cmp.aggregation is None:
        # (reconciliations always compare the shared aggregate columns, so it isn't shown)
        badges.append(_chip(ctx.t("spec_common_schema")))
    if cmp.dup_keys_mode == "compare":
        badges.append(_chip(ctx.t("spec_dup_compare")))
    if cmp.partitions > 1:
        badges.append(_chip(ctx.t("spec_partitions", n=ctx.int(cmp.partitions))))
    n_src_cols = len(cmp._src_columns) - len(cmp.keys)
    if len(cmp.compared_columns) < n_src_cols:
        badges.append(
            _chip(
                ctx.t(
                    "spec_columns_compared",
                    n=ctx.int(len(cmp.compared_columns)),
                    total=ctx.int(n_src_cols),
                )
            )
        )
    return " ".join(badges)


# ----------------------------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------------------------


def _schema_rows(cmp: Comparison, ctx: _Ctx) -> list[dict[str, str]]:
    diff = cmp.schema_diff
    group = ctx.t("group_schema")
    esc = html_lib.escape
    if not cmp.compared_columns and not (diff.dtype_changed or diff.renamed or diff.case_only):
        # With no columns in common, each table's columns are listed in one row
        out = [
            {
                "group": group,
                # (in the text font, which the key column doesn't use)
                "item": f"<span style='{_SANS}'>{_chip(ctx.t(label), color)}</span>",
                "detail": ", ".join(f"<code>{esc(c)}</code>" for c in cols),
            }
            for cols, label, color in [
                (diff.only_in_source, "only_in_source", STATUS_COLORS["missing"]),
                (diff.only_in_target, "only_in_target", STATUS_COLORS["extra"]),
            ]
            if cols
        ]
    else:
        out = _unshared_column_rows(cmp, group, ctx)
    for col, s_dtype, t_dtype in diff.dtype_changed:
        out.append(
            {
                "group": group,
                "item": esc(col),
                "detail": ctx.t(
                    "type_changed",
                    source=f"<code>{esc(s_dtype)}</code>",
                    target=f"<code>{esc(t_dtype)}</code>",
                ),
            }
        )
    for s_col, t_col in diff.renamed:
        out.append(
            {
                "group": group,
                "item": esc(s_col),
                "detail": ctx.t(
                    "renamed", column=f"<code>{esc(t_col)}</code>", arg="<code>column_map=</code>"
                ),
            }
        )
    if diff.reordered:
        out.append(
            {
                "group": group,
                "item": ctx.t("column_order"),
                "detail": ctx.t("column_order_detail"),
            }
        )
    return out


def _unshared_column_rows(cmp: Comparison, group: str, ctx: _Ctx) -> list[dict[str, str]]:
    """One row per column that's in only one of the tables."""
    diff = cmp.schema_diff
    esc = html_lib.escape
    case_pairs = dict(diff.case_only)
    out = []
    for col in diff.only_in_source:
        hint = ""
        if col in case_pairs:
            hint = f" ({ctx.t('case_only_hint', column=f'<code>{esc(case_pairs[col])}</code>')})"
        out.append(
            {
                "group": group,
                "item": esc(col),
                "detail": _chip(ctx.t("only_in_source"), STATUS_COLORS["missing"]) + hint,
            }
        )
    for col in diff.only_in_target:
        out.append(
            {
                "group": group,
                "item": esc(col),
                "detail": _chip(ctx.t("only_in_target"), STATUS_COLORS["extra"]),
            }
        )
    return out


def _column_rows(cmp: Comparison, ctx: _Ctx) -> list[dict[str, str]]:
    n_matched = cmp.n_match + cmp.n_changed
    mismatched = sorted(
        ((col, n) for col, n in cmp.column_mismatches.items() if n > 0),
        key=lambda x: -x[1],
    )
    if not mismatched:
        return []
    group = ctx.t(
        "group_columns", n=ctx.int(len(mismatched)), total=ctx.int(len(cmp.compared_columns))
    )
    target_of = {s.source: s.target for s in cmp._specs}
    out = []
    for col, n in mismatched:
        pct = n / n_matched * 100 if n_matched else 0.0
        bar = (
            "<div style='display:inline-block; vertical-align:middle; width:120px; height:7px; "
            "background:#EEE; border-radius:2px; margin-right:8px; margin-left:8px; "
            "overflow:hidden;'>"
            f"<div style='width:{max(pct, 1):.2f}%; height:100%; "
            f"background:{STATUS_COLORS['changed']};'></div></div>"
        )
        item = html_lib.escape(col)
        if target_of[col] != col:
            item += f" {ctx.arrow} {html_lib.escape(target_of[col])}"
        detail = ctx.t(
            ctx.u("matched_rows_differ"), n=ctx.int(n), total=ctx.int(n_matched), pct=ctx.pct(pct)
        )
        out.append({"group": group, "item": item, "detail": f"{bar}{detail}"})
    return out


def _changed_rows(cmp: Comparison, limit: int, ctx: _Ctx) -> list[dict[str, Any]]:
    if cmp.n_changed == 0:
        return []
    frame = cmp._fetch("changed", limit)
    title = ctx.t("group_changed_groups" if ctx.unit == "groups" else "group_changed")
    group = _sample_group(title, len(frame), cmp.n_changed, ctx)
    key_cols = cmp._key_out_names()
    records = list(frame.rows(named=True))

    # One line per changed value (column, old value, new value, and the difference), aligned
    # across all of the section's rows
    row_lines = []
    for row in records:
        changed = [spec for i, spec in enumerate(cmp._specs) if row.get(f"__pb_d_{i}")]
        lines = []
        for spec in changed[:_MAX_PREVIEW_COLS]:
            old = row[f"{spec.source}_source"]
            new = row[f"{spec.source}_target"]
            old_text, new_text = _pair_texts(old, new, _CHANGED_VALUE_CHARS)
            line = [
                _Cell(html_lib.escape(spec.source), spec.source, style="color:#666;"),
                _Cell(
                    "<span style='color:#555; text-decoration:line-through; "
                    f"text-decoration-color:#AAA;'>{_fmt_pair_value(old_text, old)}</span>",
                    old_text,
                    gap=6,
                ),
                _Cell(ctx.arrow, "→", gap=6),
                # (one more character for the highlight's padding)
                _Cell(
                    "<span style='background:#FDF1DC; padding:0 3px; border-radius:2px;'>"
                    f"{_fmt_pair_value(new_text, new)}</span>",
                    new_text + " ",
                    gap=6,
                ),
            ]
            delta = _delta_text(old, new)
            if delta is not None:
                line.append(
                    _Cell(
                        f"<span style='color:#888; font-size:11px;'>"
                        f"({html_lib.escape(delta)})</span>",
                        f"({delta})",
                    )
                )
            lines.append(line)
        if len(changed) > len(lines):
            more = ctx.t("n_more", n=ctx.int(len(changed) - len(lines)))
            lines.append([_Cell(f"<span style='color:#999;'>{more}</span>", more, ltr=False)])
        row_lines.append(lines)

    # The table names label the old and new values
    header = [
        _Cell("", ""),
        _header_cell(cmp.source_name),
        _Cell("", "", gap=6),
        _header_cell(cmp.target_name),
    ]
    widths = _grid_widths([header] + [line for lines in row_lines for line in lines])
    keys = _KeyGrid(cmp, key_cols, records, ctx)
    out = [keys.header_row(group, _grid_line(header, widths))]
    for row, lines in zip(records, row_lines):
        detail = "".join(_grid_line(line, widths) for line in lines)
        out.append({"group": group, "item": keys.cell(row), "detail": detail})
    return out


def _missing_extra_rows(
    cmp: Comparison, status: str, limit: int, ctx: _Ctx
) -> list[dict[str, Any]]:
    count = cmp.n_missing if status == "missing" else cmp.n_extra
    if count == 0:
        return []
    title = ctx.t("group_missing" if status == "missing" else "group_extra")
    other_rows = cmp.n_target_rows if status == "missing" else cmp.n_source_rows
    if other_rows == 0 or cmp._nothing_compared:
        # Every row of one table is missing from the other (which is empty), so a sample would
        # only repeat the table; or there are no values to show
        if other_rows == 0:
            note = ctx.t("summary_target_empty" if status == "missing" else "summary_source_empty")
        else:
            note = ctx.t("no_values_note")
        return [
            {
                "group": _sample_group(title, count, count, ctx),
                "item": "",
                "detail": f"<span style='color:#666;'>{note}</span>",
            }
        ]
    frame = cmp._fetch(status, limit)
    group = _sample_group(title, len(frame), count, ctx)
    key_cols = cmp._key_out_names()
    records = list(frame.rows(named=True))
    if cmp.mode == "keyed" and _nothing_in_common(cmp):
        # With no keys in common, the keys alone show how the tables differ (e.g. an offset, or
        # keys formatted differently), so they're listed inline
        sep = " <span style='color:#999;'>&middot;</span> "
        keys = sep.join(
            "<span style='white-space:nowrap;'><bdi dir='ltr'>"
            + _key_text(row, key_cols)
            + "</bdi></span>"
            for row in records
        )
        if any(_is_null(row[k]) for row in records for k in key_cols):
            keys += f" <span style='color:#999; font-size:11px;'>{ctx.t('null_key_note')}</span>"
        names = html_lib.escape(", ".join(key_cols))
        return [
            {
                "group": group,
                "item": f"<span style='color:#8A8A8A;'>{names}</span>",
                "detail": f"<div style='line-height:1.7; {_MONO}'>{keys}</div>",
            }
        ]
    if cmp.mode == "multiset":
        exclude = {"n_source", "n_target"}
        keys = None
    else:
        exclude = set(key_cols)
        keys = _KeyGrid(cmp, key_cols, records, ctx)

    # A grid of the rows' values, under a line of column names: as many columns as fit beside the
    # key column (and at least one)
    cols = [c for c in frame.columns if c not in exclude and not c.startswith("__pb_")]
    room = _GRID_CHARS - (keys.width if keys is not None else 0)
    shown = []
    for c in cols[:_MAX_PREVIEW_COLS]:
        width = _grid_widths([[_header_cell(c)]] + [[_value_cell(row[c])] for row in records])[0]
        if shown and width + _GAP_CHARS > room:
            break
        shown.append(c)
        room -= width + _GAP_CHARS
    header = [_header_cell(c) for c in shown]
    if len(cols) > len(shown):
        more = ctx.t("n_more", n=ctx.int(len(cols) - len(shown)))
        header.append(_Cell(more, more, ltr=False))
    lines = [[_value_cell(row[c]) for c in shown] for row in records]
    widths = _grid_widths([header] + lines)

    out = []
    if shown:
        header_html = _grid_line(header, widths)
        if keys is not None:
            out.append(keys.header_row(group, header_html))
        else:
            out.append({"group": group, "item": "", "detail": header_html, "header": True})
    for row, line in zip(records, lines):
        if keys is not None:
            item = keys.cell(row)
        else:
            # Distinct rows, with how many surplus copies one table has over the other (when
            # there's more than one)
            copies = abs(row["n_source"] - row["n_target"])
            item = f"&times;{ctx.int(copies)}" if copies > 1 else ""
        out.append({"group": group, "item": item, "detail": _grid_line(line, widths)})
    return out


def _dup_rows(cmp: Comparison, limit: int, ctx: _Ctx) -> list[dict[str, Any]]:
    if cmp.n_dup_key == 0:
        return []
    frame = cmp._fetch("dup_key", limit)
    group = _sample_group(ctx.t("group_dup"), len(frame), cmp.n_dup_key_values, ctx, noun="keys")
    key_cols = list(cmp.keys)
    records = list(frame.rows(named=True))
    keys = _KeyGrid(cmp, key_cols, records, ctx)

    pair_diffs: dict[tuple[Any, ...], set[str]] | None = None
    if cmp.dup_keys_mode == "compare":
        pairs = cmp._dup_pairs(limit)
        pair_diffs = {}
        if pairs is not None:
            for prow in pairs.rows(named=True):
                k = tuple(prow[c] for c in key_cols)
                cols = pair_diffs.setdefault(k, set())
                for i, spec in enumerate(cmp._specs):
                    if prow.get(f"__pb_d_{i}"):
                        cols.add(spec.source)

    out = [keys.header_row(group, "")]
    for row in records:
        detail = ctx.t(
            "dup_counts", n_source=ctx.int(row["n_source"]), n_target=ctx.int(row["n_target"])
        )
        if pair_diffs is not None:
            k = tuple(row[c] for c in key_cols)
            if row["n_source"] == 0 or row["n_target"] == 0:
                detail += f" &middot; <span style='color:#666;'>{ctx.t('no_pairs')}</span>"
            elif pair_diffs.get(k):
                ordered = [s.source for s in cmp._specs if s.source in pair_diffs[k]]
                columns = ", ".join(f"<code>{html_lib.escape(c)}</code>" for c in ordered)
                detail += " &middot; " + ctx.t("pairs_differ_in", columns=columns)
            else:
                detail += " &middot; " + ctx.t("all_pairs_identical")
        out.append({"group": group, "item": keys.cell(row), "detail": detail})
    return out


# ----------------------------------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cell:
    """A cell of a section's grid: its HTML, and its plain text (which sets the column width)."""

    html: str
    text: str
    title: str | None = None
    style: str = ""
    gap: int = 14  # pixels before the next cell
    ltr: bool = True  # (data, as opposed to translated text, reads left to right in any language)


def _text_width(text: str) -> int:
    """The width of text in a monospaced font, in characters (wide characters count twice)."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _header_cell(name: str) -> _Cell:
    """A column (or table) name above a grid column, shortened if it's long."""
    text = _clip(name, _GRID_VALUE_CHARS)
    return _Cell(html_lib.escape(text), text, title=name if text != name else None)


def _value_cell(value: Any, width: int = _GRID_VALUE_CHARS) -> _Cell:
    """A value in a grid, with the full value as a tooltip if it's shortened."""
    text = _value_text(value, width)
    full = _value_text(value, 500)
    return _Cell(_fmt_value(value, width), text, title=full if full != text else None)


def _grid_widths(lines: list[list[_Cell]]) -> list[int]:
    """The width of each grid column (in characters), from the widest cell in the column."""
    n = max((len(line) for line in lines), default=0)
    return [
        max((_text_width(line[i].text) for line in lines if i < len(line)), default=0)
        for i in range(n)
    ]


def _grid_line(cells: list[_Cell], widths: list[int]) -> str:
    """
    One line of a grid. Every cell but the last has its column's width, so that the columns of
    every line with the same widths line up (the font is monospaced, so `ch` units are exact).
    """
    parts = []
    for i, cell in enumerate(cells):
        width = f" width:{widths[i]}ch;" if i < len(cells) - 1 else ""
        title = f" title='{html_lib.escape(cell.title, quote=True)}'" if cell.title else ""
        # (isolated inside the cell, so that the cell's margin follows the report's direction)
        content = f"<bdi dir='ltr'>{cell.html}</bdi>" if cell.ltr and cell.html else cell.html
        parts.append(
            f"<span{title} style='display:inline-block;{width} overflow:hidden; "
            f"text-overflow:ellipsis; vertical-align:top; margin-inline-end:{cell.gap}px;"
            f"{cell.style}'>{content}</span>"
        )
    return f"<div style='white-space:nowrap; line-height:1.7; {_MONO}'>{''.join(parts)}</div>"


class _KeyGrid:
    """
    The key column of a section: the key values of each row (aligned, if there are several keys)
    and a header row naming the key columns.
    """

    def __init__(self, cmp: Comparison, key_cols: list[str], records: list[dict], ctx: _Ctx):
        self.ctx = ctx
        self.key_cols = key_cols
        self.positional = key_cols == ["_row_num_"]
        self.header = [] if self.positional else [_header_cell(k) for k in key_cols]
        self.widths = _grid_widths([self.header] + [self._cells(row) for row in records])

    def _cells(self, row: dict[str, Any]) -> list[_Cell]:
        if self.positional:
            label = self.ctx.t("row_n", n=self.ctx.int(row["_row_num_"]))
            return [_Cell(html_lib.escape(label), label, ltr=False)]
        cells = [_value_cell(row[k], _MAX_VALUE_CHARS) for k in self.key_cols]
        if any(_is_null(row[k]) for k in self.key_cols):
            note = self.ctx.t("null_key_note")
            cells.append(_Cell(note, note, style="color:#999; font-size:11px;", ltr=False))
        return cells

    @property
    def width(self) -> int:
        """The width of the key column (in characters, roughly counting the gaps between keys)."""
        return sum(self.widths) + _GAP_CHARS * (len(self.widths) - 1)

    def cell(self, row: dict[str, Any]) -> str:
        return _grid_line(self._cells(row), self.widths)

    def header_row(self, group: str, detail: str) -> dict[str, Any]:
        """A row labeling the key column and the grid columns of the section."""
        item = _grid_line(self.header, self.widths) if self.header else ""
        return {"group": group, "item": item, "detail": detail, "header": True}


# ----------------------------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------------------------


def _sample_group(title: str, shown: int, total: int, ctx: _Ctx, noun: str | None = None) -> str:
    """A section title with its count, e.g. 'Changed rows (showing 10 of 400 rows)'."""
    noun = noun or ctx.unit
    if shown >= total:
        count = ctx.t_count(f"n_{noun}", total)
    else:
        count = ctx.t(f"showing_{noun}", shown=ctx.int(shown), total=ctx.int(total))
    return f"{title} ({count})"


def _key_text(row: dict[str, Any], key_cols: list[str]) -> str:
    """A row's key as HTML: its value, or a tuple of values for a composite key."""
    values = [_fmt_value(row[k]) for k in key_cols]
    return values[0] if len(values) == 1 else f"({', '.join(values)})"


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color}; color:white; padding:1px 6px; border-radius:3px; "
        f"font-size:11px; font-weight:bold; letter-spacing:0.3px;'>{text}</span>"
    )


def _chip(text: str, color: str = "#555") -> str:
    return (
        f"<span style='border:1px solid {color}; color:{color}; padding:0 5px; "
        f"border-radius:3px; font-size:11px; white-space:nowrap;'>{text}</span>"
    )


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not _is_null(value)


def _float_text(value: float, digits: int = 6) -> str:
    """A float with `digits` significant digits, in positional notation unless it's very large or
    small (so that 1234567.0 is `1234567` rather than `1.23457e+06`)."""
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    magnitude = math.floor(math.log10(abs(value)))
    if not -5 <= magnitude < 15:
        return f"{value:.{digits}g}"
    text = f"{value:.{max(digits - 1 - magnitude, 0)}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _clip(text: str, width: int, start: int = 0) -> str:
    """At most `width` characters of `text` from `start`, with an ellipsis where it's cut."""
    if len(text) <= width:
        return text
    if start <= 0:
        return text[: width - 1] + "…"
    tail = text[min(start, len(text) - (width - 1)) :]
    if len(tail) <= width - 1:
        return "…" + tail
    return "…" + tail[: width - 2] + "…"


def _value_text(value: Any, width: int = _MAX_VALUE_CHARS) -> str:
    """The plain-text display of a value (`NULL` for nulls)."""
    if _is_null(value):
        return "NULL"
    if isinstance(value, float):
        text = _float_text(value)
    elif isinstance(value, datetime.datetime):
        text = value.isoformat(sep=" ")
    else:
        text = str(value)
    return _clip(text, width)


def _visible_whitespace(text: str) -> str:
    """A quoted string with escaped tabs and newlines, and spaces shown as `␣`."""
    return repr(text).replace(" ", "␣")


def _pair_texts(old: Any, new: Any, width: int = _MAX_VALUE_CHARS) -> tuple[str, str]:
    """
    The plain-text display of a changed value, before and after. Unlike formatting each value on
    its own, this makes sure that the difference is visible: floats get as many digits as it takes
    to tell them apart, long strings are shown around the first difference, and strings that differ
    only in whitespace are quoted with their whitespace escaped.
    """
    if _is_null(old) or _is_null(new):
        return _value_text(old, width), _value_text(new, width)
    if isinstance(old, float) and isinstance(new, float):
        for digits in range(6, 17):
            a, b = _float_text(old, digits), _float_text(new, digits)
            if a != b:
                return a, b
        return repr(old), repr(new)
    if isinstance(old, str) and isinstance(new, str):
        if old != new and old.split() == new.split():
            return _clip(_visible_whitespace(old), width), _clip(_visible_whitespace(new), width)
        first = next(
            (i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new))
        )
        if first >= width - 1 and max(len(old), len(new)) > width:
            # The difference would be cut off, so show the text around it
            start = max(first - width // 3, 1)
            return _clip(old, width, start), _clip(new, width, start)
    return _value_text(old, width), _value_text(new, width)


def _duration_text(seconds: float) -> str:
    """A signed duration in its two largest units, e.g. `+3h`, `-1d 2h`, or `+250ms`."""
    sign = "-" if seconds < 0 else "+"
    s = abs(seconds)
    if 0 < s < 1e-3:
        return f"{sign}{s * 1e6:g}µs"
    if 0 < s < 1:
        return f"{sign}{s * 1e3:g}ms"
    parts = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= size:
            n = int(s // size)
            parts.append(f"{n}{unit}")
            s -= n * size
    if s or not parts:
        parts.append(f"{s:g}s")
    return sign + " ".join(parts[:2])


def _delta_text(old: Any, new: Any) -> str | None:
    """The difference between two numbers, datetimes, dates, or durations (`None` otherwise)."""
    if _is_number(old) and _is_number(new):
        delta = new - old
        if isinstance(delta, int):
            return f"{delta:+d}"
        text = _float_text(delta)
        return text if text.startswith("-") else "+" + text
    if isinstance(old, decimal.Decimal) and isinstance(new, decimal.Decimal):
        return f"{new - old:+f}"
    temporal = (datetime.datetime, datetime.timedelta)
    if isinstance(old, temporal) and type(old) is type(new):
        try:
            return _duration_text((new - old).total_seconds())
        except TypeError:  # e.g., timezone-aware and naive datetimes
            return None
    if isinstance(old, datetime.date) and type(old) is type(new):
        return f"{(new - old).days:+,}d"
    return None


def _fmt_value(value: Any, width: int = _MAX_VALUE_CHARS) -> str:
    if _is_null(value):
        return "<span style='color:#999; font-style:italic;'>NULL</span>"
    return html_lib.escape(_value_text(value, width))


def _fmt_pair_value(text: str, value: Any) -> str:
    if _is_null(value):
        return _fmt_value(value)
    return html_lib.escape(text)
