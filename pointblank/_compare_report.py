from __future__ import annotations

import datetime
import html as html_lib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from great_tables import GT, google_font, html, loc, style

from pointblank._constants import RTL_LANGUAGES
from pointblank._constants_translations import COMPARISON_REPORT_TEXT

if TYPE_CHECKING:
    from pointblank.compare import Comparison

# Colors for row statuses (shared by the status bar, legend, and section labels)
STATUS_COLORS = {
    "match": "#4CA64C",
    "changed": "#E8A33D",
    "missing": "#CF142B",
    "extra": "#3D7CC9",
    "dup_key": "#8E5BB5",
}

_MAX_VALUE_CHARS = 40
_MAX_PREVIEW_COLS = 8


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

    @property
    def rtl(self) -> bool:
        return self.lang in RTL_LANGUAGES

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

    ctx = _Ctx(lang=lang, locale=locale or lang)

    rows: list[dict[str, str]] = []
    rows += _schema_rows(cmp, ctx)
    rows += _column_rows(cmp, ctx)
    rows += _changed_rows(cmp, limit, ctx)
    rows += _missing_extra_rows(cmp, "missing", limit, ctx)
    rows += _missing_extra_rows(cmp, "extra", limit, ctx)
    rows += _dup_rows(cmp, limit, ctx)

    if not rows:
        rows.append(
            {
                "group": ctx.t("group_result"),
                "item": _badge("✓", STATUS_COLORS["match"]),
                "detail": ctx.t_count("all_rows_match", cmp.n),
            }
        )

    if _is_lib_present("polars"):
        import polars as pl

        df: Any = pl.DataFrame(rows, schema=["group", "item", "detail"], orient="row")
    else:  # pragma: no cover
        import pandas as pd

        df = pd.DataFrame(rows, columns=["group", "item", "detail"])

    gt_tbl = (
        GT(df, groupname_col="group", id="pb_comparison_tbl")
        .tab_header(
            title=html(title or ctx.t("title")),
            subtitle=html(_header_html(cmp, lang=ctx.lang, locale=ctx.locale)),
        )
        .cols_label(item="", detail="")
        .opt_table_font(font=google_font("IBM Plex Sans"))
        .opt_align_table_header(align="right" if ctx.rtl else "left")
        .tab_style(
            style=style.text(font=google_font("IBM Plex Mono"), size="12px"),
            locations=loc.body(columns="item"),
        )
        .tab_style(style=style.text(size="12px"), locations=loc.body(columns="detail"))
        .tab_style(
            style=[style.text(weight="bold", size="12px"), style.fill(color="#F6F7F8")],
            locations=loc.row_groups(),
        )
        .cols_width(cases={"item": "30%", "detail": "70%"})
        .tab_options(column_labels_hidden=True, table_width="100%")
    )
    if ctx.rtl:
        gt_tbl = gt_tbl.tab_style(
            style=style.css("direction: rtl; text-align: right;"),
            locations=[loc.body(), loc.row_groups()],
        )
    return gt_tbl


# ----------------------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------------------


def _header_html(cmp: Comparison, lang: str = "en", locale: str | None = None) -> str:
    ctx = _Ctx(lang=lang, locale=locale or lang)
    esc = html_lib.escape
    if cmp.mode == "keyed":
        keys = ", ".join(f"<code>{esc(k)}</code>" for k in cmp.keys)
        alignment = ctx.t("aligned_by_key", keys=keys)
    elif cmp.mode == "multiset":
        alignment = ctx.t("aligned_multiset")
    else:
        alignment = ctx.t("aligned_by_position")

    tables_line = (
        f"<strong>{esc(cmp.source_name)}</strong> ({ctx.t_count('n_rows', cmp.n_source_rows)}) "
        f"&rarr; <strong>{esc(cmp.target_name)}</strong> "
        f"({ctx.t_count('n_rows', cmp.n_target_rows)}), {alignment}"
    )

    if cmp.identical:
        verdict = _badge(ctx.t("identical"), STATUS_COLORS["match"])
    else:
        pct = (cmp.n_failed / cmp.n * 100) if cmp.n else 0.0
        verdict = (
            _badge(ctx.t("different"), STATUS_COLORS["missing"])
            + " "
            + ctx.t("rows_fail", n_failed=ctx.int(cmp.n_failed), n=ctx.int(cmp.n), pct=ctx.pct(pct))
        )
        if not cmp.schema_ok:
            verdict += " &middot; " + _badge(ctx.t("schema_differs"), STATUS_COLORS["missing"])
            verdict += f" <span style='color:#666;'>{ctx.t('strict_schema_note')}</span>"

    lines = [tables_line, verdict, _status_bar(cmp, ctx), _legend(cmp, ctx)]
    spec = _spec_badges(cmp, ctx)
    if spec:
        lines.append(spec)

    direction = " direction:rtl; text-align:right;" if ctx.rtl else ""
    return (
        f"<div style='font-size:13px; line-height:1.9;{direction}'>" + "<br>".join(lines) + "</div>"
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
        "overflow:hidden; background:#EEE; margin-top:4px;'>" + "".join(segments) + "</div>"
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
        parts.append(
            f"<span style='white-space:nowrap; margin-right:12px;'>{dot}"
            f"{ctx.t(f'status_{status}')} <strong>{ctx.int(count)}</strong></span>"
        )
    return "".join(parts)


def _spec_badges(cmp: Comparison, ctx: _Ctx) -> str:
    esc = html_lib.escape
    badges = []
    tolerant = [s.source for s in cmp._specs if s.atol is not None or s.rtol is not None]
    if tolerant:
        badges.append(_chip(ctx.t("spec_tolerance", columns=esc(", ".join(tolerant)))))
    normalized = [s.source for s in cmp._specs if s.normalize]
    if normalized:
        badges.append(_chip(ctx.t("spec_normalized", columns=esc(", ".join(normalized)))))
    if not cmp.null_equal:
        badges.append(_chip("null &ne; null"))
    if cmp.schema_mode == "common":
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
    out = []
    esc = html_lib.escape
    case_pairs = dict(diff.case_only)
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
            item += f" &rarr; {html_lib.escape(target_of[col])}"
        detail = ctx.t(
            "matched_rows_differ", n=ctx.int(n), total=ctx.int(n_matched), pct=ctx.pct(pct)
        )
        out.append({"group": group, "item": item, "detail": f"{bar}{detail}"})
    return out


def _changed_rows(cmp: Comparison, limit: int, ctx: _Ctx) -> list[dict[str, str]]:
    if cmp.n_changed == 0:
        return []
    frame = cmp._fetch("changed", limit)
    group = _sample_group(ctx.t("group_changed"), len(frame), cmp.n_changed, ctx)
    key_cols = cmp._key_out_names()
    out = []
    for row in frame.rows(named=True):
        chips = []
        for i, spec in enumerate(cmp._specs):
            if not row.get(f"__pb_d_{i}"):
                continue
            old = row[f"{spec.source}_source"]
            new = row[f"{spec.source}_target"]
            chips.append(
                "<span style='white-space:nowrap; margin-right:14px;'>"
                f"<span style='color:#666;'>{html_lib.escape(spec.source)}:</span> "
                f"<span style='text-decoration:line-through; color:#999;'>{_fmt_value(old)}</span>"
                f" &rarr; <span style='background:#FDF1DC; padding:0 3px; border-radius:2px;'>"
                f"{_fmt_value(new)}</span>{_fmt_delta(old, new)}</span>"
            )
        out.append(
            {"group": group, "item": _key_label(row, key_cols, ctx), "detail": " ".join(chips)}
        )
    return out


def _missing_extra_rows(
    cmp: Comparison, status: str, limit: int, ctx: _Ctx
) -> list[dict[str, str]]:
    count = cmp.n_missing if status == "missing" else cmp.n_extra
    if count == 0:
        return []
    frame = cmp._fetch(status, limit)
    title = ctx.t("group_missing" if status == "missing" else "group_extra")
    group = _sample_group(title, len(frame), count, ctx)
    key_cols = cmp._key_out_names()
    out = []
    for row in frame.rows(named=True):
        if cmp.mode == "multiset":
            # Distinct rows, with how many surplus copies one table has over the other
            item = f"&times;{ctx.int(abs(row['n_source'] - row['n_target']))}"
            exclude = {"n_source", "n_target"}
        else:
            item = _key_label(row, key_cols, ctx)
            exclude = set(key_cols)
        out.append(
            {"group": group, "item": item, "detail": _row_preview(row, exclude=exclude, ctx=ctx)}
        )
    return out


def _dup_rows(cmp: Comparison, limit: int, ctx: _Ctx) -> list[dict[str, str]]:
    if cmp.n_dup_key == 0:
        return []
    frame = cmp._fetch("dup_key", limit)
    group = _sample_group(ctx.t("group_dup"), len(frame), cmp.n_dup_key_values, ctx, noun="keys")
    key_cols = list(cmp.keys)

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

    out = []
    for row in frame.rows(named=True):
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
        out.append({"group": group, "item": _key_label(row, key_cols, ctx), "detail": detail})
    return out


# ----------------------------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------------------------


def _sample_group(title: str, shown: int, total: int, ctx: _Ctx, noun: str = "rows") -> str:
    """A section title with its count, e.g. 'Changed rows (showing 10 of 400 rows)'."""
    if shown >= total:
        count = ctx.t_count(f"n_{noun}", total)
    else:
        count = ctx.t(f"showing_{noun}", shown=ctx.int(shown), total=ctx.int(total))
    return f"{title} ({count})"


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


def _fmt_value(value: Any) -> str:
    if _is_null(value):
        return "<span style='color:#999; font-style:italic;'>NULL</span>"
    if isinstance(value, float):
        text = f"{value:.6g}"
    elif isinstance(value, datetime.datetime):
        text = value.isoformat(sep=" ")
    else:
        text = str(value)
    if len(text) > _MAX_VALUE_CHARS:
        text = text[: _MAX_VALUE_CHARS - 1] + "…"
    return html_lib.escape(text)


def _fmt_delta(old: Any, new: Any) -> str:
    numeric = (int, float)
    if (
        isinstance(old, numeric)
        and isinstance(new, numeric)
        and not isinstance(old, bool)
        and not isinstance(new, bool)
        and not _is_null(old)
        and not _is_null(new)
    ):
        delta = new - old
        return f" <span style='color:#888; font-size:11px;'>({delta:+.6g})</span>"
    if isinstance(old, datetime.datetime) and isinstance(new, datetime.datetime):
        try:
            seconds = (new - old).total_seconds()
        except TypeError:  # pragma: no cover
            return ""
        return f" <span style='color:#888; font-size:11px;'>({seconds:+g}s)</span>"
    return ""


def _key_label(row: dict[str, Any], key_cols: list[str], ctx: _Ctx) -> str:
    if key_cols == ["_row_num_"]:
        return ctx.t("row_n", n=ctx.int(row["_row_num_"]))
    if any(_is_null(row[k]) for k in key_cols):
        null_note = f" <span style='color:#999; font-size:11px;'>{ctx.t('null_key_note')}</span>"
    else:
        null_note = ""
    if len(key_cols) == 1:
        return _fmt_value(row[key_cols[0]]) + null_note
    return ", ".join(f"{html_lib.escape(k)}={_fmt_value(row[k])}" for k in key_cols) + null_note


def _row_preview(row: dict[str, Any], exclude: set[str], ctx: _Ctx) -> str:
    cols = [c for c in row if c not in exclude and not c.startswith("__pb_")]
    shown = cols[:_MAX_PREVIEW_COLS]
    parts = [
        f"<span style='white-space:nowrap; margin-right:10px;'>"
        f"<span style='color:#666;'>{html_lib.escape(c)}:</span> {_fmt_value(row[c])}</span>"
        for c in shown
    ]
    if len(cols) > len(shown):
        parts.append(
            f"<span style='color:#999;'>{ctx.t('n_more', n=ctx.int(len(cols) - len(shown)))}</span>"
        )
    return " ".join(parts)
