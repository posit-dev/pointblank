from __future__ import annotations

import datetime
import html as html_lib
from typing import TYPE_CHECKING, Any

from great_tables import GT, google_font, html, loc, style

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

STATUS_LABELS = {
    "match": "match",
    "changed": "changed",
    "missing": "missing",
    "extra": "extra",
    "dup_key": "duplicate key",
}

_MAX_VALUE_CHARS = 40
_MAX_PREVIEW_COLS = 8


def _comparison_report(cmp: Comparison, limit: int = 10, title: str | None = None) -> GT:
    """Build the tabular (GT) report for a `Comparison`."""
    from pointblank._utils import _is_lib_present

    rows: list[dict[str, str]] = []
    rows += _schema_rows(cmp)
    rows += _column_rows(cmp)
    rows += _changed_rows(cmp, limit)
    rows += _missing_extra_rows(cmp, "missing", limit)
    rows += _missing_extra_rows(cmp, "extra", limit)
    rows += _dup_rows(cmp, limit)

    if not rows:
        rows.append(
            {
                "group": "Result",
                "item": _badge("✓", STATUS_COLORS["match"]),
                "detail": f"All {cmp.n:,} rows match.",
            }
        )

    if _is_lib_present("polars"):
        import polars as pl

        df: Any = pl.DataFrame(rows, schema=["group", "item", "detail"], orient="row")
    else:  # pragma: no cover
        import pandas as pd

        df = pd.DataFrame(rows, columns=["group", "item", "detail"])

    gt_tbl = (
        GT(df, groupname_col="group")
        .tab_header(
            title=html(title or "Table Comparison"),
            subtitle=html(_header_html(cmp)),
        )
        .cols_label(item="", detail="")
        .opt_table_font(font=google_font("IBM Plex Sans"))
        .opt_align_table_header(align="left")
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
    return gt_tbl


# ----------------------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------------------


def _header_html(cmp: Comparison) -> str:
    esc = html_lib.escape
    if cmp.mode == "keyed":
        alignment = "aligned by key " + ", ".join(f"<code>{esc(k)}</code>" for k in cmp.keys)
    elif cmp.mode == "multiset":
        alignment = "compared as multisets of rows (order ignored)"
    else:
        alignment = "aligned by row position"

    tables_line = (
        f"<strong>{esc(cmp.source_name)}</strong> ({cmp.n_source_rows:,} rows) "
        f"&rarr; <strong>{esc(cmp.target_name)}</strong> ({cmp.n_target_rows:,} rows), "
        f"{alignment}"
    )

    if cmp.identical:
        verdict = _badge("IDENTICAL", STATUS_COLORS["match"])
    else:
        pct = (cmp.n_failed / cmp.n * 100) if cmp.n else 0.0
        verdict = _badge("DIFFERENT", STATUS_COLORS["missing"]) + (
            f" {cmp.n_failed:,} of {cmp.n:,} rows fail ({_fmt_pct(pct)})"
        )
        if not cmp.schema_ok:
            verdict += " &middot; " + _badge("SCHEMA DIFFERS", STATUS_COLORS["missing"])
            verdict += (
                " <span style='color:#666;'>(strict schema check: all rows count as failing)</span>"
            )

    lines = [tables_line, verdict, _status_bar(cmp), _legend(cmp)]
    spec = _spec_badges(cmp)
    if spec:
        lines.append(spec)

    return "<div style='font-size:13px; line-height:1.9;'>" + "<br>".join(lines) + "</div>"


def _status_bar(cmp: Comparison) -> str:
    total = cmp.n
    if total == 0:
        return "<span style='color:#666;'>(no rows)</span>"
    segments = []
    for status, count in cmp.status_counts.items():
        if count == 0:
            continue
        width = max(count / total * 100, 0.5)
        segments.append(
            f"<div title='{STATUS_LABELS[status]}: {count:,}' style='width:{width:.3f}%; "
            f"background:{STATUS_COLORS[status]}; height:100%;'></div>"
        )
    return (
        "<div style='display:flex; width:100%; max-width:600px; height:8px; border-radius:2px; "
        "overflow:hidden; background:#EEE; margin-top:4px;'>" + "".join(segments) + "</div>"
    )


def _legend(cmp: Comparison) -> str:
    parts = []
    for status, count in cmp.status_counts.items():
        if count == 0 and status == "dup_key":
            continue
        dot = (
            f"<span style='display:inline-block; width:9px; height:9px; border-radius:2px; "
            f"background:{STATUS_COLORS[status]}; margin-right:4px;'></span>"
        )
        parts.append(
            f"<span style='white-space:nowrap; margin-right:12px;'>{dot}{STATUS_LABELS[status]} "
            f"<strong>{count:,}</strong></span>"
        )
    return "".join(parts)


def _spec_badges(cmp: Comparison) -> str:
    badges = []
    tolerant = [s.source for s in cmp._specs if s.atol is not None or s.rtol is not None]
    if tolerant:
        badges.append(_chip("tolerance: " + ", ".join(tolerant)))
    normalized = [s.source for s in cmp._specs if s.normalize]
    if normalized:
        badges.append(_chip("normalized: " + ", ".join(normalized)))
    if not cmp.null_equal:
        badges.append(_chip("null &ne; null"))
    if cmp.schema_mode == "common":
        badges.append(_chip("schema: common columns"))
    if cmp.dup_keys_mode == "compare":
        badges.append(_chip("duplicate keys compared pairwise"))
    if cmp.partitions > 1:
        badges.append(_chip(f"computed in {cmp.partitions} partitions"))
    n_src_cols = len(cmp._src_columns) - len(cmp.keys)
    if len(cmp.compared_columns) < n_src_cols:
        badges.append(_chip(f"{len(cmp.compared_columns)} of {n_src_cols} columns compared"))
    return " ".join(badges)


# ----------------------------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------------------------


def _schema_rows(cmp: Comparison) -> list[dict[str, str]]:
    diff = cmp.schema_diff
    group = "Schema differences"
    out = []
    esc = html_lib.escape
    case_pairs = dict(diff.case_only)
    for col in diff.only_in_source:
        hint = ""
        if col in case_pairs:
            hint = f" (target has <code>{esc(case_pairs[col])}</code>: differs only by case)"
        out.append(
            {"group": group, "item": esc(col), "detail": _chip("only in source", "#CF142B") + hint}
        )
    for col in diff.only_in_target:
        out.append({"group": group, "item": esc(col), "detail": _chip("only in target", "#3D7CC9")})
    for col, s_dtype, t_dtype in diff.dtype_changed:
        out.append(
            {
                "group": group,
                "item": esc(col),
                "detail": f"type <code>{esc(s_dtype)}</code> &rarr; <code>{esc(t_dtype)}</code>",
            }
        )
    for s_col, t_col in diff.renamed:
        out.append(
            {
                "group": group,
                "item": esc(s_col),
                "detail": f"renamed &rarr; <code>{esc(t_col)}</code> (via <code>column_map=</code>)",
            }
        )
    if diff.reordered:
        out.append(
            {
                "group": group,
                "item": "(column order)",
                "detail": "shared columns are in a different order",
            }
        )
    return out


def _column_rows(cmp: Comparison) -> list[dict[str, str]]:
    n_matched = cmp.n_match + cmp.n_changed
    mismatched = sorted(
        ((col, n) for col, n in cmp.column_mismatches.items() if n > 0),
        key=lambda x: -x[1],
    )
    if not mismatched:
        return []
    group = f"Column mismatches ({len(mismatched)} of {len(cmp.compared_columns)} columns)"
    target_of = {s.source: s.target for s in cmp._specs}
    out = []
    for col, n in mismatched:
        pct = n / n_matched * 100 if n_matched else 0.0
        bar = (
            "<div style='display:inline-block; vertical-align:middle; width:120px; height:7px; "
            "background:#EEE; border-radius:2px; margin-right:8px; overflow:hidden;'>"
            f"<div style='width:{max(pct, 1):.2f}%; height:100%; "
            f"background:{STATUS_COLORS['changed']};'></div></div>"
        )
        item = html_lib.escape(col)
        if target_of[col] != col:
            item += f" &rarr; {html_lib.escape(target_of[col])}"
        out.append(
            {
                "group": group,
                "item": item,
                "detail": f"{bar}{n:,} of {n_matched:,} matched rows differ ({_fmt_pct(pct)})",
            }
        )
    return out


def _changed_rows(cmp: Comparison, limit: int) -> list[dict[str, str]]:
    if cmp.n_changed == 0:
        return []
    frame = cmp._fetch("changed", limit)
    group = _sample_group("Changed rows", len(frame), cmp.n_changed, "changed")
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
        out.append({"group": group, "item": _key_label(row, key_cols), "detail": " ".join(chips)})
    return out


def _missing_extra_rows(cmp: Comparison, status: str, limit: int) -> list[dict[str, str]]:
    count = cmp.n_missing if status == "missing" else cmp.n_extra
    if count == 0:
        return []
    frame = cmp._fetch(status, limit)
    title = "Missing from target" if status == "missing" else "Extra in target"
    group = _sample_group(title, len(frame), count, status)
    key_cols = cmp._key_out_names()
    out = []
    for row in frame.rows(named=True):
        if cmp.mode == "multiset":
            # Distinct rows, with how many surplus copies one table has over the other
            item = f"&times;{abs(row['n_source'] - row['n_target']):,}"
            exclude = {"n_source", "n_target"}
        else:
            item = _key_label(row, key_cols)
            exclude = set(key_cols)
        out.append({"group": group, "item": item, "detail": _row_preview(row, exclude=exclude)})
    return out


def _dup_rows(cmp: Comparison, limit: int) -> list[dict[str, str]]:
    if cmp.n_dup_key == 0:
        return []
    frame = cmp._fetch("dup_key", limit)
    group = _sample_group("Duplicate keys", len(frame), cmp.n_dup_key_values, "dup_key", noun="key")
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
        detail = f"source &times;{row['n_source']:,} &middot; target &times;{row['n_target']:,}"
        if pair_diffs is not None:
            k = tuple(row[c] for c in key_cols)
            if row["n_source"] == 0 or row["n_target"] == 0:
                detail += " &middot; <span style='color:#666;'>no pairs to compare</span>"
            elif pair_diffs.get(k):
                ordered = [s.source for s in cmp._specs if s.source in pair_diffs[k]]
                detail += " &middot; pairs differ in: " + ", ".join(
                    f"<code>{html_lib.escape(c)}</code>" for c in ordered
                )
            else:
                detail += " &middot; all pairs identical"
        out.append({"group": group, "item": _key_label(row, key_cols), "detail": detail})
    return out


# ----------------------------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------------------------


def _sample_group(title: str, shown: int, total: int, status: str, noun: str = "row") -> str:
    nouns = noun if total == 1 else noun + "s"
    count = f"{total:,} {nouns}" if shown >= total else f"showing {shown:,} of {total:,} {nouns}"
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


def _fmt_pct(pct: float) -> str:
    if pct == 0:
        return "0%"
    if pct < 0.1:
        return "<0.1%"
    if pct >= 99.95 and pct < 100:
        return ">99.9%"
    return f"{pct:.1f}%"


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


def _key_label(row: dict[str, Any], key_cols: list[str]) -> str:
    if key_cols == ["_row_num_"]:
        return f"row {row['_row_num_']}"
    if any(_is_null(row[k]) for k in key_cols):
        null_note = " <span style='color:#999; font-size:11px;'>(null key: can't be matched)</span>"
    else:
        null_note = ""
    if len(key_cols) == 1:
        return _fmt_value(row[key_cols[0]]) + null_note
    return ", ".join(f"{html_lib.escape(k)}={_fmt_value(row[k])}" for k in key_cols) + null_note


def _row_preview(row: dict[str, Any], exclude: set[str]) -> str:
    cols = [c for c in row if c not in exclude and not c.startswith("__pb_")]
    shown = cols[:_MAX_PREVIEW_COLS]
    parts = [
        f"<span style='white-space:nowrap; margin-right:10px;'>"
        f"<span style='color:#666;'>{html_lib.escape(c)}:</span> {_fmt_value(row[c])}</span>"
        for c in shown
    ]
    if len(cols) > len(shown):
        parts.append(f"<span style='color:#999;'>+{len(cols) - len(shown)} more</span>")
    return " ".join(parts)
