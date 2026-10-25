"""
Gallery of `col_schema_match()` step reports, for visual review.

Run with `make gallery-schema` (or `python scripts/schema_step_report_gallery.py`). The gallery is
written to `scripts/schema_step_report_gallery.html` (which is ignored by git); use `--open` to open it in a browser and `--only` to limit it to some scenarios or cases
(e.g., `--only reordered,extra-c1-o0`).

Each scenario (a schema checked against the same six-column table) is shown under all four
combinations of `complete=` and `in_order=`. For every case, the gallery describes what the report
should show and why, citing the design rules listed at the top of the page. Those expectations come
from `expected_report()`, an implementation of the design rules that's independent of pointblank's
own matching code. Each report is checked against it, so any disagreement in the result, the
layout of the rows, the marks, or the styling of a row is flagged in the gallery (and makes the
script exit with a non-zero status).
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import re
import sys
import warnings
import webbrowser
from pathlib import Path

import polars as pl

import pointblank as pb
from pointblank.validate import _step_report_schema

warnings.filterwarnings("ignore")

OUTPUT = Path(__file__).with_suffix(".html")

TBL = pl.DataFrame(
    {
        "id": [1, 2],
        "Name": ["a", "b"],
        "score": [1.5, 2.5],
        "date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        "date_time": [datetime.datetime(2024, 1, 1, 1), datetime.datetime(2024, 1, 2, 2)],
        "active": [True, False],
    }
)

TGT = pb.Schema(tbl=TBL).columns
FULL = list(TGT)

COMBOS = [(True, True), (True, False), (False, True), (False, False)]

CHECK = "&check;"
CROSS = "&cross;"

# The design rules for the report; each expectation in the gallery cites the rules behind it
RULES = {
    "R1": (
        "Rows follow the table",
        "There is one row per table column, in table order, and the TABLE side never moves. The "
        "table is what's being validated, so it's the fixed frame of reference that the schema is "
        "laid against.",
    ),
    "R2": (
        "The schema is shown as supplied",
        "Every schema column appears exactly once, with its position in the schema and its dtypes "
        "exactly as supplied (not case-folded or otherwise normalised), so the report can always "
        "be read back against the schema the user wrote.",
    ),
    "R3": (
        "What the marks judge",
        "The mark beside a schema column name judges the column: it's in the table, it isn't a "
        "repeat and (with `in_order=True`) it's in its place. The mark beside the dtype judges the "
        "dtype, and it's only given when the column itself is correct; otherwise it's a dash, "
        "since the row doesn't show that column's own table column to compare against. With no "
        "dtype supplied, both the dtype and its mark are blank. When several dtypes are supplied, "
        "the one that matched is underlined.",
    ),
    "R4": (
        "In order: the schema is laid out in schema order",
        "With `in_order=True`, the matched schema columns fill the rows of the table columns they "
        "matched, in schema order. A column that's out of order then visibly sits beside the wrong "
        "table column, gets a cross, and has its position shown in red. Swapping two columns puts "
        "both out of place, so both are marked. With `complete=True` as well, the schema should "
        "be the table itself, so every schema column (matched or not) is in the row of its own "
        "position: the schema reads from the top row down with no gaps (even for a subset), a "
        "renamed column sits beside the table column it should have been, and any table columns "
        "left over are struck through together at the bottom.",
    ),
    "R5": (
        "Any order: each column sits beside its match",
        "With `in_order=False`, each schema column is shown beside the table column it matched, so "
        "the names always line up. A different order isn't a failure, but the positions of the "
        "columns not in table order are shown in blue with a dot to draw attention to them.",
    ),
    "R6": (
        "Table columns left out of the schema",
        "With `complete=True`, a table column that the schema doesn't name is a failure: its row "
        "gets a cross in place of the schema position, a red fill, and a line struck through the "
        "empty schema cells. With `complete=False` it's allowed, so the row is left white with just "
        "a quiet blue dot in place of the schema position (noting the gap without marking it as a "
        "failure).",
    ),
    "R7": (
        "Schema columns that match nothing",
        "A schema column not found in the table can't be shown beside a table column, so it goes "
        "in an extra row after the table columns (below a separator line), with a grey table side, "
        "a cross, and a dash for its dtype. The exception is with `complete=True` and "
        "`in_order=True`, where every schema column is in the row of its own position (R4), so it "
        "only needs an extra row if the schema is longer than the table.",
    ),
    "R8": (
        "Repeated schema columns always fail",
        "A table can't have two columns of the same name, so a schema column that repeats an "
        "earlier one (after case-folding, with `case_sensitive_colnames=False`) never matches and "
        "is shown like a column not found (R7), whatever the other options.",
    ),
    "R9": (
        "Names match in full",
        "Column names match in full (`date` doesn't match `date_time`). When case is ignored, an "
        "exact-case match is preferred. Dtypes match in full unless `full_match_dtypes=False`, in "
        "which case a supplied dtype only needs to be part of the table's dtype.",
    ),
    "R10": (
        "The marks agree with the result",
        "A passing report shows no crosses and a failing report shows at least one, so the reason "
        "for a failure can always be found in the report.",
    ),
}


def swap(cols, i, j):
    cols = list(cols)
    cols[i], cols[j] = cols[j], cols[i]
    return cols


def replace(cols, name, new):
    return [new if c[0] == name else c for c in cols]


# The scenarios: (key, title, purpose, schema columns, extra `col_schema_match()` options)
SCENARIOS = [
    (
        "exact",
        "Exact match",
        "The baseline: the schema names every column, in order, with the right dtypes. Every "
        "report should be entirely ticks.",
        FULL,
        {},
    ),
    (
        "names_only",
        "Exact match, names only",
        "A schema with no dtypes checks only the names, so the dtype cells should be blank rather "
        "than ticked or dashed.",
        [(c,) for c, _ in TGT],
        {},
    ),
    (
        "reordered",
        "Two columns swapped (`Name` and `score`)",
        "The simplest ordering problem. In order, both swapped columns should sit beside the wrong "
        "table columns; in any order, everything lines up but the swap is pointed out.",
        swap(FULL, 1, 2),
        {},
    ),
    (
        "reversed",
        "All columns reversed",
        "An extreme ordering problem, where every column is out of place in order and every "
        "position is pointed out in any order.",
        FULL[::-1],
        {},
    ),
    (
        "subset_in_order",
        "Subset in order (`id`, `score`, `date_time`)",
        "A partial schema that respects the table's order: fine unless the schema must be "
        "complete. With `complete=False`, the skipped table columns must not be blamed on the "
        "order; with both options, the schema should be the table itself, so it's laid out from "
        "the top and `score` and `date_time` are out of place.",
        [FULL[0], FULL[2], FULL[4]],
        {},
    ),
    (
        "subset_out_of_order",
        "Subset out of order (`score`, `id`)",
        "A partial schema in the wrong order. With `complete=False`, order problems are shown "
        "among the matched columns only (skipping over the table columns the schema leaves "
        "out); with both options, the schema is laid out from the top with no gaps.",
        [FULL[2], FULL[0]],
        {},
    ),
    (
        "missing_middle",
        "All but one column (`score` left out)",
        "One table column left out of an otherwise complete schema; only `complete=True` should "
        "fail, and only because of that one row.",
        [x for x in FULL if x[0] != "score"],
        {},
    ),
    (
        "extra",
        "All columns plus one not in the table (`extra`, last)",
        "A schema column that's not in the table always fails, and it needs a row of its own "
        "since there's no table column to put it beside.",
        FULL + [("extra", "Int64")],
        {},
    ),
    (
        "extra_first",
        "All columns plus one not in the table (`extra`, first)",
        "Like `extra`, but the unmatched column comes first, so it must not make the other "
        "(correctly ordered) columns look out of order. The exception is `complete=True` with "
        "`in_order=True`, where the schema should be the table itself: there the extra column "
        "pushes every other column down a position, so they are all out of place.",
        [("extra", "Int64")] + FULL,
        {},
    ),
    (
        "extra_reordered",
        "Two columns swapped plus one not in the table",
        "Two kinds of problem in one schema; both should be visible together.",
        swap(FULL, 1, 2) + [("extra", "Int64")],
        {},
    ),
    (
        "renamed",
        "One column renamed (`Name` as `full_name`)",
        "A rename looks like a column not found plus a table column left out; the report should "
        "show both halves.",
        replace(FULL, "Name", ("full_name", "String")),
        {},
    ),
    (
        "no_overlap",
        "No columns in common (`x`, `y`)",
        "Nothing matches, so nothing should be ticked and every table column is either struck "
        "through (complete) or dotted.",
        [("x", "Int64"), ("y", "String")],
        {},
    ),
    (
        "dtype_wrong",
        "One wrong dtype (`score` as Int64)",
        "A dtype problem only, so the only cross should be on the `score` dtype.",
        replace(FULL, "score", ("score", "Int64")),
        {},
    ),
    (
        "dtype_wrong_reordered",
        "Wrong dtype and two columns swapped",
        "In order, the wrong dtype is on a misplaced column and so goes unscored (a dash); in any "
        "order, it's the one cross.",
        swap(replace(FULL, "score", ("score", "Int64")), 1, 2),
        {},
    ),
    (
        "multi_dtype_ok",
        "Several dtypes, one matching (`id` as Int32 or Int64)",
        "Alternative dtypes pass when any one matches, and the one that matched is underlined.",
        replace(FULL, "id", ("id", ["Int32", "Int64"])),
        {},
    ),
    (
        "multi_dtype_bad",
        "Several dtypes, none matching (`id` as Int32 or Float64)",
        "Alternative dtypes fail when none match, with nothing underlined.",
        replace(FULL, "id", ("id", ["Int32", "Float64"])),
        {},
    ),
    (
        "prefix_names",
        "A name that's a prefix of another (`date` only)",
        "`date` must match the `date` column and not `date_time`.",
        [("date", "Date")],
        {},
    ),
    (
        "prefix_names_2",
        "Prefix-sharing names reversed (`date_time`, `date`)",
        "Both names match in full, and in the wrong order.",
        [FULL[4], FULL[3]],
        {},
    ),
    (
        "duplicate",
        "Column repeated (`id` twice, then the rest)",
        "A repeated column can never be satisfied, so every combination of options fails because "
        "of the repeat. With `complete=True` and `in_order=True` the repeat also pushes the "
        "columns after it down a position, so they are out of place too.",
        [FULL[0], FULL[0]] + FULL[1:],
        {},
    ),
    (
        "duplicate_subset",
        "Column repeated in a subset (`id`, `id`, `score`)",
        "A repeat fails even when the schema doesn't need to be complete or in order.",
        [FULL[0], FULL[0], FULL[2]],
        {},
    ),
    (
        "duplicate_names_only",
        "Column repeated, names only (`id`, `Name`, `id`)",
        "A repeat with no dtypes, placed after another column.",
        [("id",), ("Name",), ("id",)],
        {},
    ),
    (
        "case_colname",
        "Name case differs (`name` for `Name`), case-sensitive",
        "With case-sensitive names, `name` is simply not in the table.",
        replace(FULL, "Name", ("name", "String")),
        {},
    ),
    (
        "case_colname_ci",
        "Name case differs, `case_sensitive_colnames=False`",
        "Ignoring case, this is an exact match; the report shows `name` as supplied.",
        replace(FULL, "Name", ("name", "String")),
        {"case_sensitive_colnames": False},
    ),
    (
        "case_colname_ci_subset",
        "Case-insensitive subset (`NAME`, `SCORE`)",
        "A partial schema in another case, in order.",
        [("NAME", "String"), ("SCORE", "Float64")],
        {"case_sensitive_colnames": False},
    ),
    (
        "case_colname_ci_reordered",
        "Case-insensitive, swapped (`ID`, `SCORE`, `NAME`)",
        "Ordering problems are judged the same way when case is ignored.",
        [("ID", "Int64"), ("SCORE", "Float64"), ("NAME", "String")],
        {"case_sensitive_colnames": False},
    ),
    (
        "case_colname_ci_duplicate",
        "Case-insensitive repeat (`id`, `ID`)",
        "When case is ignored, `ID` repeats `id` and so always fails.",
        [("id", "Int64"), ("ID", "Int64")],
        {"case_sensitive_colnames": False},
    ),
    (
        "case_dtype",
        "Dtype case differs (`int64`), case-sensitive",
        "With case-sensitive dtypes, `int64` doesn't match `Int64`.",
        replace(FULL, "id", ("id", "int64")),
        {},
    ),
    (
        "case_dtype_ci",
        "Dtype case differs, `case_sensitive_dtypes=False`",
        "Ignoring case, the dtype matches, and it's still shown as supplied (`int64`).",
        replace(FULL, "id", ("id", "int64")),
        {"case_sensitive_dtypes": False},
    ),
    (
        "partial_dtype",
        "Partial dtypes (`Int`, `Float`, `Datetime`), full match",
        "Partial dtypes don't match when dtypes must match in full.",
        [("id", "Int"), ("score", "Float"), ("date_time", "Datetime")],
        {},
    ),
    (
        "partial_dtype_ok",
        "Partial dtypes, `full_match_dtypes=False`",
        "Partial dtypes match when they're part of the table's dtype.",
        [("id", "Int"), ("score", "Float"), ("date_time", "Datetime")],
        {"full_match_dtypes": False},
    ),
    (
        "all_relaxed",
        "Everything relaxed (case-insensitive names and dtypes, partial dtypes)",
        "All the relaxing options together, with the columns also out of order.",
        [("SCORE", "float"), ("ID", "int")],
        {
            "case_sensitive_colnames": False,
            "case_sensitive_dtypes": False,
            "full_match_dtypes": False,
        },
    ),
]


def expected_report(
    schema,
    complete,
    in_order,
    case_sensitive_colnames=True,
    case_sensitive_dtypes=True,
    full_match_dtypes=True,
):
    """
    The result and the rows that the design rules call for, worked out independently of
    pointblank's own matching code.
    """

    def norm(name):
        return name if case_sensitive_colnames else name.lower()

    tgt_names = [c for c, _ in TGT]

    # Match each schema column to a table column (R8, R9)
    cols, seen = [], set()
    for i, spec in enumerate(schema):
        name = spec[0]
        dtypes = None if len(spec) == 1 else ([spec[1]] if isinstance(spec[1], str) else spec[1])
        col = {"pos": i + 1, "name": name, "dtypes": dtypes, "target": None, "dup": False}

        if norm(name) in seen:
            col["dup"] = True
        else:
            seen.add(norm(name))
            if name in tgt_names:
                col["target"] = tgt_names.index(name)
            elif not case_sensitive_colnames:
                hits = [j for j, t in enumerate(tgt_names) if t.lower() == name.lower()]
                col["target"] = hits[0] if hits else None

        if col["target"] is not None and dtypes is not None:
            actual = TGT[col["target"]][1]
            ok = False
            for d in dtypes:
                a, d = (actual, d) if case_sensitive_dtypes else (actual.lower(), d.lower())
                ok = ok or (d == a if full_match_dtypes else d in a)
            col["dtype_ok"] = ok

        cols.append(col)

    matched = [c for c in cols if c["target"] is not None]
    unmatched = [c for c in cols if c["target"] is None]

    # When the schema should be complete and in order, it should be the table itself, so each
    # schema column belongs in the row of its own position (R4)
    packed = complete and in_order

    # A matched column is out of place if it isn't where the table order puts it (R4, R5)
    if packed:
        for c in matched:
            c["out_of_place"] = c["target"] != c["pos"] - 1
    else:
        slots = sorted(c["target"] for c in matched)
        for c, slot in zip(matched, slots):
            c["out_of_place"] = c["target"] != slot

    passed = (
        not unmatched
        and all(c.get("dtype_ok", True) for c in matched)
        and (not complete or len(matched) == len(TGT))
        and (not in_order or not any(c["out_of_place"] for c in matched))
    )

    # The pairs of table column and schema column (either may be absent) for each row (R1, R4,
    # R5, R7)
    if packed:
        layout = [
            (k if k < len(TGT) else None, cols[k] if k < len(cols) else None)
            for k in range(max(len(TGT), len(cols)))
        ]
    else:
        if in_order:
            beside = dict(zip(slots, matched))
        else:
            beside = {c["target"]: c for c in matched}
        layout = [(j, beside.get(j)) for j in range(len(TGT))] + [(None, c) for c in unmatched]

    def schema_side(c, name_ok):
        if c["dtypes"] is None:
            dtype_mark = ""
        elif not name_ok:
            dtype_mark = "dash"
        else:
            dtype_mark = "check" if c["dtype_ok"] else "cross"
        return {
            "pos": c["pos"],
            "name": c["name"],
            "name_mark": "check" if name_ok else "cross",
            "dtype_mark": dtype_mark,
            "underline": dtype_mark == "check" and len(c["dtypes"]) > 1,
            "not_found": c["target"] is None,
            "dup": c["dup"],
        }

    rows = []
    for j, c in layout:
        if j is None:
            row = {"target": None, "target_name": ""}
        else:
            row = {"target": j + 1, "target_name": TGT[j][0]}

        if c is None:
            row.update(
                {
                    "pos": None,
                    "name": "",
                    "name_mark": "",
                    "dtype_mark": "",
                    "underline": False,
                    "not_found": False,
                    "dup": False,
                }
            )
            row["style"] = "struck" if complete else "skipped"
        else:
            found = c["target"] is not None
            out_of_place = found and c["out_of_place"]
            row.update(schema_side(c, name_ok=found and not (in_order and out_of_place)))
            if j is None:
                row["style"] = "unmatched"
            elif in_order and out_of_place:
                row["style"] = "red_pos"
            elif not in_order and out_of_place:
                row["style"] = "blue_dot"
            else:
                row["style"] = None

        rows.append(row)

    return passed, cols, rows


def describe(passed, cols, rows, complete, in_order, opts):
    """The expectations for a case, as (text, rules) pairs."""

    def names(rs, key="name"):
        return ", ".join(f"`{r[key]}`" for r in rs)

    def positions(rs):
        return ", ".join(str(r["pos"]) for r in rs)

    out = []

    if passed:
        out.append(("Passes, so the report shows no crosses.", ["R10"]))
    else:
        out.append(("Fails, so at least one cross explains why.", ["R10"]))

    red = [r for r in rows if r["style"] == "red_pos"]
    if red:
        out.append(
            (
                f"Out of place, in schema order: {names(red)} (at position(s) {positions(red)}) "
                f"sit beside table column(s) {names(red, 'target_name')}, each with a cross, a red "
                "position, and a dash for any dtype.",
                ["R4", "R3"],
            )
        )

    blue = [r for r in rows if r["style"] == "blue_dot"]
    if blue:
        out.append(
            (
                f"Order differs (allowed): the positions of {names(blue)} ({positions(blue)}) are "
                "blue with a dot, and they still get ticks.",
                ["R5"],
            )
        )

    struck = [r for r in rows if r["style"] == "struck"]
    if struck and in_order:
        # The schema is laid out from the top, so the struck rows are where it runs out, which
        # needn't be the rows of the table columns it leaves out
        named = {c["target"] for c in cols if c["target"] is not None}
        left_out = [{"name": t} for j, (t, _) in enumerate(TGT) if j not in named]
        out.append(
            (
                f"The schema has only {len(cols)} column(s), so rows "
                f"{', '.join(str(r['target']) for r in struck)} are struck through together at the "
                "bottom (a cross in the position slot, a red fill and a line); the table columns "
                f"it leaves out are {names(left_out)}.",
                ["R4", "R6"],
            )
        )
    elif struck:
        out.append(
            (
                f"Missing from the complete schema: {names(struck, 'target_name')}, each with a "
                "cross in the position slot, a red fill and a struck-through line.",
                ["R6"],
            )
        )

    skipped = [r for r in rows if r["style"] == "skipped"]
    if skipped:
        out.append(
            (
                f"Left out of the schema (allowed): {names(skipped, 'target_name')} have a blue "
                "dot in place of the schema position, a white (unfilled) schema side and no marks.",
                ["R6"],
            )
        )

    def placement(rs):
        inline = [r for r in rs if r["target"] is not None]
        extra = [r for r in rs if r["target"] is None]
        parts = []
        if inline:
            parts.append(
                f"in the row(s) of their own position, beside {names(inline, 'target_name')}"
            )
        if extra:
            parts.append(
                f"{'with ' + names(extra) + ' ' if inline else ''}in extra rows below "
                "the table columns"
            )
        return " and ".join(parts)

    not_found = [r for r in rows if r["not_found"] and not r["dup"]]
    if not_found:
        out.append(
            (
                f"Not in the table: {names(not_found)} (position(s) {positions(not_found)}), "
                f"{placement(not_found)}, each with a cross and a dash for any dtype.",
                ["R7", "R4", "R9"] if complete and in_order else ["R7", "R9"],
            )
        )

    dups = [r for r in rows if r["dup"]]
    if dups:
        out.append(
            (
                f"Repeated: {names(dups)} at position(s) {positions(dups)} never matches and is "
                f"shown {placement(dups)}, with a cross.",
                ["R8", "R4", "R7"] if complete and in_order else ["R8", "R7"],
            )
        )

    bad_dtypes = [r for r in rows if r["dtype_mark"] == "cross"]
    if bad_dtypes:
        out.append(
            (f"Wrong dtype: {names(bad_dtypes)}, with a cross beside the dtype.", ["R3", "R9"])
        )

    # A wrong dtype on a column that's out of place or unmatched isn't scored in the report
    unscored = [
        c
        for c in cols
        if c.get("dtype_ok") is False
        and any(r["pos"] == c["pos"] and r["dtype_mark"] == "dash" for r in rows)
    ]
    if unscored:
        out.append(
            (
                f"The wrong dtype of {names(unscored)} isn't scored (it's a dash) since the column "
                "is out of place; it only shows as a dtype mismatch in the validation report note.",
                ["R3"],
            )
        )

    if any(c["dtypes"] is None for c in cols):
        out.append(("Columns given without a dtype have a blank dtype and mark.", ["R3"]))

    if any(c["dtypes"] and len(c["dtypes"]) > 1 for c in cols):
        under = [r for r in rows if r["underline"]]
        if under:
            out.append(
                (f"The dtype that matched is underlined for {names(under)}.", ["R3"]),
            )
        else:
            out.append(("No dtype is underlined, since none matched.", ["R3"]))

    ignored = [
        k for k in ("case_sensitive_colnames", "case_sensitive_dtypes") if not opts.get(k, True)
    ]
    if ignored:
        out.append(
            (
                "Names and dtypes are shown as supplied, in their own case, even though "
                + " and ".join(f"`{k}=False`" for k in ignored)
                + ".",
                ["R2", "R9"],
            )
        )

    if not opts.get("full_match_dtypes", True):
        out.append(
            (
                "Partial dtypes are shown as supplied, and match when they're part of the "
                "table's dtype (`full_match_dtypes=False`).",
                ["R2", "R9"],
            )
        )

    ticked = [r for r in rows if r["name_mark"] == "check" and r["style"] is None]
    if ticked and not passed:
        out.append((f"Everything else is ticked ({names(ticked)}).", ["R3"]))

    return out


def body_rows(report_html):
    """The `<tr>` elements of the body of a rendered report."""
    body = report_html.split("<tbody", 1)[1].split("</tbody>", 1)[0]
    return re.findall(r"<tr>.*?</tr>", body, flags=re.S)


def index_exp_cell(row_html):
    return re.search(
        r'<td style="([^"]*)" headers="[^"]*index_exp"[^>]*>(.*?)</td>', row_html, re.S
    )


def check_report(passed, rows, actual_passed, df, report_html):
    """Differences between the expected report and the actual one."""

    problems = []

    if actual_passed != passed:
        expected, actual = ("PASS" if passed else "FAIL"), ("PASS" if actual_passed else "FAIL")
        problems.append(f"result is {actual}, expected {expected}")

    actual_rows = df.to_dicts()
    if len(actual_rows) != len(rows):
        return problems + [f"{len(actual_rows)} rows, expected {len(rows)}"]

    def mark(x):
        if CHECK in x:
            return "check"
        if CROSS in x:
            return "cross"
        if x == "&mdash;":
            return "dash"
        return x

    html_rows = body_rows(report_html)

    for k, (exp, act, row_html) in enumerate(zip(rows, actual_rows, html_rows), start=1):
        where = f"row {k}"
        got = {
            "target": act["index_target"],
            "target_name": act["col_name_target"],
            "pos": act["index_exp"],
            "name": act["col_name_exp"],
            "name_mark": mark(act["col_name_exp_correct"]),
            "dtype_mark": mark(act["dtype_exp_correct"]),
            "underline": "text-decoration: underline" in act["dtype_exp"],
        }
        for key, value in got.items():
            if value != exp[key]:
                problems.append(f"{where}: {key} is {value!r}, expected {exp[key]!r}")

        index_style, index_content = index_exp_cell(row_html).groups()
        styles = {
            "struck": "linear-gradient" in row_html and CROSS in index_content,
            "blue_dot": "border-radius: 50%" in index_content and "#336699" in index_style,
            "red_pos": "color: #CF142B;font-weight: bold" in index_style,
            "skipped": "border-radius: 50%" in index_content
            and not re.search(r"\d", re.sub(r"<[^>]*>", "", index_content))
            and "#F3F3F3" not in index_style,
            "unmatched": 'index_target" class' in row_html
            and "#F3F3F3" in row_html.split("index_target", 1)[0][-300:],
        }
        for style, shown in styles.items():
            if shown != (exp["style"] == style):
                problems.append(
                    f"{where}: {style} styling is {'shown' if shown else 'missing'}"
                    f" (expected {'it' if exp['style'] == style else 'none'})"
                )

    return problems


def md(text):
    """Escape text for HTML, showing `backticked` spans as code."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text))


def fmt_schema(cols):
    return html.escape(json.dumps([list(x) if not isinstance(x, str) else x for x in cols]))


def badge(text, bg, fg="#fff"):
    return f"<span class='badge' style='background:{bg};color:{fg}'>{text}</span>"


def rule_links(rules):
    return " ".join(
        f"<a class='rule' href='#{r}' title='{html.escape(RULES[r][0])}'>{r}</a>" for r in rules
    )


def build(only=None):
    sections, flagged, n_cases = [], [], 0

    for s, (key, title, purpose, schema, opts) in enumerate(SCENARIOS, start=1):
        cards = []

        for c, (complete, in_order) in enumerate(COMBOS, start=1):
            case_id = f"{key}-c{int(complete)}-o{int(in_order)}"
            if only and key not in only and case_id not in only:
                continue

            n_cases += 1
            args = dict(complete=complete, in_order=in_order, **opts)
            opt_txt = ", ".join(f"{k}={v}" for k, v in args.items())
            passed, cols, rows = expected_report(schema, **args)

            try:
                # Copy the column specs so that any mutation of the schema can't leak
                columns = [
                    tuple(list(x) if isinstance(x, list) else x for x in col) for col in schema
                ]
                validation = (
                    pb.Validate(TBL)
                    .col_schema_match(pb.Schema(columns=columns), **args)
                    .interrogate()
                )
                actual_passed = bool(validation.validation_info[0].all_passed)
                info = validation.validation_info[0].val_info
                report = validation.get_step_report(i=1).as_raw_html()
                df = _step_report_schema(1, info, None, "en", debug_return_df=True)
                plain = _step_report_schema(1, info, None, "en").as_raw_html()
                problems = check_report(passed, rows, actual_passed, df, plain)
                debug = html.escape(json.dumps(info, indent=1, default=str))
            except Exception as e:  # noqa: BLE001
                actual_passed, report, debug = None, "", ""
                problems = [f"error: {e!r}"]

            for p in problems:
                flagged.append((case_id, f"{s}.{c}", title, opt_txt, p))

            badges = badge("PASS" if passed else "FAIL", "#4CA64C" if passed else "#B22222")
            if problems:
                badges += badge(f"{len(problems)} problem(s)", "#7B2CBF")
            else:
                badges += badge("report as expected", "#E8E8E8", "#555")

            expect_items = "".join(
                f"<li>{md(text)} {rule_links(rules)}</li>"
                for text, rules in describe(passed, cols, rows, complete, in_order, opts)
            )
            problem_items = "".join(f"<li>{html.escape(p)}</li>" for p in problems)

            cards.append(
                f"<div class='card' id='{case_id}'>"
                f"<div class='name'><a href='#{case_id}'>{s}.{c}</a> {md(title)} "
                f"&mdash; <code>complete={complete}, in_order={in_order}</code> {badges}</div>"
                f"<div class='id'>{case_id} &middot; <code>{html.escape(opt_txt)}</code></div>"
                f"<div class='expect'><b>Expect</b><ul>{expect_items}</ul></div>"
                + (
                    f"<div class='problems'><b>Problems</b><ul>{problem_items}</ul></div>"
                    if problems
                    else ""
                )
                + f"{report}<details><summary>schema validation info</summary>"
                f"<pre class='dbg'>{debug}</pre></details></div>"
            )

        if cards:
            sections.append(
                f"<h2 class='sec' id='{key}'>{s}. {md(title)}</h2>"
                f"<p class='look'><b>Purpose:</b> {md(purpose)}<br>"
                f"<b>Schema:</b> <code>{fmt_schema(schema)}</code>"
                + (f"<br><b>Options:</b> <code>{html.escape(str(opts))}</code>" if opts else "")
                + "</p>"
                + "".join(cards)
            )

    tgt_rows = "".join(
        f"<tr><td>{i}</td><td><code>{c}</code></td><td><code>{d}</code></td></tr>"
        for i, (c, d) in enumerate(TGT, 1)
    )
    rules = "".join(
        f"<dt id='{r}'>{r}. {md(t)}</dt><dd>{md(d)}</dd>" for r, (t, d) in RULES.items()
    )
    flagged_items = (
        "".join(
            f"<li><a href='#{cid}'>{num} {md(t)}</a> &mdash; "
            f"<code>{html.escape(o)}</code>: {html.escape(p)}</li>"
            for cid, num, t, o, p in flagged
        )
        or "<li>None</li>"
    )
    toc = "".join(f"<li><a href='#{key}'>{md(title)}</a></li>" for key, title, *_ in SCENARIOS)

    page = (
        "<html><head><meta charset='utf-8'><title>col_schema_match() step report gallery</title>"
        "<style>"
        "body{font-family:sans-serif;width:1100px;margin:20px;background:#fafafa;color:#222}"
        ".sec{border-bottom:2px solid #333;margin-top:48px}"
        ".look{font-size:14px;color:#333}"
        ".card{background:#fff;border:1px solid #ddd;padding:12px;margin:16px 0}"
        ".name{font-weight:600;font-size:15px;margin-bottom:2px}"
        ".name a{color:#336699;text-decoration:none}"
        ".id{font-size:12px;color:#888;margin-bottom:8px}"
        ".expect,.problems{font-size:13px;margin:0 0 10px}"
        ".expect ul,.problems ul{margin:4px 0;padding-left:20px}"
        ".problems{color:#7B2CBF}"
        ".rule{font:600 10px monospace;color:#336699;border:1px solid #6699CC;border-radius:3px;"
        "padding:0 3px;text-decoration:none}"
        ".badge{font:600 11px sans-serif;padding:2px 6px;border-radius:3px;margin-left:6px}"
        ".dbg{font-size:11px;max-height:300px;overflow:auto;background:#f6f6f6}"
        "details summary{font-size:12px;color:#888;cursor:pointer;margin-top:6px}"
        ".tgt td{padding:2px 10px;font-size:13px}"
        "dt{font-weight:600;margin-top:8px}dd{margin-left:20px;font-size:14px}"
        "</style></head><body>"
        "<h1>col_schema_match() step report gallery</h1>"
        f"<p>{n_cases} case(s): each scenario is checked against the table below under the four "
        "combinations of <code>complete</code> and <code>in_order</code>. Each case lists what the "
        "report should show and the design rules behind it. Those expectations come from an "
        "implementation of the rules that's independent of pointblank's matching code, and each "
        "report is checked against them (the result, every row, every mark, and the styling of "
        "each row); any differences are listed below and on the case itself.</p>"
        f"<h3>Target table</h3><table class='tgt'>{tgt_rows}</table>"
        f"<h3>Problems ({len(flagged)})</h3><ul>{flagged_items}</ul>"
        f"<h3>Design rules</h3><dl>{rules}</dl>"
        f"<h3>Scenarios</h3><ol>{toc}</ol>" + "".join(sections) + "</body></html>"
    )

    return page, n_cases, flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT, help="where to write the gallery")
    parser.add_argument("--open", action="store_true", help="open the gallery in a browser")
    parser.add_argument(
        "--only", help="comma-separated scenario keys or case IDs (e.g., reordered,extra-c1-o0)"
    )
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    page, n_cases, flagged = build(only=only)

    args.output.write_text(page, encoding="utf-8")
    print(f"Wrote {n_cases} cases to {args.output} ({len(flagged)} problem(s))")
    for cid, num, _, _, problem in flagged:
        print(f"  {num} {cid}: {problem}")

    if args.open:
        webbrowser.open(args.output.resolve().as_uri())

    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
