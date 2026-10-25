"""
Tests of `col_schema_match()` results and step reports across many schemas and combinations of
options. The result of each case is checked against `reference_passed()`, an independent
implementation of the documented matching rules, and each step report is checked for being
consistent with the result and for accounting for every column of the table and the schema.
"""

import datetime

import great_tables as GT
import polars as pl
import pytest

import pointblank as pb
from pointblank.schema import _match_schema_columns, _schema_columns_out_of_order
from pointblank.validate import _create_col_schema_match_note_html, _step_report_schema

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

CHECK = "&check;"
CROSS = "&cross;"


def swap(cols, i, j):
    cols = list(cols)
    cols[i], cols[j] = cols[j], cols[i]
    return cols


def replace(cols, name, new):
    return [new if c[0] == name else c for c in cols]


# Each scenario is a schema (as a list of column specs) and any extra `col_schema_match()` options
SCENARIOS = {
    "exact": (FULL, {}),
    "names_only": ([(c,) for c, _ in TGT], {}),
    "reordered": (swap(FULL, 1, 2), {}),
    "reversed": (FULL[::-1], {}),
    "subset_in_order": ([FULL[0], FULL[2], FULL[4]], {}),
    "subset_out_of_order": ([FULL[2], FULL[0]], {}),
    "missing_middle": ([x for x in FULL if x[0] != "score"], {}),
    "extra": (FULL + [("extra", "Int64")], {}),
    "extra_first": ([("extra", "Int64")] + FULL, {}),
    "extra_reordered": (swap(FULL, 1, 2) + [("extra", "Int64")], {}),
    "renamed": (replace(FULL, "Name", ("full_name", "String")), {}),
    "no_overlap": ([("x", "Int64"), ("y", "String")], {}),
    "dtype_wrong": (replace(FULL, "score", ("score", "Int64")), {}),
    "dtype_wrong_reordered": (swap(replace(FULL, "score", ("score", "Int64")), 1, 2), {}),
    "multi_dtype_ok": (replace(FULL, "id", ("id", ["Int32", "Int64"])), {}),
    "multi_dtype_bad": (replace(FULL, "id", ("id", ["Int32", "Float64"])), {}),
    "prefix_names": ([("date", "Date")], {}),
    "prefix_names_2": ([FULL[4], FULL[3]], {}),
    "duplicate": ([FULL[0], FULL[0]] + FULL[1:], {}),
    "duplicate_subset": ([FULL[0], FULL[0], FULL[2]], {}),
    "duplicate_names_only": ([("id",), ("Name",), ("id",)], {}),
    "case_colname": (replace(FULL, "Name", ("name", "String")), {}),
    "case_colname_ci": (
        replace(FULL, "Name", ("name", "String")),
        {"case_sensitive_colnames": False},
    ),
    "case_colname_ci_subset": (
        [("NAME", "String"), ("SCORE", "Float64")],
        {"case_sensitive_colnames": False},
    ),
    "case_colname_ci_reordered": (
        [("ID", "Int64"), ("SCORE", "Float64"), ("NAME", "String")],
        {"case_sensitive_colnames": False},
    ),
    "case_colname_ci_duplicate": (
        [("id", "Int64"), ("ID", "Int64")],
        {"case_sensitive_colnames": False},
    ),
    "case_dtype": (replace(FULL, "id", ("id", "int64")), {}),
    "case_dtype_ci": (replace(FULL, "id", ("id", "int64")), {"case_sensitive_dtypes": False}),
    "partial_dtype": ([("id", "Int"), ("score", "Float"), ("date_time", "Datetime")], {}),
    "partial_dtype_ok": (
        [("id", "Int"), ("score", "Float"), ("date_time", "Datetime")],
        {"full_match_dtypes": False},
    ),
    "all_relaxed": (
        [("SCORE", "float"), ("ID", "int")],
        {
            "case_sensitive_colnames": False,
            "case_sensitive_dtypes": False,
            "full_match_dtypes": False,
        },
    ),
}

COMBOS = [(True, True), (True, False), (False, True), (False, False)]

CASES = [
    pytest.param(key, complete, in_order, id=f"{key}-complete_{complete}-in_order_{in_order}")
    for key in SCENARIOS
    for complete, in_order in COMBOS
]


def reference_passed(
    expected,
    complete,
    in_order,
    case_sensitive_colnames=True,
    case_sensitive_dtypes=True,
    full_match_dtypes=True,
):
    """Independent implementation of the documented `col_schema_match()` rules."""

    def norm(name):
        return name if case_sensitive_colnames else name.lower()

    exp_names = [norm(x[0]) for x in expected]
    tgt_names = [norm(c) for c, _ in TGT]
    tgt_dtypes = dict(zip(tgt_names, [d for _, d in TGT]))

    # A schema naming the same column twice can never be satisfied by a table
    if len(set(exp_names)) != len(exp_names):
        return False

    # Every schema column must be in the table, with a matching dtype (if one is given)
    for x, name in zip(expected, exp_names):
        if name not in tgt_dtypes:
            return False
        if len(x) > 1:
            options = [x[1]] if isinstance(x[1], str) else list(x[1])
            actual = tgt_dtypes[name]
            if not case_sensitive_dtypes:
                options, actual = [o.lower() for o in options], actual.lower()
            if not any((o == actual) if full_match_dtypes else (o in actual) for o in options):
                return False

    # A complete schema must name every column of the table
    if complete and set(exp_names) != set(tgt_names):
        return False

    # The schema columns must appear in the same relative order as in the table
    if in_order and exp_names != [n for n in tgt_names if n in exp_names]:
        return False

    return True


def report_df(validation):
    return _step_report_schema(
        step=1,
        schema_info=validation.validation_info[0].val_info,
        header=None,
        lang="en",
        debug_return_df=True,
    )


def run_case(key, complete, in_order):
    expected, opts = SCENARIOS[key]

    # Copy the column specs so that any mutation of the schema can't leak between cases
    schema = pb.Schema(
        columns=[tuple(list(x) if isinstance(x, list) else x for x in col) for col in expected]
    )
    validation = (
        pb.Validate(TBL)
        .col_schema_match(schema, complete=complete, in_order=in_order, **opts)
        .interrogate()
    )
    return expected, opts, validation


@pytest.mark.parametrize("key, complete, in_order", CASES)
def test_col_schema_match_result(key, complete, in_order):
    expected, opts, validation = run_case(key, complete, in_order)

    assert validation.validation_info[0].all_passed == reference_passed(
        expected, complete=complete, in_order=in_order, **opts
    )


@pytest.mark.parametrize("key, complete, in_order", CASES)
def test_col_schema_match_step_report(key, complete, in_order):
    expected, opts, validation = run_case(key, complete, in_order)

    passed = validation.validation_info[0].all_passed
    rows = report_df(validation).to_dicts()

    n_tgt = len(TGT)

    # Table columns missing from a complete schema get a cross in place of the schema position
    # (added when the report is rendered)
    n_missing = sum(1 for r in rows[:n_tgt] if r["index_exp"] is None) if complete else 0

    marks = " ".join(r["col_name_exp_correct"] + " " + r["dtype_exp_correct"] for r in rows)
    marks += f" {CROSS}" * n_missing

    # The report is consistent with the result: a passing report has no failure marks, and a
    # failing report shows at least one
    if passed:
        assert CROSS not in marks
    else:
        assert CROSS in marks

    # The table columns are all shown once, in table order, and come first
    assert [r["index_target"] for r in rows[:n_tgt]] == list(range(1, n_tgt + 1))
    assert [r["col_name_target"] for r in rows[:n_tgt]] == [c for c, _ in TGT]
    assert all(r["index_target"] is None for r in rows[n_tgt:])

    # The schema columns are all shown once, with their positions in the schema
    shown = sorted((r["index_exp"], r["col_name_exp"]) for r in rows if r["index_exp"] is not None)
    assert shown == [(i + 1, x[0]) for i, x in enumerate(expected)]

    def same_name(r):
        if opts.get("case_sensitive_colnames", True):
            return r["col_name_exp"] == r["col_name_target"]
        return r["col_name_exp"].lower() == r["col_name_target"].lower()

    shown_rows = [r for r in rows[:n_tgt] if r["col_name_exp"]]

    if in_order:
        # The matched schema columns are shown in schema order, and one is marked as correct
        # exactly when it sits alongside the table column of the same name
        positions = [r["index_exp"] for r in shown_rows]
        assert positions == sorted(positions)
        for r in shown_rows:
            assert (CHECK in r["col_name_exp_correct"]) == same_name(r)
    else:
        # A schema column shown alongside a table column has a matching name
        for r in shown_rows:
            assert same_name(r)

    # Schema columns that matched no table column (the rows after the table columns) are marked
    # as incorrect
    for r in rows[n_tgt:]:
        assert CROSS in r["col_name_exp_correct"]

    # A dtype is scored only when its column is correct, and is otherwise marked with a dash
    for r in rows:
        if r["index_exp"] is None:
            continue
        spec = expected[r["index_exp"] - 1]
        if len(spec) == 1:
            assert r["dtype_exp_correct"] == ""
        elif CROSS in r["col_name_exp_correct"]:
            assert r["dtype_exp_correct"] == "&mdash;"
        else:
            assert CHECK in r["dtype_exp_correct"] or CROSS in r["dtype_exp_correct"]

    # Table columns left out of the schema have no marks in the schema columns
    for r in rows[:n_tgt]:
        if not r["col_name_exp"]:
            assert r["col_name_exp_correct"] == r["dtype_exp_correct"] == ""

    # The rendered report has a cross (and a struck-through row) for each table column missing
    # from a complete schema
    report_html = _step_report_schema(
        step=1,
        schema_info=validation.validation_info[0].val_info,
        header=None,
        lang="en",
    ).as_raw_html()
    assert report_html.count("linear-gradient") == 4 * n_missing

    # Without `in_order=True`, a matched schema column is never marked as incorrect
    if not in_order:
        for r in rows[:n_tgt]:
            if r["col_name_exp"]:
                assert CHECK in r["col_name_exp_correct"]

    # The dtypes are shown as they were supplied
    for r in rows:
        if r["index_exp"] is not None:
            spec = expected[r["index_exp"] - 1]
            if len(spec) == 1:
                assert r["dtype_exp"] == "&mdash;"
            else:
                for dtype in [spec[1]] if isinstance(spec[1], str) else spec[1]:
                    assert dtype in r["dtype_exp"]

    # The full report (with its header) can be generated
    assert isinstance(validation.get_step_report(i=1), GT.GT)


def test_col_schema_match_step_report_labels():
    validation = pb.Validate(TBL).col_schema_match(pb.Schema(columns=FULL)).interrogate()
    report_html = validation.get_step_report(i=1).as_raw_html()

    assert ">TABLE<" in report_html
    assert ">SCHEMA<" in report_html
    assert "TARGET" not in report_html
    assert "EXPECTED" not in report_html


def test_col_schema_match_does_not_modify_schema():
    columns = [("id", ["INT64", "Int32"]), ("Name", "STRING")]
    schema = pb.Schema(columns=columns)

    validation = (
        pb.Validate(TBL)
        .col_schema_match(schema, complete=False, case_sensitive_dtypes=False)
        .interrogate()
    )

    assert validation.validation_info[0].all_passed
    assert schema.columns == [("id", ["INT64", "Int32"]), ("Name", "STRING")]

    # The report shows the dtypes as supplied, with the matching one underlined
    df = report_df(validation)
    assert "INT64</span> | Int32" in df["dtype_exp"][0]
    assert df["dtype_exp"][1] == "STRING"


def test_col_schema_match_step_report_reordered():
    # With `Name` and `score` swapped, the schema columns are shown in schema order so that both
    # sit alongside the wrong table column and are flagged, with their dtypes left unscored
    validation = (
        pb.Validate(TBL).col_schema_match(pb.Schema(columns=swap(FULL, 1, 2))).interrogate()
    )
    df = report_df(validation)

    assert df["col_name_exp"].to_list() == [c for c, _ in swap(TGT, 1, 2)]
    assert df["index_exp"].to_list() == [1, 2, 3, 4, 5, 6]
    assert [CROSS in x for x in df["col_name_exp_correct"]] == [False, True, True] + [False] * 3
    assert df["dtype_exp_correct"].to_list()[1:3] == ["&mdash;", "&mdash;"]
    assert all(
        CHECK in x for x in df["dtype_exp_correct"].to_list()[3:] + [df["dtype_exp_correct"][0]]
    )

    # Without `in_order=True`, each schema column is shown alongside the table column it matched
    validation = (
        pb.Validate(TBL)
        .col_schema_match(pb.Schema(columns=swap(FULL, 1, 2)), in_order=False)
        .interrogate()
    )
    df = report_df(validation)

    assert df["col_name_exp"].to_list() == [c for c, _ in TGT]
    assert df["index_exp"].to_list() == [1, 3, 2, 4, 5, 6]
    assert not any(CROSS in x for x in df["col_name_exp_correct"])

    # The schema positions of the swapped columns are pointed out (in blue, with a dot)
    report_html = _step_report_schema(
        step=1,
        schema_info=validation.validation_info[0].val_info,
        header=None,
        lang="en",
    ).as_raw_html()
    body_rows = report_html.split("<tbody")[1].split("<tr>")[1:]
    assert ["border-radius: 50%" in x for x in body_rows] == [False, True, True] + [False] * 3
    assert ["#336699" in x for x in body_rows] == [False, True, True] + [False] * 3


def test_col_schema_match_step_report_subset_in_order():
    # A subset in table order passes and shows no failures; the skipped table columns are blank
    schema = pb.Schema(columns=[FULL[0], FULL[2], FULL[4]])
    validation = (
        pb.Validate(TBL).col_schema_match(schema, complete=False, in_order=True).interrogate()
    )
    df = report_df(validation)

    assert validation.validation_info[0].all_passed
    assert df["col_name_exp"].to_list() == ["id", "", "score", "", "date_time", ""]
    assert df["index_exp"].to_list() == [1, None, 2, None, 3, None]
    assert not any(CROSS in x for x in df["col_name_exp_correct"])


def test_col_schema_match_step_report_prefix_names():
    # `date` matches `date`, not `date_time` (and `date_time` matches `date_time`)
    schema = pb.Schema(columns=[FULL[4], FULL[3]])
    validation = (
        pb.Validate(TBL).col_schema_match(schema, complete=False, in_order=False).interrogate()
    )
    df = report_df(validation)

    assert validation.validation_info[0].all_passed
    assert df["col_name_exp"].to_list() == ["", "", "", "date", "date_time", ""]
    assert df["index_exp"].to_list() == [None, None, None, 2, 1, None]


def test_col_schema_match_step_report_unmatched_columns():
    # Unmatched schema columns follow the table columns, with their positions in the schema
    schema = pb.Schema(
        columns=[("extra", "Int64")] + replace(FULL, "Name", ("full_name", "String"))
    )
    validation = pb.Validate(TBL).col_schema_match(schema, in_order=False).interrogate()
    df = report_df(validation)

    assert df.height == 8
    assert df["col_name_exp"].to_list()[6:] == ["extra", "full_name"]
    assert df["index_exp"].to_list()[6:] == [1, 3]

    # `Name` is in the table but not in the (complete) schema, so its row has a cross in place of
    # the schema position and is struck through
    assert df["col_name_exp"][1] == ""
    assert df["index_exp"][1] is None

    report_html = _step_report_schema(
        step=1,
        schema_info=validation.validation_info[0].val_info,
        header=None,
        lang="en",
    ).as_raw_html()
    rows_html = report_html.split("<tr>")
    name_row = next(x for x in rows_html if ">Name<" in x)
    assert CROSS in name_row
    assert name_row.count("linear-gradient") == 4
    assert all("linear-gradient" not in x for x in rows_html if ">Name<" not in x)


@pytest.mark.parametrize("complete, in_order", COMBOS)
def test_col_schema_match_duplicate_columns_fail(complete, in_order):
    # A schema naming a column twice never passes; the repeat is shown as unmatched
    for columns in ([FULL[0], FULL[0]] + FULL[1:], [("id",), ("id",)], [FULL[0], FULL[0], FULL[2]]):
        validation = (
            pb.Validate(TBL)
            .col_schema_match(pb.Schema(columns=columns), complete=complete, in_order=in_order)
            .interrogate()
        )
        df = report_df(validation)

        assert not validation.validation_info[0].all_passed
        assert validation.validation_info[0].val_info["columns_duplicated"] == ["id"]
        assert df["col_name_exp"].to_list().count("id") == 2

        # The repeat is shown in the row of its own position when the schema should be the
        # table itself (complete and in order), otherwise in the first row after the table's
        repeat_row = 1 if complete and in_order else len(TGT)
        assert df["index_exp"][repeat_row] == 2
        assert CROSS in df["col_name_exp_correct"][repeat_row]


def test_col_schema_match_complete_in_order_lays_out_schema_from_top():
    # With complete=True and in_order=True the schema columns read top to bottom with no gaps,
    # and the table columns the schema runs out before are struck through together at the bottom
    validation = (
        pb.Validate(TBL)
        .col_schema_match(pb.Schema(columns=[("score",), ("id",)]), complete=True, in_order=True)
        .interrogate()
    )
    df = report_df(validation)

    assert df["col_name_exp"].to_list() == ["score", "id", "", "", "", ""]
    assert df["index_exp"].to_list() == [1, 2, None, None, None, None]
    assert all(CROSS in m for m in df["col_name_exp_correct"].to_list()[:2])

    report_html = _step_report_schema(
        step=1,
        schema_info=validation.validation_info[0].val_info,
        header=None,
        lang="en",
    ).as_raw_html()
    body_rows = [x for x in report_html.split("<tr>") if "index_target" in x and "<td" in x]
    assert len(body_rows) == len(TGT)
    for k, row in enumerate(body_rows):
        assert row.count("linear-gradient") == (4 if k >= 2 else 0)


def test_col_schema_match_complete_in_order_renamed_column_stays_in_place():
    # A renamed column sits beside the table column it should have been, so the rest line up
    columns = [("full_name" if c == "Name" else c,) for c, _ in TGT]
    validation = (
        pb.Validate(TBL)
        .col_schema_match(pb.Schema(columns=columns), complete=True, in_order=True)
        .interrogate()
    )
    df = report_df(validation)
    j = [c for c, _ in TGT].index("Name")

    assert df.height == len(TGT)
    assert df["col_name_target"].to_list() == [c for c, _ in TGT]
    assert df["index_exp"].to_list() == list(range(1, len(TGT) + 1))
    marks = df["col_name_exp_correct"].to_list()
    assert CROSS in marks[j]
    assert all(CHECK in m for k, m in enumerate(marks) if k != j)


def test_match_schema_columns_case_insensitive():
    # An exact-case match is preferred over a case-insensitive one
    target = [("a", "Int64"), ("A", "String")]
    matches = _match_schema_columns(
        target_schema=target,
        expect_schema=[("A", "String")],
        case_sensitive_colnames=False,
        case_sensitive_dtypes=True,
        full_match_dtypes=True,
    )

    assert matches[0]["matched_to"] == "A"
    assert matches[0]["target_index"] == 1
    assert matches[0]["dtype_matched"]

    # Names differing only in case are duplicates when matching ignores case
    matches = _match_schema_columns(
        target_schema=target,
        expect_schema=[("a",), ("A",)],
        case_sensitive_colnames=False,
        case_sensitive_dtypes=True,
        full_match_dtypes=True,
    )

    assert [m["duplicate"] for m in matches] == [False, True]
    assert matches[1]["target_index"] is None


@pytest.mark.parametrize(
    "target_indices, out_of_order",
    [
        ([], set()),
        ([0, 1, 2], set()),
        ([0, 2, 4], set()),
        ([1, 0, 2], {0, 1}),
        ([2, 0, 1], {0, 1, 2}),
        ([0, 3, 1, 2], {1, 2, 3}),
        ([3, 1, 2, 0], {0, 3}),
        ([0, None, 2, 1], {2, 3}),
    ],
)
def test_schema_columns_out_of_order(target_indices, out_of_order):
    matches = [{"target_index": i} for i in target_indices]

    assert _schema_columns_out_of_order(matches) == out_of_order


def test_col_schema_match_note_failure_summary():
    # A subset (`complete=False`) in order with one wrong dtype only reports the dtype mismatch
    schema = pb.Schema(columns=[("id", "Int64"), ("score", "Int64")])
    validation = pb.Validate(TBL).col_schema_match(schema, complete=False).interrogate()
    note = _create_col_schema_match_note_html(validation.validation_info[0].val_info)

    assert "1 dtype mismatch(es)" in note
    assert "column count mismatch" not in note
    assert "wrong order" not in note

    # Columns out of order and a duplicated column
    schema = pb.Schema(columns=[("score",), ("id",), ("id",)])
    validation = pb.Validate(TBL).col_schema_match(schema, complete=False).interrogate()
    note = _create_col_schema_match_note_html(validation.validation_info[0].val_info)

    assert "1 unmatched column(s)" in note
    assert "2 column(s) in wrong order" in note
    assert "column count mismatch" not in note
