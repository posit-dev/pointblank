import warnings

import pytest

import pointblank as pb
from pointblank import Validate, load_dataset, yaml_interrogate


def _step_signatures(validation):
    """A comparable summary of a plan's steps (independent of interrogation results)."""

    def _norm(value):
        if isinstance(value, pb.Schema):
            return ("Schema", tuple(value.columns))
        return value

    return [
        (
            s.assertion_type,
            _norm(s.column),
            _norm(s.values),
            None if s.inclusive is None else tuple(s.inclusive),
            s.na_pass,
        )
        for s in validation.validation_info
    ]


def _exec_code(code, data):
    """Execute generated `to_code()` output (with `your_data` bound to `data`)."""
    namespace = {}
    exec(code.replace("your_data", "data"), {"pb": pb, "data": data}, namespace)
    return namespace["validation"]


@pytest.fixture
def small_table():
    return load_dataset(dataset="small_table")


# ─── to_code() ───────────────────────────────────────────────────────────────────


def test_to_code_returns_parseable_python(small_table):
    import ast

    validation = Validate(data=small_table).col_vals_gt(columns="d", value=100)
    code = validation.to_code()

    assert "import pointblank as pb" in code
    assert "pb.Validate(" in code
    assert "data=your_data" in code
    assert '.col_vals_gt(columns="d", value=100)' in code
    # The generated code should parse without error
    ast.parse(code)


def test_to_code_roundtrips_diverse_methods(small_table):
    validation = (
        Validate(data=small_table, tbl_name="small_table", label="My Plan")
        .col_vals_gt(columns="d", value=100)
        .col_vals_lt(columns="c", value=10)
        .col_vals_between(columns="c", left=1, right=10, inclusive=(False, True))
        .col_vals_in_set(columns="f", set=["low", "mid", "high"])
        .col_vals_regex(columns="b", pattern="[0-9]-[a-z]{3}")
        .col_vals_null(columns="c")
        .col_vals_not_null(columns="a")
        .col_exists(columns="a")
        .rows_distinct()
        .rows_complete()
        .row_count_match(count=13)
        .col_count_match(count=8)
    )

    rebuilt = _exec_code(validation.to_code(), small_table)
    assert _step_signatures(validation) == _step_signatures(rebuilt)


def test_to_code_roundtrips_schema_match(small_table):
    schema = pb.Schema(columns=[("a", "Int64"), ("b", "String")])
    validation = Validate(data=small_table).col_schema_match(schema=schema, complete=False)

    rebuilt = _exec_code(validation.to_code(), small_table)
    original_schema = validation.validation_info[0].values["schema"]
    rebuilt_schema = rebuilt.validation_info[0].values["schema"]
    assert original_schema.columns == rebuilt_schema.columns
    assert rebuilt.validation_info[0].values["complete"] is False


def test_to_code_coalesces_adjacent_column_steps(small_table):
    validation = (
        Validate(data=small_table).col_vals_not_null(columns="a").col_vals_not_null(columns="b")
    )
    code = validation.to_code()
    assert '.col_vals_not_null(columns=["a", "b"])' in code


def test_to_code_renders_top_level_arguments(small_table):
    validation = Validate(
        data=small_table,
        tbl_name="my_table",
        label="A Label",
        thresholds=pb.Thresholds(warning=0.1, error=0.25, critical=0.35),
        owner="data-team",
        version="1.2.0",
    ).col_vals_gt(columns="d", value=0)

    code = validation.to_code()
    assert 'tbl_name="my_table"' in code
    assert 'label="A Label"' in code
    assert "thresholds=pb.Thresholds(warning=0.1, error=0.25, critical=0.35)" in code
    assert 'owner="data-team"' in code
    assert 'version="1.2.0"' in code


def test_to_code_omits_default_language(small_table):
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=0)
    code = validation.to_code()
    assert "lang=" not in code
    assert "locale=" not in code


def test_to_code_step_level_thresholds_not_duplicated_when_matching_global(small_table):
    thresholds = pb.Thresholds(warning=0.1)
    validation = Validate(data=small_table, thresholds=thresholds).col_vals_gt(columns="d", value=0)
    code = validation.to_code()
    # Threshold appears once (top-level), not repeated on the step.
    assert code.count("pb.Thresholds(") == 1


def test_to_code_step_level_thresholds_rendered_when_differing(small_table):
    validation = Validate(data=small_table).col_vals_gt(
        columns="d", value=0, thresholds=pb.Thresholds(warning=0.5)
    )
    code = validation.to_code()
    assert "thresholds=pb.Thresholds(warning=0.5)" in code


def test_to_code_na_pass_rendered_only_when_true(small_table):
    validation = (
        Validate(data=small_table)
        .col_vals_gt(columns="d", value=0, na_pass=True)
        .col_vals_lt(columns="c", value=100)
    )
    code = validation.to_code()
    assert code.count("na_pass=True") == 1


def test_to_code_warns_on_unserializable_pre(small_table):
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=0, pre=lambda df: df)
    with pytest.warns(UserWarning, match="could not be fully serialized"):
        code = validation.to_code()
    # Placeholder keeps the code valid.
    import ast

    ast.parse(code)


def test_to_code_warns_on_specially(small_table):
    validation = Validate(data=small_table).specially(expr=lambda df: df.shape[0] > 0)
    with pytest.warns(UserWarning, match="could not be fully serialized"):
        code = validation.to_code()
    import ast

    ast.parse(code)


# ─── to_yaml() ───────────────────────────────────────────────────────────────────


def test_to_code_col_vals_in_table_uses_ref_table_placeholder():
    # The reference table can't be serialized, so it becomes a placeholder (with a warning); the
    # step itself, with its columns and options, is kept
    import polars as pl

    tbl = pl.DataFrame({"a": [1, 2, 3], "b": ["p", "q", "r"]})
    ref = pl.DataFrame({"x": [1, 2], "y": ["p", "q"]})
    validation = (
        Validate(data=tbl)
        .col_vals_in_table(columns="a", ref_table=ref, ref_column="x", na_pass=True)
        .col_vals_in_table(columns=["a", "b"], ref_table=ref, ref_column=["x", "y"])
    )

    with pytest.warns(UserWarning, match="reference table that cannot be serialized"):
        code = validation.to_code()

    assert (
        '.col_vals_in_table(columns="a", ref_table=your_reference_table, ref_column="x", '
        "na_pass=True)" in code
    )

    # With the placeholder bound to a table, the generated code gives the same plan
    namespace = {}
    exec(
        code.replace("your_data", "data"),
        {"pb": pb, "data": tbl, "your_reference_table": ref},
        namespace,
    )
    rebuilt = namespace["validation"].interrogate()
    original = validation.interrogate()
    assert rebuilt.n_failed(scalar=False) == original.n_failed(scalar=False) == {1: 1, 2: 1}

    with pytest.warns(UserWarning, match="reference table that cannot be serialized"):
        yaml_text = validation.to_yaml()
    assert "ref_table: your_reference_table" in yaml_text


def test_col_vals_in_table_ref_path_roundtrips(tmp_path):
    # A path (or connection string) is kept as-is, so the plan round-trips without a placeholder
    import polars as pl

    ref_path = tmp_path / "ref.csv"
    pl.DataFrame({"x": [1, 2]}).write_csv(ref_path)
    tbl = pl.DataFrame({"a": [1, 2, 3]})
    validation = Validate(data=tbl).col_vals_in_table(
        columns="a", ref_table=str(ref_path), ref_column="x"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        code = validation.to_code()
        yaml_text = validation.to_yaml()

    assert f"ref_table={str(ref_path)!r}" in code or f'ref_table="{ref_path}"' in code
    assert _exec_code(code, tbl).interrogate().n_failed(i=1, scalar=True) == 1

    # With `set_tbl=`, the placeholder `tbl: your_data` is never loaded
    assert "tbl: your_data" in yaml_text
    rebuilt = yaml_interrogate(yaml_text, set_tbl=tbl)
    assert rebuilt.n_failed(i=1, scalar=True) == 1


def test_to_yaml_roundtrips_via_yaml_interrogate(small_table):
    validation = (
        Validate(data=small_table, tbl_name="small_table", label="My Plan")
        .col_vals_gt(columns="d", value=100)
        .col_vals_between(columns="c", left=1, right=10, inclusive=(False, True))
        .col_vals_in_set(columns="f", set=["low", "mid", "high"])
        .col_vals_not_null(columns="a")
        .rows_distinct()
        .row_count_match(count=13)
    )

    yaml_str = validation.to_yaml()
    rebuilt = yaml_interrogate(yaml_str)
    assert _step_signatures(validation) == _step_signatures(rebuilt)


def test_to_yaml_sets_tbl_from_tbl_name(small_table):
    validation = Validate(data=small_table, tbl_name="small_table").col_vals_gt(
        columns="d", value=0
    )
    yaml_str = validation.to_yaml()
    assert "tbl: small_table" in yaml_str


def test_to_yaml_uses_placeholder_when_no_tbl_name(small_table):
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=0)
    yaml_str = validation.to_yaml()
    assert "tbl: your_data" in yaml_str


def test_to_yaml_writes_to_path(small_table, tmp_path):
    validation = Validate(data=small_table, tbl_name="small_table").col_vals_gt(
        columns="d", value=0
    )
    out = tmp_path / "plan.yaml"
    returned = validation.to_yaml(path=out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == returned


def test_to_yaml_roundtrips_schema_match(small_table):
    schema = pb.Schema(columns=[("a", "Int64"), ("b", "String")])
    validation = Validate(data=small_table, tbl_name="small_table").col_schema_match(schema=schema)
    yaml_str = validation.to_yaml()
    rebuilt = yaml_interrogate(yaml_str)
    assert rebuilt.validation_info[0].values["schema"].columns == schema.columns


def test_to_yaml_warns_on_unserializable(small_table):
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=0, pre=lambda df: df)
    with pytest.warns(UserWarning, match="could not be fully serialized"):
        validation.to_yaml()


def test_to_code_roundtrips_column_to_column_comparison(small_table):
    validation = Validate(data=small_table).col_vals_gt(columns="a", value=pb.col("d"))
    code = validation.to_code()
    assert 'value=pb.col("d")' in code
    rebuilt = _exec_code(code, small_table)
    from pointblank.column import Column

    assert isinstance(rebuilt.validation_info[0].values, Column)


def test_to_yaml_roundtrips_column_to_column_comparison(small_table):
    # A column comparison value must round-trip, not degrade to a bare string literal
    validation = Validate(data=small_table, tbl_name="small_table").col_vals_gt(
        columns="a", value=pb.col("d")
    )
    yaml_str = validation.to_yaml()
    assert "python: pb.col('d')" in yaml_str
    rebuilt = yaml_interrogate(yaml_str)
    from pointblank.column import Column

    assert isinstance(rebuilt.validation_info[0].values, Column)


def test_aggregate_method_columns_render_cleanly(small_table):
    # Aggregate methods store `column` as a list; it should serialize as a clean column name
    validation = Validate(data=small_table, tbl_name="small_table").col_sum_eq(
        columns="d", value=100
    )
    code = validation.to_code()
    assert '.col_sum_eq(columns="d", value=100)' in code

    rebuilt = _exec_code(code, small_table)
    assert rebuilt.validation_info[0].assertion_type == "col_sum_eq"

    yaml_rebuilt = yaml_interrogate(validation.to_yaml())
    assert yaml_rebuilt.validation_info[0].assertion_type == "col_sum_eq"


def test_data_freshness_roundtrips(small_table):
    validation = Validate(data=small_table, tbl_name="small_table").data_freshness(
        column="date_time", max_age="1 day"
    )
    code = validation.to_code()
    assert '.data_freshness(column="date_time", max_age="1 day")' in code

    rebuilt = _exec_code(code, small_table)
    assert rebuilt.validation_info[0].assertion_type == "data_freshness"

    yaml_rebuilt = yaml_interrogate(validation.to_yaml())
    assert yaml_rebuilt.validation_info[0].assertion_type == "data_freshness"


def test_empty_plan_serializes(small_table):
    validation = Validate(data=small_table)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = validation.to_code()
        yaml_str = validation.to_yaml()
    assert "pb.Validate(" in code
    assert "steps: []" in yaml_str


# ─── Serialization parity across every validation method ────────────────────────
#
# Each validation method needs a case here: it must serialize with `to_code()` and `to_yaml()`
# (not be skipped), and the serialized plan must rebuild into the same steps. Methods whose
# arguments can't be serialized (callables, LLM prompts) are checked for a buildable plan only.

_MISSING = pb.MissingSpec(reasons={8: "refused", -99: "not_asked"}, categories={"x": ["refused"]})


def _serialization_cases(ref_path):
    import polars as pl

    return {
        "col_vals_gt": lambda v: v.col_vals_gt(columns="d", value=1000, na_pass=True),
        "col_vals_lt": lambda v: v.col_vals_lt(columns="c", value=pb.col("a"), missing=_MISSING),
        "col_vals_eq": lambda v: v.col_vals_eq(columns="a", value=3),
        "col_vals_ne": lambda v: v.col_vals_ne(columns="a", value=3, segments="f"),
        "col_vals_ge": lambda v: v.col_vals_ge(
            columns="a", value=3, segments=("f", ["low", "mid"])
        ),
        "col_vals_le": lambda v: v.col_vals_le(columns="a", value=3, segments=["f", ("e", True)]),
        "col_vals_between": lambda v: v.col_vals_between(
            columns="c", left=1, right=8, inclusive=(False, True)
        ),
        "col_vals_outside": lambda v: v.col_vals_outside(columns="c", left=2, right=4),
        "col_vals_in_set": lambda v: v.col_vals_in_set(columns="f", set=["low", "mid"]),
        "col_vals_not_in_set": lambda v: v.col_vals_not_in_set(columns="f", set=["low"]),
        "col_vals_null": lambda v: v.col_vals_null(columns="c"),
        "col_vals_not_null": lambda v: v.col_vals_not_null(columns=["a", "c"]),
        "col_vals_regex": lambda v: v.col_vals_regex(columns="b", pattern="^[0-5]", inverse=True),
        "col_vals_within_spec": lambda v: v.col_vals_within_spec(columns="b", spec="isbn"),
        "col_vals_str_len": lambda v: v.col_vals_str_len(columns="b", min_val=9, max_val=9),
        "col_vals_increasing": lambda v: v.col_vals_increasing(
            columns="date_time", allow_stationary=True
        ),
        "col_vals_decreasing": lambda v: v.col_vals_decreasing(columns="a"),
        "col_vals_in_table": lambda v: v.col_vals_in_table(
            columns="a", ref_table=str(ref_path), ref_column="a"
        ),
        "col_exists": lambda v: v.col_exists(columns=["a", "z"]),
        "col_pct_null": lambda v: v.col_pct_null(columns="c", p=0.15, tol=(0.05, 0.1)),
        "col_pct_missing": lambda v: v.col_pct_missing(
            columns="c", missing=_MISSING, max_pct=0.3, category="x"
        ),
        "col_missing_coded": lambda v: v.col_missing_coded(columns="c", missing=_MISSING),
        "col_missing_only_coded": lambda v: v.col_missing_only_coded(
            columns="c", missing=_MISSING, min_val=0, max_val=7
        ),
        "col_missing_consistent": lambda v: v.col_missing_consistent(
            columns=["a", "c"], missing=_MISSING, when_reason="refused"
        ),
        "rows_distinct": lambda v: v.rows_distinct(columns_subset=["f"]),
        "rows_complete": lambda v: v.rows_complete(),
        "col_count_match": lambda v: v.col_count_match(count=7, inverse=True),
        "row_count_match": lambda v: v.row_count_match(count=10, tol=3),
        "col_schema_match": lambda v: v.col_schema_match(
            schema=pb.Schema(columns=[("a", "Int64"), ("z",)]), complete=False
        ),
        "data_freshness": lambda v: v.data_freshness(
            column="date_time", max_age="2 days", reference_time="2016-01-31 00:00:00"
        ),
        "tbl_match": lambda v: v.tbl_match(tbl_compare=str(ref_path)),
        "tbl_reconcile": lambda v: v.tbl_reconcile(
            tbl_compare=str(ref_path), keys="f", measures="a"
        ),
        "col_sum_gt": lambda v: v.col_sum_gt(columns="a", value=50),
        "col_avg_le": lambda v: v.col_avg_le(columns="a", value=4, tol=0.5),
        "col_sd_eq": lambda v: v.col_sd_eq(columns="a", value=2, tol=1),
        "col_vals_expr": lambda v: v.col_vals_expr(expr=pl.col("a") > 1),
        "conjointly": lambda v: v.conjointly(lambda df: pl.col("a") > 1),
        "specially": lambda v: v.specially(expr=lambda df: [True]),
        "prompt": lambda v: v.prompt(
            prompt="Values look plausible", model="anthropic:claude-sonnet-5", batch_size=10
        ),
    }


# Methods whose plans can't be fully serialized (callables, or LLM calls at interrogation), so
# only the rebuilt plan's structure is compared
_LOSSY_METHODS = {"col_vals_expr", "conjointly", "specially", "prompt"}

# Public `Validate` methods that don't add validation steps
_NON_STEP_METHODS = {
    "above_threshold", "add_steps", "all_passed", "assert_below_threshold",
    "assert_dimension_scores", "assert_passing", "critical", "error", "f_failed", "f_passed",
    "from_prompt", "get_data_extracts", "get_dataframe_report", "get_dimension_scores",
    "get_health_score", "get_json_report", "get_note", "get_notes", "get_scorecard",
    "get_step_report", "get_sundered_data", "get_tabular_report", "interrogate", "n",
    "n_failed", "n_passed", "set_tbl", "suggest_improvements", "to_code", "to_json_schema",
    "to_yaml", "warning",
}  # fmt: skip


@pytest.fixture
def ref_path(tmp_path, small_table):
    path = tmp_path / "ref.csv"
    small_table.head(10).write_csv(path)
    return path


def _results(validation):
    return [
        (s.assertion_type, s.n, s.n_passed, s.n_failed, s.eval_error)
        for s in validation.validation_info
    ]


def test_every_validation_method_has_a_serialization_case(tmp_path):
    import inspect

    from pointblank._agg import is_valid_agg

    public = {
        name
        for name, _ in inspect.getmembers(Validate, inspect.isfunction)
        if not name.startswith("_")
    }
    step_methods = public - _NON_STEP_METHODS
    cases = set(_serialization_cases(tmp_path / "ref.csv"))

    # Aggregate methods are covered by one case per aggregate kind
    uncovered = {m for m in step_methods - cases if not is_valid_agg(m)}
    assert not uncovered, f"Add serialization cases for: {sorted(uncovered)}"


@pytest.mark.parametrize("method", sorted(_serialization_cases(None)))
@pytest.mark.parametrize("interrogated", [False, True])
def test_every_validation_method_serializes(method, interrogated, small_table, ref_path):
    build = _serialization_cases(ref_path)[method]
    lossy = method in _LOSSY_METHODS

    validation = build(Validate(data=small_table, tbl_name="small_table"))
    if interrogated and not lossy:
        validation = validation.interrogate()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        code = validation.to_code()
        yaml_str = validation.to_yaml()
    messages = [str(w.message) for w in caught]
    assert not any("not supported by plan serialization" in m for m in messages), messages
    if not lossy:
        assert not messages, messages

    rebuilt_code = _exec_code(code, small_table)

    # `set_tbl=` is used, so the YAML's `tbl` is never loaded
    yaml_str = yaml_str.replace("tbl: small_table\n", "tbl: not_a_loadable_table\n", 1)
    if lossy:
        from pointblank.yaml import YAMLValidator

        validator = YAMLValidator()
        rebuilt_yaml = validator.build_validation(validator.load_config(yaml_str), tbl=small_table)
        expected = [s.assertion_type for s in validation.validation_info]
        assert [s.assertion_type for s in rebuilt_code.validation_info] == expected
        assert [s.assertion_type for s in rebuilt_yaml.validation_info] == expected
        return

    rebuilt_yaml = yaml_interrogate(yaml_str, set_tbl=small_table)
    expected = _results(validation if interrogated else validation.interrogate())
    assert not any(eval_error for *_, eval_error in expected)
    assert _results(rebuilt_code.interrogate()) == expected
    assert _results(rebuilt_yaml) == expected


def test_to_yaml_after_interrogation_writes_original_segments(small_table):
    # Interrogation expands a segmented step into one step per segment; serializing writes the
    # original `segments=` value once, not the expanded (column, value) pairs
    validation = (
        Validate(data=small_table, tbl_name="small_table")
        .col_vals_gt(columns="a", value=2, segments="f")
        .col_vals_gt(columns="a", value=2, segments=("f", "low"))
        .interrogate()
    )
    assert len(validation.validation_info) == 4

    code = validation.to_code()
    assert '.col_vals_gt(columns="a", value=2, segments="f")' in code
    assert '.col_vals_gt(columns="a", value=2, segments=("f", "low"))' in code

    yaml_str = validation.to_yaml()
    assert "segments: f\n" in yaml_str
    assert "segments:\n    - - f\n      - low\n" in yaml_str

    rebuilt = yaml_interrogate(yaml_str)
    assert [s.segments for s in rebuilt.validation_info] == [
        s.segments for s in validation.validation_info
    ]
    assert _results(rebuilt) == _results(validation)


def test_yaml_segments_pairs_and_columns(small_table):
    # In a `segments:` list, a string is a column and a two-item list is a (column, value) pair
    yaml_str = """
tbl: small_table
steps:
- col_vals_gt:
    columns: a
    value: 2
    segments: [f, [e, true], [f, [low, mid]]]
"""
    validation = yaml_interrogate(yaml_str)
    assert [s.segments for s in validation.validation_info] == [
        ("f", "high"),
        ("f", "low"),
        ("f", "mid"),
        ("e", True),
        ("f", "low"),
        ("f", "mid"),
    ]
    assert not any(s.eval_error for s in validation.validation_info)

    bad = yaml_str.replace("[f, [e, true], [f, [low, mid]]]", "[f, [e, true, false]]")
    with pytest.raises(Exception, match="Invalid 'segments' item"):
        yaml_interrogate(bad)


def test_yaml_interrogate_set_tbl_does_not_load_config_tbl(small_table):
    yaml_str = """
tbl: this_is_not_a_dataset
steps:
- col_vals_not_null:
    columns: a
"""
    validation = yaml_interrogate(yaml_str, set_tbl=small_table)
    assert validation.all_passed()


@pytest.mark.parametrize("value", ["pb.col('a')", 'pb.col("a")', "pb.ref('a')"])
def test_yaml_column_reference_value_without_python_block(small_table, value):
    from pointblank.column import Column, ReferenceColumn

    yaml_str = f"""
tbl: small_table
steps:
- col_vals_gt:
    columns: d
    value: {value}
"""
    if value.startswith("pb.ref"):
        from pointblank.yaml import YAMLValidator

        validator = YAMLValidator()
        built = validator.build_validation(validator.load_config(yaml_str))
        assert isinstance(built.validation_info[0].values, ReferenceColumn)
        return

    validation = yaml_interrogate(yaml_str)
    assert isinstance(validation.validation_info[0].values, Column)
    assert validation.all_passed()
    assert f"value={value}" in pb.yaml_to_python(yaml_str)


def test_to_code_conjointly_renders_positional_expressions(small_table):
    validation = Validate(data=small_table).conjointly(lambda df: df["a"] > 0)
    with pytest.warns(UserWarning, match="could not be fully serialized"):
        code = validation.to_code()
    assert ".conjointly(lambda df: df)" in code
    assert _exec_code(code, small_table).validation_info[0].assertion_type == "conjointly"


def test_to_code_renders_seg_group(small_table):
    validation = Validate(data=small_table).col_vals_gt(
        columns="a", value=2, segments=("f", pb.seg_group([["low", "mid"], ["high"]]))
    )
    code = validation.to_code()
    assert 'segments=("f", pb.seg_group([["low", "mid"], ["high"]]))' in code

    yaml_str = validation.to_yaml().replace("tbl: your_data", "tbl: small_table")
    rebuilt = yaml_interrogate(yaml_str)
    assert _results(rebuilt) == _results(validation.interrogate())
