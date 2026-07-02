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
