import sys
from unittest.mock import patch

import pytest

import pointblank as pb
from pointblank import EditValidation, Validate, load_dataset
from pointblank._utils_ai import _diff_plan_steps, _extract_chain_steps
from pointblank.edit import (
    _check_syntax,
    _extract_code,
    _normalize_to_code,
)


class _FakeChat:
    """A stand-in chatlas chat that returns canned responses in sequence."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def chat(self, prompt, stream=False, echo="none"):
        self.prompts.append(prompt)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _patch_chat(responses):
    """Patch the chat-instance factory used by EditValidation to return a fake chat."""
    fake = _FakeChat(responses)
    return patch("pointblank.edit._create_chat_instance", return_value=fake), fake


@pytest.fixture
def small_table():
    return load_dataset(dataset="small_table")


# ─── helpers ───────────────────────────────────────────────────────────────────────


def test_extract_code_strips_python_fence():
    text = "here you go\n```python\nimport pointblank as pb\nx = 1\n```\nthanks"
    assert _extract_code(text) == "import pointblank as pb\nx = 1"


def test_extract_code_strips_bare_fence():
    text = "```\nx = 1\n```"
    assert _extract_code(text) == "x = 1"


def test_extract_code_no_fence_returns_trimmed():
    assert _extract_code("  x = 1  ") == "x = 1"


def test_check_syntax_ok():
    ok, msg = _check_syntax("import pointblank as pb\nx = pb.Validate")
    assert ok
    assert msg == ""


def test_check_syntax_detects_syntax_error():
    ok, msg = _check_syntax("def (:\n")
    assert not ok
    assert "SyntaxError" in msg


def test_check_syntax_detects_unknown_method():
    code = "validation = pb.Validate(data=your_data).col_vals_grater(columns='a', value=1)"
    ok, msg = _check_syntax(code)
    assert not ok
    assert "col_vals_grater" in msg


def test_normalize_from_validate_object(small_table):
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=0)
    code = _normalize_to_code(validation)
    assert "pb.Validate(" in code
    assert ".col_vals_gt(" in code


def test_normalize_from_code_string_passthrough():
    code = "validation = pb.Validate(data=your_data).col_vals_gt(columns='d', value=0)"
    assert _normalize_to_code(code) == code


def test_normalize_from_yaml_string():
    yaml_str = """
tbl: small_table
steps:
- col_vals_gt:
    columns: [d]
    value: 100
"""
    code = _normalize_to_code(yaml_str)
    assert "pb.Validate(" in code
    assert "col_vals_gt" in code


def test_normalize_from_yaml_file(tmp_path):
    yaml_file = tmp_path / "plan.yaml"
    yaml_file.write_text(
        "tbl: small_table\nsteps:\n- col_vals_gt:\n    columns: [d]\n    value: 100\n"
    )
    code = _normalize_to_code(str(yaml_file))
    assert "col_vals_gt" in code


def test_normalize_from_py_file(tmp_path):
    py_file = tmp_path / "plan.py"
    py_file.write_text("validation = pb.Validate(data=your_data)\n")
    code = _normalize_to_code(str(py_file))
    assert "pb.Validate(data=your_data)" in code


def test_normalize_rejects_bad_type():
    with pytest.raises(TypeError):
        _normalize_to_code(12345)


# ─── construction / failure paths ────────────────────────────────────────────────


def test_edit_fail_no_chatlas():
    with patch.dict(sys.modules, {"chatlas": None}):
        with pytest.raises(ImportError):
            EditValidation(validation="x = 1", instruction="do it", model="anthropic:m")


def test_edit_fail_invalid_provider(small_table):
    pytest.importorskip("chatlas")
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=0)
    with pytest.raises(ValueError, match="Unsupported provider"):
        EditValidation(validation=validation, instruction="do it", model="invalid:model")


def test_edit_fail_model_without_colon(small_table):
    pytest.importorskip("chatlas")
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=0)
    with pytest.raises(ValueError, match="provider:model"):
        EditValidation(validation=validation, instruction="do it", model="anthropic")


def test_edit_accept_strips_model_added_interrogate(small_table):
    pytest.importorskip("chatlas")
    resp = (
        "```python\nimport pointblank as pb\n"
        'validation = pb.Validate(data=your_data).col_vals_gt(columns="d", value=50)'
        ".interrogate()\n```"
    )
    ctx, _ = _patch_chat([resp])
    with ctx:
        edited = EditValidation(
            validation=Validate(data=small_table).col_vals_gt(columns="d", value=100),
            instruction="loosen",
            model="anthropic:claude-opus-4-8",
            data=small_table,
        )
    plan = edited.accept()
    # accept() must return a plan that is NOT yet interrogated, per its contract
    assert plan.validation_info[0].n_passed is None
    assert plan.interrogate().validation_info[0].n_passed is not None


def test_edit_accept_finds_validate_under_other_name(small_table):
    pytest.importorskip("chatlas")
    resp = (
        "```python\nimport pointblank as pb\n"
        'my_plan = pb.Validate(data=your_data).col_vals_gt(columns="d", value=50)\n```'
    )
    ctx, _ = _patch_chat([resp])
    with ctx:
        edited = EditValidation(
            validation=Validate(data=small_table).col_vals_gt(columns="d", value=100),
            instruction="x",
            model="anthropic:claude-opus-4-8",
            data=small_table,
        )
    plan = edited.accept()
    assert isinstance(plan, pb.Validate)


# ─── full flow (mocked model) ────────────────────────────────────────────────────

_EDITED = (
    "```python\n"
    "import pointblank as pb\n"
    "\n"
    "validation = (\n"
    "    pb.Validate(\n"
    "        data=your_data,  # Replace your_data with the actual data variable\n"
    "    )\n"
    '    .col_vals_gt(columns="d", value=50)\n'
    '    .col_vals_not_null(columns="a")\n'
    ")\n"
    "\n"
    "validation\n"
    "```"
)


def test_edit_full_flow_to_code_and_diff(small_table):
    pytest.importorskip("chatlas")
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=100)
    ctx, fake = _patch_chat([_EDITED])
    with ctx:
        edited = EditValidation(
            validation=validation,
            instruction="Loosen d to 50 and add a not-null on a",
            model="anthropic:claude-opus-4-8",
            data=small_table,
        )

    code = edited.to_code()
    assert "value=50" in code
    assert "col_vals_not_null" in code
    assert edited.validate_syntax()

    diff = edited.diff()
    assert "value=100" in diff
    assert "value=50" in diff
    assert diff.startswith("---")

    # The prompt should include the current plan and the instruction
    assert "value=100" in fake.prompts[0]
    assert "Loosen d to 50" in fake.prompts[0]


def test_edit_accept_returns_validate(small_table):
    pytest.importorskip("chatlas")
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=100)
    ctx, _ = _patch_chat([_EDITED])
    with ctx:
        edited = EditValidation(
            validation=validation,
            instruction="edit",
            model="anthropic:claude-opus-4-8",
            data=small_table,
        )
    plan = edited.accept()
    assert isinstance(plan, pb.Validate)
    assert [s.assertion_type for s in plan.validation_info] == [
        "col_vals_gt",
        "col_vals_not_null",
    ]
    # Not yet interrogated
    assert plan.validation_info[0].n_passed is None

    # Fully runnable
    result = plan.interrogate()
    assert result.validation_info[0].n_passed is not None


def test_edit_accept_requires_data():
    pytest.importorskip("chatlas")
    ctx, _ = _patch_chat([_EDITED])
    with ctx:
        edited = EditValidation(
            validation="validation = pb.Validate(data=your_data)",
            instruction="edit",
            model="anthropic:claude-opus-4-8",
        )
    with pytest.raises(ValueError, match="requires a data table"):
        edited.accept()


def test_edit_changed_steps(small_table):
    pytest.importorskip("chatlas")
    validation = (
        Validate(data=small_table)
        .col_vals_gt(columns="d", value=100)
        .col_vals_regex(columns="b", pattern="x")
    )
    ctx, _ = _patch_chat([_EDITED])
    with ctx:
        edited = EditValidation(
            validation=validation,
            instruction="edit",
            model="anthropic:claude-opus-4-8",
            data=small_table,
        )
    changes = edited.changed_steps()
    actions = {(c["action"], c["method"]) for c in changes}
    # d threshold changed (modify), regex removed, not_null added
    assert ("modify", "col_vals_gt") in actions
    assert ("remove", "col_vals_regex") in actions
    assert ("add", "col_vals_not_null") in actions


def test_edit_preview_returns_gt(small_table):
    pytest.importorskip("chatlas")
    pytest.importorskip("polars")
    from great_tables import GT

    validation = Validate(data=small_table).col_vals_gt(columns="d", value=100)
    ctx, _ = _patch_chat([_EDITED])
    with ctx:
        edited = EditValidation(
            validation=validation,
            instruction="edit",
            model="anthropic:claude-opus-4-8",
            data=small_table,
        )
    gt = edited.preview()
    assert isinstance(gt, GT)
    # Renders without error
    assert isinstance(gt.as_raw_html(), str)


# ─── shared step-parsing helpers ─────────────────────────────────────────────────


def test_extract_chain_steps_parses_methods():
    code = (
        "validation = (\n"
        "    pb.Validate(data=your_data)\n"
        '    .col_vals_gt(columns="d", value=100)\n'
        "    .rows_distinct()\n"
        "    .interrogate()\n"
        ")"
    )
    steps = _extract_chain_steps(code)
    assert [s["method"] for s in steps] == ["col_vals_gt", "rows_distinct"]
    assert steps[0]["kwargs"]["value"] == "100"


def test_diff_plan_steps_add_remove_modify():
    old = _extract_chain_steps(
        "pb.Validate(data=your_data)"
        '.col_vals_gt(columns="d", value=100).col_vals_regex(columns="b", pattern="x")'
    )
    new = _extract_chain_steps(
        "pb.Validate(data=your_data)"
        '.col_vals_gt(columns="d", value=50).col_vals_not_null(columns="a")'
    )
    changes = _diff_plan_steps(old, new)
    actions = {(c["action"], c["method"]) for c in changes}
    assert ("modify", "col_vals_gt") in actions
    assert ("remove", "col_vals_regex") in actions
    assert ("add", "col_vals_not_null") in actions


def test_extract_chain_steps_handles_bad_code():
    assert _extract_chain_steps("def (:\n") == []
    assert _extract_chain_steps("x = 1") == []


# ─── Validate.suggest_improvements / Validate.from_prompt ─────────────────────────


def test_suggest_improvements_returns_editvalidation(small_table):
    pytest.importorskip("chatlas")
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=100)
    ctx, fake = _patch_chat([_EDITED])
    with ctx:
        proposal = validation.suggest_improvements(model="anthropic:claude-opus-4-8")
    assert isinstance(proposal, EditValidation)
    # The prompt should mention uncovered columns and reference the data profile
    assert "no validation coverage" in fake.prompts[0]
    assert "json" in fake.prompts[0].lower()


def test_auto_improvement_instruction_flags_uncovered_and_thresholds(small_table):
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=100)
    instruction = validation._auto_improvement_instruction()
    # Parse the comma-separated uncovered-columns list from the instruction
    listed = instruction.split("no validation coverage:")[1].split(".")[0]
    uncovered = {c.strip() for c in listed.split(",")}
    # 'd' is covered; 'a' is not
    assert "a" in uncovered
    assert "d" not in uncovered
    assert "threshold" in instruction.lower()


def test_from_prompt_starts_from_empty_plan(small_table):
    pytest.importorskip("chatlas")
    base = Validate(data=small_table, tbl_name="small_table").col_vals_gt(columns="d", value=100)
    ctx, fake = _patch_chat([_EDITED])
    with ctx:
        proposal = base.from_prompt(
            "ensure a has no nulls and d is positive", model="anthropic:claude-opus-4-8"
        )
    assert isinstance(proposal, EditValidation)
    # The starting plan sent to the model must be empty (no col_vals_* before the instruction)
    current_plan = proposal.original_code
    assert "col_vals" not in current_plan
    assert "pb.Validate(" in current_plan
    plan = proposal.accept()
    assert isinstance(plan, pb.Validate)


def test_edit_reprompts_on_bad_syntax(small_table):
    pytest.importorskip("chatlas")
    bad = "```python\nvalidation = pb.Validate(data=your_data).col_vals_bogus(columns='a')\n```"
    good = (
        '```python\nvalidation = pb.Validate(data=your_data).col_vals_gt(columns="d", value=1)\n```'
    )
    validation = Validate(data=small_table).col_vals_gt(columns="d", value=100)
    ctx, fake = _patch_chat([bad, good])
    with ctx:
        edited = EditValidation(
            validation=validation,
            instruction="edit",
            model="anthropic:claude-opus-4-8",
            data=small_table,
            max_reprompts=1,
        )
    # The re-prompt should have kicked in and produced valid code
    assert edited.validate_syntax()
    assert len(fake.prompts) == 2
    assert "has a problem" in fake.prompts[1]
