from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pointblank.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class _FakeChat:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def chat(self, prompt, stream=False, echo="none"):
        self.prompts.append(prompt)
        return self.response


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

_PLAN_PY = (
    "import pointblank as pb\n"
    "\n"
    "validation = (\n"
    "    pb.Validate(\n"
    "        data=your_data,  # Replace your_data with the actual data variable\n"
    "    )\n"
    '    .col_vals_gt(columns="d", value=100)\n'
    ")\n"
    "\n"
    "validation\n"
)


def test_edit_requires_instruction(runner, tmp_path):
    plan = tmp_path / "plan.py"
    plan.write_text(_PLAN_PY)
    result = runner.invoke(cli, ["edit", str(plan), "-m", "anthropic:x"])
    assert result.exit_code != 0
    assert "instruction" in result.output


def test_edit_requires_model(runner, tmp_path):
    plan = tmp_path / "plan.py"
    plan.write_text(_PLAN_PY)
    result = runner.invoke(cli, ["edit", str(plan), "-i", "do it"])
    assert result.exit_code != 0
    assert "model" in result.output


def test_edit_shows_diff_and_summary(runner, tmp_path):
    pytest.importorskip("chatlas")
    plan = tmp_path / "plan.py"
    plan.write_text(_PLAN_PY)
    fake = _FakeChat(_EDITED)
    with patch("pointblank.edit._create_chat_instance", return_value=fake):
        result = runner.invoke(
            cli,
            [
                "edit",
                str(plan),
                "-i",
                "loosen d to 50 and add not-null on a",
                "-m",
                "anthropic:claude-opus-4-8",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "value=50" in result.output
    assert "added" in result.output


def test_edit_writes_output_file(runner, tmp_path):
    pytest.importorskip("chatlas")
    plan = tmp_path / "plan.py"
    plan.write_text(_PLAN_PY)
    out = tmp_path / "plan2.py"
    fake = _FakeChat(_EDITED)
    with patch("pointblank.edit._create_chat_instance", return_value=fake):
        result = runner.invoke(
            cli,
            [
                "edit",
                str(plan),
                "-i",
                "edit it",
                "-m",
                "anthropic:claude-opus-4-8",
                "-o",
                str(out),
                "-y",
            ],
        )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "value=50" in out.read_text()
