import sys

import pytest
from unittest.mock import patch

from pointblank.validate import load_dataset
from pointblank.draft import DraftValidation


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


_DRAFT = (
    "```python\n"
    "import pointblank as pb\n"
    "\n"
    "validation = (\n"
    "    pb.Validate(data=your_data)\n"
    '    .col_vals_gt(columns="d", value=100)\n'
    '    .col_vals_not_null(columns="a")\n'
    "    .interrogate()\n"
    ")\n"
    "\n"
    "validation\n"
    "```"
)


def test_draft_full_flow_mocked():
    pytest.importorskip("chatlas")
    small_table = load_dataset(dataset="small_table")
    fake = _FakeChat([_DRAFT])
    with patch("pointblank.draft._create_chat_instance", return_value=fake):
        draft = DraftValidation(data=small_table, model="anthropic:claude-opus-4-8")

    assert "col_vals_gt" in str(draft)
    assert draft.validate_syntax()
    changes = draft.changed_steps()
    assert {c["action"] for c in changes} == {"add"}
    assert {c["method"] for c in changes} == {"col_vals_gt", "col_vals_not_null"}


def test_draft_reprompts_on_bad_syntax():
    pytest.importorskip("chatlas")
    small_table = load_dataset(dataset="small_table")
    bad = "```python\nvalidation = pb.Validate(data=your_data).col_vals_bogus(columns='a')\n```"
    fake = _FakeChat([bad, _DRAFT])
    with patch("pointblank.draft._create_chat_instance", return_value=fake):
        draft = DraftValidation(data=small_table, model="anthropic:claude-opus-4-8")

    assert draft.validate_syntax()
    assert len(fake.prompts) == 2
    assert "has a problem" in fake.prompts[1]


def test_draft_fail_no_chatlas():
    with patch.dict(sys.modules, {"chatlas": None}):
        with pytest.raises(ImportError):
            DraftValidation(data="data", model="model")


def test_draft_fail_invalid_provider():
    small_table = load_dataset(dataset="small_table")

    with pytest.raises(ValueError):
        DraftValidation(data=small_table, model="invalid:model")


def test_draft_fail_azure_openai_missing_endpoint(monkeypatch):
    pytest.importorskip("openai")
    pytest.importorskip("chatlas")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-06-01")
    small_table = load_dataset(dataset="small_table")
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        DraftValidation(data=small_table, model="azure-openai:my-deployment")


def test_draft_fail_azure_openai_missing_api_version(monkeypatch):
    pytest.importorskip("openai")
    pytest.importorskip("chatlas")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.delenv("OPENAI_API_VERSION", raising=False)
    small_table = load_dataset(dataset="small_table")
    with pytest.raises(ValueError, match="OPENAI_API_VERSION"):
        DraftValidation(data=small_table, model="azure-openai:my-deployment")
