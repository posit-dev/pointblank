import sys

import pytest
from unittest.mock import patch

from pointblank.validate import load_dataset
from pointblank.draft import DraftValidation


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
