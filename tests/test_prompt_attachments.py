"""Tests for multi-modal ``attachments=`` support on ``Validate.prompt()``."""

from __future__ import annotations

import pathlib
import warnings

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from pointblank._utils_ai import (
    _LLMConfig,
    _PromptBuilder,
    _prepare_attachments,
)
from pointblank.validate import Validate


# A minimal valid PDF (header + EOF marker). content_pdf_file just reads bytes
# and base64-encodes them, so any well-formed PDF stub works.
_PDF_BYTES = (
    b"%PDF-1.1\n"
    b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Kids [] /Count 0 >> endobj\n"
    b"xref\n0 3\n0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000054 00000 n \n"
    b"trailer << /Size 3 /Root 1 0 R >>\n"
    b"startxref\n100\n%%EOF\n"
)


@pytest.fixture
def png_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Generate a real, decodable 1x1 PNG via Pillow (pulled in by chatlas)."""
    pil = pytest.importorskip("PIL.Image")
    p = tmp_path / "sample.png"
    pil.new("RGB", (1, 1), color=(255, 0, 0)).save(p, format="PNG")
    return p


@pytest.fixture
def pdf_path(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "sample.pdf"
    p.write_bytes(_PDF_BYTES)
    return p


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame({"description": ["red apple", "yellow banana"]})


# ============================================================================
# _prepare_attachments unit tests
# ============================================================================


def test_prepare_attachments_none_returns_empty():
    assert _prepare_attachments(None) == []


def test_prepare_attachments_empty_list_returns_empty():
    assert _prepare_attachments([]) == []


def test_prepare_attachments_rejects_non_list():
    with pytest.raises(TypeError, match="attachments must be a list or tuple"):
        _prepare_attachments("just_a_string.png")  # type: ignore[arg-type]


def test_prepare_attachments_rejects_unsupported_extension():
    pytest.importorskip("chatlas")
    with pytest.raises(ValueError, match="Unsupported attachment extension"):
        _prepare_attachments(["report.docx"])


def test_prepare_attachments_local_png(png_path: pathlib.Path):
    chatlas = pytest.importorskip("chatlas")
    result = _prepare_attachments([str(png_path)])
    assert len(result) == 1
    # Should be a chatlas Content image (inline, base64-encoded from disk).
    assert isinstance(result[0], chatlas.types.Content)
    assert "Image" in type(result[0]).__name__


def test_prepare_attachments_local_image_no_resize_warning(png_path: pathlib.Path):
    """Auto-coerced local images must not surface chatlas's MissingResizeWarning."""
    pytest.importorskip("chatlas")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _prepare_attachments([str(png_path)])
    assert not any(
        "MissingResizeWarning" in type(w.category).__name__
        or "resize" in str(w.message).lower()
        for w in captured
    ), [str(w.message) for w in captured]


def test_prepare_attachments_local_pdf_via_pathlib(pdf_path: pathlib.Path):
    chatlas = pytest.importorskip("chatlas")
    result = _prepare_attachments([pdf_path])  # pathlib.Path object, not str
    assert len(result) == 1
    assert isinstance(result[0], chatlas.types.Content)
    assert "PDF" in type(result[0]).__name__


def test_prepare_attachments_image_url():
    chatlas = pytest.importorskip("chatlas")
    # Image URLs are lazy in chatlas (no network call), so a fake host is fine.
    result = _prepare_attachments(["https://example.com/diagram.png"])
    assert len(result) == 1
    assert type(result[0]).__name__ == "ContentImageRemote"


def test_prepare_attachments_image_url_ignores_query_string():
    pytest.importorskip("chatlas")
    # Extension detection must strip ``?…`` before checking the suffix.
    result = _prepare_attachments(["https://example.com/diagram.png?cache=1"])
    assert len(result) == 1


def test_prepare_attachments_passes_content_through():
    """Already-built chatlas Content objects should not be re-coerced."""
    pytest.importorskip("chatlas")
    sentinel = object()  # opaque non-string, non-path — must pass through
    result = _prepare_attachments([sentinel])
    assert result == [sentinel]


def test_prepare_attachments_mixed_inputs(png_path: pathlib.Path):
    """A mix of paths, pathlib.Path, and prebuilt Content should all be accepted."""
    chatlas = pytest.importorskip("chatlas")
    prebuilt = chatlas.content_image_url("https://example.com/x.jpg")
    result = _prepare_attachments([str(png_path), png_path, prebuilt])
    assert len(result) == 3
    # First two are coerced into Content; third is the original prebuilt.
    assert result[2] is prebuilt


# ============================================================================
# Validate.prompt() integration with attachments=
# ============================================================================


def test_prompt_stores_attachments_in_ai_config(png_path: pathlib.Path, sample_data: pd.DataFrame):
    pytest.importorskip("chatlas")
    v = Validate(data=sample_data).prompt(
        prompt="Each description names a real fruit.",
        model="anthropic:claude-opus-4-6",
        attachments=[str(png_path)],
    )
    stored = v.validation_info[0].values["attachments"]
    assert len(stored) == 1
    assert "Image" in type(stored[0]).__name__


def test_prompt_backwards_compatible_without_attachments(sample_data: pd.DataFrame):
    """Existing callers that don't pass attachments= must keep working unchanged."""
    v = Validate(data=sample_data).prompt(
        prompt="Each description names a real fruit.",
        model="anthropic:claude-opus-4-6",
    )
    # The new key exists but is empty — no chatlas dependency exercised.
    assert v.validation_info[0].values["attachments"] == []


def test_prompt_rejects_bad_attachment_extension_at_step_definition(
    sample_data: pd.DataFrame,
):
    """Errors should surface when the step is defined, not later at interrogate()."""
    pytest.importorskip("chatlas")
    with pytest.raises(ValueError, match="Unsupported attachment extension"):
        Validate(data=sample_data).prompt(
            prompt="x",
            model="anthropic:claude-opus-4-6",
            attachments=["notes.txt"],
        )


# ============================================================================
# _AIValidationEngine forwards attachments to chat.chat()
# ============================================================================


def _make_engine_with_mock_chat(attachments):
    """Build an _AIValidationEngine whose ``chat`` is a Mock — no real chatlas call."""
    from pointblank._utils_ai import _AIValidationEngine

    cfg = _LLMConfig(provider="anthropic", model="claude-opus-4-6", api_key="test")
    fake_chat = MagicMock()
    fake_chat.chat.return_value = '[{"index": 0, "result": true}]'

    with patch("pointblank._utils_ai._create_chat_instance", return_value=fake_chat):
        engine = _AIValidationEngine(cfg, attachments=attachments)
    return engine, fake_chat


def _single_batch():
    return {
        "batch_id": 0,
        "start_row": 0,
        "end_row": 1,
        "data": {
            "columns": ["description"],
            "rows": [{"description": "red apple", "_pb_row_index": 0}],
            "batch_info": {"start_row": 0, "num_rows": 1, "columns_count": 1},
        },
    }


def test_engine_passes_attachments_to_chat_chat():
    sentinel_a, sentinel_b = object(), object()
    engine, fake_chat = _make_engine_with_mock_chat([sentinel_a, sentinel_b])

    builder = _PromptBuilder("Each row names a real fruit.")
    engine.validate_batches([_single_batch()], builder)

    fake_chat.chat.assert_called_once()
    args, kwargs = fake_chat.chat.call_args
    # First positional arg is the text prompt; the rest are the attachments.
    assert isinstance(args[0], str)
    assert "Each row names a real fruit." in args[0]
    assert args[1] is sentinel_a
    assert args[2] is sentinel_b
    assert kwargs == {"stream": False, "echo": "none"}


def test_engine_without_attachments_does_not_pass_extra_args():
    engine, fake_chat = _make_engine_with_mock_chat(None)

    builder = _PromptBuilder("Each row names a real fruit.")
    engine.validate_batches([_single_batch()], builder)

    fake_chat.chat.assert_called_once()
    args, kwargs = fake_chat.chat.call_args
    assert len(args) == 1  # only the prompt — backwards-compatible
    assert kwargs == {"stream": False, "echo": "none"}


def test_engine_validate_single_batch_passes_attachments():
    sentinel = object()
    engine, fake_chat = _make_engine_with_mock_chat([sentinel])

    builder = _PromptBuilder("Each row names a real fruit.")
    engine.validate_single_batch(_single_batch(), builder)

    args, _ = fake_chat.chat.call_args
    assert args[1] is sentinel
