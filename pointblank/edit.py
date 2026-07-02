from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from importlib_resources import files

from pointblank._constants import MODEL_PROVIDERS
from pointblank._utils_ai import (
    _check_syntax,
    _create_chat_instance,
    _diff_plan_steps,
    _extract_chain_steps,
    _extract_code,
)

__all__ = [
    "EditValidation",
]


def _yaml_text_to_code(yaml_text: str) -> str:
    """Convert a YAML validation config (string) to Pointblank Python code."""
    from pointblank.yaml import yaml_to_python

    return _extract_code(yaml_to_python(yaml_text))


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> Any:
    """Build a DataFrame from a list of row dicts, preferring Polars, falling back to Pandas."""
    columns = list(rows[0].keys())
    try:
        import polars as pl

        return pl.DataFrame(rows)
    except ImportError:  # pragma: no cover
        pass
    try:
        import pandas as pd

        return pd.DataFrame(rows, columns=columns)
    except ImportError:  # pragma: no cover
        raise ImportError(
            "`EditValidation.review()` requires either Polars or Pandas to be installed."
        )


def _normalize_to_code(validation: Any) -> str:
    """Normalize any accepted plan input to a canonical Pointblank code string.

    Accepts a `Validate` object, a code string, a YAML string, or a path to a `.py`/`.yaml`
    file. YAML inputs are converted to equivalent Python code so the model always receives the
    plan in a single, consistent format.
    """
    from pointblank.validate import Validate

    if isinstance(validation, Validate):
        return validation.to_code()

    if isinstance(validation, Path):
        return _code_from_path(validation)

    if isinstance(validation, str):
        # Is it a path to an existing file?
        candidate = None
        if len(validation) < 1024 and "\n" not in validation:
            try:
                p = Path(validation)
                if p.exists() and p.is_file():
                    candidate = p
            except OSError:  # pragma: no cover
                candidate = None
        if candidate is not None:
            return _code_from_path(candidate)

        # Not a file path: decide whether the string is YAML or Python code
        stripped = validation.strip()
        looks_like_yaml = stripped.startswith(("tbl:", "steps:")) or (
            "\nsteps:" in stripped and "pb.Validate" not in stripped
        )
        if looks_like_yaml:
            return _yaml_text_to_code(validation)
        return stripped

    raise TypeError(
        "`validation=` must be a Validate object, a code string, a YAML string, or a file path; "
        f"got {type(validation).__name__}."
    )


def _code_from_path(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        return _yaml_text_to_code(text)
    return text.strip()


@dataclass
class EditValidation:
    """
    Edit an existing validation plan with a plain-English instruction using an LLM.

    While [`DraftValidation`](`pointblank.DraftValidation`) generates a validation plan from
    scratch by profiling a table, `EditValidation` takes an *existing* plan plus a natural-language
    instruction (e.g., "add a freshness check on `updated_at`, tighten the email regex, and drop
    the `price > 0` check") and produces a revised plan. You review the change as a diff and
    explicitly [`accept()`](`pointblank.EditValidation.accept`) it before anything runs, keeping a
    human in control.

    `EditValidation` reuses the same provider abstraction as `DraftValidation` (via the `chatlas`
    package), supporting the `"anthropic"`, `"openai"`, `"ollama"`, `"bedrock"`, and
    `"azure-openai"` providers. Install the optional requirements with
    `pip install pointblank[generate]`.

    :::{.callout-warning}
    The `EditValidation` class is experimental. Please report any issues you encounter in the
    [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).
    :::

    Parameters
    ----------
    validation
        The existing validation plan to edit. This can be a [`Validate`](`pointblank.Validate`)
        object, a Python code string (such as the output of
        [`Validate.to_code()`](`pointblank.Validate.to_code`)), a YAML configuration string, or a
        path to a `.py` or `.yaml`/`.yml` file. All inputs are normalized to Pointblank code before
        being sent to the model.
    instruction
        A plain-English description of the changes to make to the plan.
    model
        The model to be used. This should be in the form of `provider:model` (e.g.,
        `"anthropic:claude-opus-4-8"`). Supported providers are `"anthropic"`, `"openai"`,
        `"ollama"`, `"bedrock"`, and `"azure-openai"`.
    data
        Optional data table. When provided, a [`DataScan`](`pointblank.DataScan`) summary of the
        table is included in the prompt so the model can make *informed* edits (e.g., tightening a
        range using observed min/max values). It is also used as the default data source when
        building a [`Validate`](`pointblank.Validate`) via
        [`accept()`](`pointblank.EditValidation.accept`).
    api_key
        The API key to be used for the model.
    verify_ssl
        Whether to verify SSL certificates when making requests to the LLM provider. Defaults to
        `True`. Use with caution as disabling SSL verification can pose security risks.
    max_reprompts
        The maximum number of times to automatically re-prompt the model when the returned code
        fails the syntax/lint check, before surfacing the result to the user. Defaults to `1`.

    Returns
    -------
    EditValidation
        An `EditValidation` object exposing the revised plan through
        [`to_code()`](`pointblank.EditValidation.to_code`),
        [`diff()`](`pointblank.EditValidation.diff`),
        [`changed_steps()`](`pointblank.EditValidation.changed_steps`), and
        [`accept()`](`pointblank.EditValidation.accept`).

    Notes on Data Sent to the Model Provider
    ----------------------------------------
    As with `DraftValidation`, only a JSON summary of the table (generated by `DataScan`) is ever
    sent to the model, never the underlying rows. If `data=` is omitted, no table information is
    sent at all; the model edits the plan using only the plan code and your instruction.

    Examples
    --------
    ```python
    import pointblank as pb

    # An existing plan
    validation = (
        pb.Validate(data=pb.load_dataset("small_table"))
        .col_vals_gt(columns="d", value=100)
        .col_vals_regex(columns="b", pattern="[0-9]-[a-z]{3}")
    )

    # Edit it with a plain-English instruction
    edited = pb.EditValidation(
        validation=validation,
        instruction="Loosen the `d` check to value=50 and add a not-null check on column `a`.",
        model="anthropic:claude-opus-4-8",
        data=pb.load_dataset("small_table"),
    )

    print(edited.diff())        # review the change
    plan = edited.accept()      # a Validate you can .interrogate()
    ```
    """

    validation: Any
    instruction: str
    model: str
    data: Any = None
    api_key: str | None = None
    verify_ssl: bool = True
    max_reprompts: int = 1
    original_code: str = field(init=False)
    response: str = field(init=False)
    edited_code: str = field(init=False)

    def __post_init__(self) -> None:
        # Verify chatlas is available (mirrors DraftValidation)
        try:
            import chatlas  # noqa
        except ImportError:
            raise ImportError(
                "The `chatlas` package is required to use the `EditValidation` class. "
                "Please install it using `pip install chatlas`."
            )

        # Validate the model string and provider up front
        if ":" not in self.model:
            raise ValueError(
                f"`model=` must be in 'provider:model' form (e.g., "
                f"'anthropic:claude-opus-4-8'); got {self.model!r}."
            )
        provider, model_name = self.model.split(sep=":", maxsplit=1)
        if provider not in MODEL_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers are {MODEL_PROVIDERS}."
            )

        # Normalize whatever plan input we were given into canonical Pointblank code
        self.original_code = _normalize_to_code(self.validation)

        # Assemble the edit prompt and run the model (with a syntax-guardrail re-prompt loop)
        prompt = self._build_prompt()
        self._run_edit(provider, model_name, prompt)

    def _build_prompt(self) -> str:
        # Read the bundled API/examples text so the model knows valid methods and signatures
        with files("pointblank.data").joinpath("api-docs.txt").open(encoding="utf-8") as f:
            api_and_examples_text = f.read()

        # Optionally include a DataScan summary for informed edits
        datascan_block = ""
        if self.data is not None:
            from pointblank.datascan import DataScan

            tbl_json = DataScan(data=self.data).to_json()
            datascan_block = (
                "Here is a JSON summary of the table the plan validates. Use it to make "
                "informed edits (e.g., choosing realistic bounds or set members):\n"
                "```json\n"
                f"{tbl_json}\n"
                "```\n"
            )

        return (
            f"{api_and_examples_text}"
            "--------------------------\n"
            "You are editing an EXISTING Pointblank validation plan. Here is the current plan:\n"
            "```python\n"
            f"{self.original_code}\n"
            "```\n"
            f"{datascan_block}"
            "Apply the following change to the plan:\n"
            f"{self.instruction}\n"
            "\n"
            "Return the COMPLETE revised plan as a single piece of Python code inside "
            "```python + ``` code fences. Don't provide any text before or after this block.\n"
            "\n"
            "Rules for this task:\n"
            "  - change only what the instruction implies; preserve all unrelated steps verbatim\n"
            "  - keep the same overall structure and style as the current plan\n"
            "  - keep the `data=your_data` argument and its comment exactly as-is\n"
            "  - only use documented Pointblank validation methods and valid keyword arguments\n"
            "  - do not invoke `load_dataset()`; the data source is represented by `your_data`\n"
        )

    def _run_edit(self, provider: str, model_name: str, prompt: str) -> None:
        chat = _create_chat_instance(
            provider=provider,
            model_name=model_name,
            api_key=self.api_key,
            verify_ssl=self.verify_ssl,
            system_prompt="You are a terse assistant and a Python expert.",
        )

        response = str(chat.chat(prompt, stream=False, echo="none"))
        code = _extract_code(response)

        # Syntax/lint guardrail: re-prompt on failure up to `max_reprompts` times
        attempts = 0
        ok, message = _check_syntax(code)
        while not ok and attempts < self.max_reprompts:
            attempts += 1
            reprompt = (
                f"The revised plan you returned has a problem: {message}. "
                "Return the COMPLETE corrected plan again inside ```python + ``` code fences, "
                "with no text outside the block."
            )
            response = str(chat.chat(reprompt, stream=False, echo="none"))
            code = _extract_code(response)
            ok, message = _check_syntax(code)

        self.response = response
        self.edited_code = code

    @classmethod
    def from_plans(
        cls,
        original: Any,
        revised: Any,
        instruction: str = "Manual plan comparison",
    ) -> "EditValidation":
        """
        Build an `EditValidation` from two known plans, without calling a model.

        This is a lightweight way to compare an original plan against a revised one and review the
        difference with [`diff()`](`pointblank.EditValidation.diff`),
        [`changed_steps()`](`pointblank.EditValidation.changed_steps`), and
        [`review()`](`pointblank.EditValidation.review`). It is useful for reviewing the change
        between two saved versions of a plan (for example, in code review) and does not require an
        LLM provider or any API key.

        Parameters
        ----------
        original
            The original plan. Accepts anything the `validation=` argument accepts: a
            [`Validate`](`pointblank.Validate`) object, a code string, a YAML string, or a file
            path.
        revised
            The revised plan, in any of the same forms as `original`.
        instruction
            An optional label describing the comparison (shown as the subtitle in
            [`review()`](`pointblank.EditValidation.review`)).

        Returns
        -------
        EditValidation
            An `EditValidation` whose `original`/`revised` plans are the ones supplied, ready for
            [`diff()`](`pointblank.EditValidation.diff`),
            [`changed_steps()`](`pointblank.EditValidation.changed_steps`),
            [`review()`](`pointblank.EditValidation.review`), and
            [`accept()`](`pointblank.EditValidation.accept`).

        Examples
        --------
        ```python
        import pointblank as pb

        v1 = pb.Validate(data=pb.load_dataset("small_table")).col_vals_gt(columns="d", value=100)
        v2 = pb.Validate(data=pb.load_dataset("small_table")).col_vals_gt(columns="d", value=50)

        comparison = pb.EditValidation.from_plans(v1, v2)
        print(comparison.diff())
        ```
        """
        # Bypass __init__/__post_init__ (which would call the model) and populate directly
        obj = cls.__new__(cls)
        obj.validation = original
        obj.instruction = instruction
        obj.model = ""
        obj.data = None
        obj.api_key = None
        obj.verify_ssl = True
        obj.max_reprompts = 0
        obj.original_code = _normalize_to_code(original)
        obj.edited_code = _normalize_to_code(revised)
        obj.response = obj.edited_code
        return obj

    def to_code(self) -> str:
        """Return the revised validation plan as Python code."""
        return self.edited_code

    def validate_syntax(self) -> bool:
        """Return whether the revised plan parses and uses only known validation methods."""
        ok, _ = _check_syntax(self.edited_code)
        return ok

    def diff(self) -> str:
        """Return a unified textual diff of the original plan versus the revised plan."""
        original = self.original_code.splitlines(keepends=True)
        edited = self.edited_code.splitlines(keepends=True)
        if original and not original[-1].endswith("\n"):
            original[-1] += "\n"
        if edited and not edited[-1].endswith("\n"):
            edited[-1] += "\n"
        return "".join(
            difflib.unified_diff(
                original,
                edited,
                fromfile="original_plan.py",
                tofile="edited_plan.py",
            )
        )

    def changed_steps(self) -> list[dict[str, Any]]:
        """
        Return a structured, step-level list of the changes between the plans.

        Unlike [`diff()`](`pointblank.EditValidation.diff`), which is a textual diff sensitive to
        formatting, this compares the two plans at the level of validation steps and is robust to
        reformatting. Each entry is a dict with an `"action"` of `"add"`, `"remove"`, or
        `"modify"`, a `"method"` name, and the relevant `"old"`/`"new"` step text.

        Returns
        -------
        list[dict]
            One record per changed step, e.g.
            `[{"action": "modify", "method": "col_vals_gt", "old": "col_vals_gt(columns=\"d\", "
            "value=100)", "new": "col_vals_gt(columns=\"d\", value=50)"}]`.
        """
        old_steps = _extract_chain_steps(self.original_code)
        new_steps = _extract_chain_steps(self.edited_code)
        return _diff_plan_steps(old_steps, new_steps)

    def review(self) -> Any:
        """
        Return a `great_tables` GT object summarizing the step-level changes.

        This renders the added, removed, and modified steps side by side for quick human review
        before accepting the edit. It pairs with [`accept()`](`pointblank.EditValidation.accept`):
        review the change, then accept it.

        Returns
        -------
        GT
            A `great_tables` table with one row per changed step.
        """
        from great_tables import GT, loc, style

        changes = self.changed_steps()

        symbols = {"add": "+ added", "remove": "− removed", "modify": "~ modified"}
        rows = [
            {
                "action": symbols.get(change["action"], change["action"]),
                "method": change["method"],
                "before": change.get("old", ""),
                "after": change.get("new", ""),
            }
            for change in changes
        ]
        if not rows:
            rows = [{"action": "(no changes)", "method": "", "before": "", "after": ""}]

        data = _rows_to_dataframe(rows)

        gt = (
            GT(data)
            .tab_header(title="Proposed plan changes", subtitle=self.instruction)
            .cols_label(action="Change", method="Method", before="Before", after="After")
        )
        # Color-code the change type
        for action_value, color in (
            ("+ added", "#e6f4ea"),
            ("− removed", "#fce8e6"),
            ("~ modified", "#fef7e0"),
        ):
            gt = gt.tab_style(
                style=style.fill(color=color),
                locations=loc.body(
                    rows=[i for i, r in enumerate(rows) if r["action"] == action_value]
                ),
            )
        return gt

    def accept(self, data: Any = None) -> Any:
        """
        Execute the revised plan and return the resulting `Validate` object.

        The returned `Validate` has NOT been interrogated; call `.interrogate()` on it to run the
        validation.

        Parameters
        ----------
        data
            The data table to bind to the plan's `your_data` placeholder. If omitted, the `data=`
            provided to `EditValidation` is used. One of the two must be present.

        Returns
        -------
        Validate
            The revised validation plan as a `Validate` object.
        """
        import pointblank as pb

        table = data if data is not None else self.data
        if table is None:
            raise ValueError(
                "`accept()` requires a data table to build a Validate object. Pass `data=` here "
                "or provide `data=` when constructing EditValidation."
            )

        ok, message = _check_syntax(self.edited_code)
        if not ok:
            raise ValueError(
                f"The revised plan cannot be accepted because it failed the syntax check: "
                f"{message}. Inspect `.to_code()` and edit it manually."
            )

        # Strip any terminal `.interrogate(...)` so the returned Validate is a plan (not yet
        # interrogated), matching this method's contract even if the model added one.
        import re

        code = re.sub(r"\.interrogate\([^()]*\)", "", self.edited_code)

        namespace: dict[str, Any] = {"pb": pb, "your_data": table}
        exec(code, namespace)

        validation = namespace.get("validation")
        if validation is None:
            # Fall back to the last Validate-like object defined by the code
            for value in reversed(list(namespace.values())):
                if isinstance(value, pb.Validate):
                    validation = value
                    break
        if validation is None:
            raise ValueError(
                "The revised plan did not define a `validation` object. Inspect `.to_code()`."
            )
        return validation

    def __str__(self) -> str:
        return self.edited_code

    def __repr__(self) -> str:
        return self.edited_code
