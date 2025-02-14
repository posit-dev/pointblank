from __future__ import annotations

from dataclasses import dataclass, field

from narwhals.typing import FrameT
from typing import Any

from pointblank._constants import MODEL_PROVIDERS
from pointblank.datascan import DataScan
from pointblank._utils import _get_api_and_examples_text

__all__ = [
    "DraftValidation",
]


@dataclass
class DraftValidation:
    """
    Draft a validation plan for a given table using an LLM.

    By using a large language model (LLM) to draft a validation plan, you can quickly generate a
    starting point for validating a table. This can be useful when you have a new table and you
    want to get a sense of how to validate it (and adjustments could always be made later). The
    `DraftValidation` class uses the `chatlas` package to draft a validation plan for a given table
    using an LLM from either the `"anthropic"` or `"openai"` provider. You can install all
    requirements for the class by using the optional install of Pointblank with `pip install
    pointblank[generate]`.

    :::{.callout-warning}
    The `DraftValidation()` class is still experimental. Please report any issues you encounter in
    the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).
    :::

    Parameters
    ----------
    data
        The data to be used for drafting a validation plan.
    model
        The model to be used. This should be in the form of `provider:model` (e.g.,
        `"anthropic:claude-3-5-sonnet-latest"`). Supported providers are `"anthropic"` and
        `"openai"`.
    api_key
        The API key to be used for the model.

    Returns
    -------
    str
        The drafted validation plan.
    """

    data: FrameT | Any
    model: str
    api_key: str | None = None
    response: str = field(init=False)

    def __post_init__(self):

        # Check that the chatlas package is installed
        try:
            import chatlas  # noqa
        except ImportError:
            raise ImportError(
                "The `chatlas` package is required to use the `DraftValidation` class. "
                "Please install it using `pip install chatlas`."
            )

        # Generate a table summary in JSON format using the `DataScan` class
        tbl_json = DataScan(data=self.data).to_json()

        # Get the LLM provider from the `model` value
        provider = self.model.split(":")[0]

        # Validate that the provider is supported
        if provider not in MODEL_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers are {MODEL_PROVIDERS}."
            )

        # Generate the API and examples text
        api_and_examples_text = _get_api_and_examples_text()

        # Get the model name from the `model` value
        model_name = self.model.split(":")[1]

        prompt = (
            f"{api_and_examples_text}"
            "--------------------------"
            "Knowing what you now know about the pointblank package in Python, can you write a "
            "thorough validation plan for the following table?"
            "Here is the JSON summary for the table:"
            " "
            "```json"
            f"{tbl_json}"
            "```"
            " "
            "Provide a single piece of Python code for a validation plan that is contained in "
            "```python + ``` code fences. Don't provide any text before or after this block. Use "
            "the same style as in the provided examples."
            " "
            "Additional notes for this task:"
            " "
            "  - structure the code with the following parts: (1) `import pointblank as pb` at the "
            "    top; (2) assign the schema to the variable `schema`, comment above should read "
            "    # Define schema based on column names and dtypes; (3) provide the validation "
            "    plan, comment above should read # The validation plan; (4) end with a single call "
            "    of `validation`"
            "  - the name of the dataset won't be known to you. Simply use the text your_data in "
            "    the `data=` arg of the `Validate` class."
            "  - use the text `Draft Validation` in the `label=` argument of `Validate`"
            "  - don't use an invocation of the `load_dataset()` function anywhere, that is just "
            "    for loading example datasets."
            "  - avoid using the validation method `col_vals_regex()` since you are only provided "
            "    with a very small set of strings."
            "  - do use the `col_schema_match()` validation step along with a `Schema` invocation "
            "    that gets the column names (`column_name` field) and the column types "
            "    (`column_type` field) from the JSON summary for the table."
            "  - when providing column types to `Schema` make sure it is taken *exactly* from the "
            "    JSON summary; so don't use 'Datetime' when the `column_type` is"
            "    'Datetime(time_unit='us', time_zone='UTC')'"
            "  - note that the `col_vals_*()` validation methods do not work with anything other "
            "    than numeric values (except `col_vals_regex()`, which is meant for text)"
            "  - always use the `row_count_match()` and `column_count_match()` validation methods"
            "  - if the number of `missing_values` for a column is `0` (provided in the JSON "
            "    summary) then use the `col_vals_not_null()` validation method (and include all "
            "    such columns) in the `columns=` arg"
            "  - use the `rows_distinct()` validation method but don't supply anything to the "
            "    `columns_subset=` arg"
            "  - use a `col_vals_*()` validation method for a column only once; in other words "
            "    don't have two validation steps where `col_vals_between()` is used for a "
            "    particular column"
            "  - when using the `col_vals_*()` methods that have the `na_pass=` argument, use "
            "    `na_pass=True` when the JSON summary indicates that a column has a non-zero "
            "    number of `missing_values`"
            "  - when providing long lists of columns, use line breaks at `[` and `]`, this is so "
            "    that the generated code isn't too wide (try to adhere to a 80 character limit "
            "    per line)"
        )

        if provider == "anthropic":

            # Check that the anthropic package is installed
            try:
                import anthropic  # noqa
            except ImportError:
                raise ImportError(
                    "The `anthropic` package is required to use the `DraftValidation` class with "
                    "the `anthropic` provider. Please install it using `pip install anthropic`."
                )

            from chatlas import ChatAnthropic

            chat = ChatAnthropic(
                model=model_name,
                system_prompt="You are a terse assistant and a Python expert.",
                api_key=self.api_key,
            )

        if provider == "openai":

            # Check that the openai package is installed
            try:
                import openai  # noqa
            except ImportError:
                raise ImportError(
                    "The `openai` package is required to use the `DraftValidation` class with the "
                    "`openai` provider. Please install it using `pip install openai`."
                )

            from chatlas import ChatOpenAI

            chat = ChatOpenAI(
                model=model_name,
                system_prompt="You are a terse assistant and a Python expert.",
                api_key=self.api_key,
            )

        self.response = str(chat.chat(prompt, stream=False, echo="none"))

    def __str__(self) -> str:
        return self.response

    def __repr__(self) -> str:
        return self.response
