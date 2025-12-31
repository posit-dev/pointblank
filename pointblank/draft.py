from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from importlib_resources import files

from pointblank._constants import MODEL_PROVIDERS
from pointblank.datascan import DataScan

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
    using an LLM from either the `"anthropic"`, `"openai"`, `"ollama"` or `"bedrock"` provider. You
    can install all requirements for the class through an optional 'generate' install of Pointblank
    via `pip install pointblank[generate]`.

    :::{.callout-warning}
    The `DraftValidation` class is still experimental. Please report any issues you encounter in
    the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).
    :::

    Parameters
    ----------
    data
        The data to be used for drafting a validation plan.
    model
        The model to be used. This should be in the form of `provider:model` (e.g.,
        `"anthropic:claude-sonnet-4-5"`). Supported providers are `"anthropic"`, `"openai"`,
        `"ollama"`, and `"bedrock"`.
    api_key
        The API key to be used for the model.
    verify_ssl
        Whether to verify SSL certificates when making requests to the LLM provider. Set to `False`
        to disable SSL verification (e.g., when behind a corporate firewall with self-signed
        certificates). Defaults to `True`. Use with caution as disabling SSL verification can pose
        security risks.

    Returns
    -------
    str
        The drafted validation plan.

    Constructing the `model` Argument
    ---------------------------------
    The `model=` argument should be constructed using the provider and model name separated by a
    colon (`provider:model`). The provider text can any of:

    - `"anthropic"` (Anthropic)
    - `"openai"` (OpenAI)
    - `"ollama"` (Ollama)
    - `"bedrock"` (Amazon Bedrock)

    The model name should be the specific model to be used from the provider. Model names are
    subject to change so consult the provider's documentation for the most up-to-date model names.

    Notes on Authentication
    -----------------------
    Providing a valid API key as a string in the `api_key` argument is adequate for getting started
    but you should consider using a more secure method for handling API keys.

    One way to do this is to load the API key from an environent variable and retrieve it using the
    `os` module (specifically the `os.getenv()` function). Places to store the API key might
    include `.bashrc`, `.bash_profile`, `.zshrc`, or `.zsh_profile`.

    Another solution is to store one or more model provider API keys in an `.env` file (in the root
    of your project). If the API keys have correct names (e.g., `ANTHROPIC_API_KEY` or
    `OPENAI_API_KEY`) then DraftValidation will automatically load the API key from the `.env` file
    and there's no need to provide the `api_key` argument. An `.env` file might look like this:

    ```plaintext
    ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

    There's no need to have the `python-dotenv` package installed when using `.env` files in this
    way.

    Notes on SSL Certificate Verification
    --------------------------------------
    By default, SSL certificate verification is enabled for all requests to LLM providers. However,
    in certain network environments (such as corporate networks with self-signed certificates or
    firewall proxies), you may encounter SSL certificate verification errors.

    To disable SSL verification, set the `verify_ssl` parameter to `False`:

    ```python
    import pointblank as pb

    data = pb.load_dataset(dataset="nycflights", tbl_type="duckdb")

    # Disable SSL verification for networks with self-signed certificates
    pb.DraftValidation(
        data=data,
        model="anthropic:claude-sonnet-4-5",
        verify_ssl=False
    )
    ```

    :::{.callout-warning}
    Disabling SSL verification (through `verify_ssl=False`) can expose your API keys and data to
    man-in-the-middle attacks. Only use this option in trusted network environments and when
    absolutely necessary.
    :::

    Notes on Data Sent to the Model Provider
    ----------------------------------------
    The data sent to the model provider is a JSON summary of the table. This data summary is
    generated internally by `DraftValidation` using the `DataScan` class. The summary includes the
    following information:

    - the number of rows and columns in the table
    - the type of dataset (e.g., Polars, DuckDB, Pandas, etc.)
    - the column names and their types
    - column level statistics such as the number of missing values, min, max, mean, and median, etc.
    - a short list of data values in each column

    The JSON summary is used to provide the model with the necessary information to draft a
    validation plan. As such, even very large tables can be used with the `DraftValidation` class
    since the contents of the table are not sent to the model provider.

    The Amazon Bedrock is a special case since it is a self-hosted model and security controls are
    in place to ensure that data is kept within the user's AWS environment. If using an Ollama
    model all data is handled locally, though only a few models are capable enough to perform the
    task of drafting a validation plan.

    Examples
    --------
    Let's look at how the `DraftValidation` class can be used to draft a validation plan for a
    table. The table to be used is `"nycflights"`, which is available here via the
    [`load_dataset()`](`pointblank.load_dataset`) function. The model to be used is
    `"anthropic:claude-sonnet-4-5"` (which performs very well compared to other LLMs). The
    example assumes that the API key is stored in an `.env` file as `ANTHROPIC_API_KEY`.

    ```python
    import pointblank as pb

    # Load the "nycflights" dataset as a DuckDB table
    data = pb.load_dataset(dataset="nycflights", tbl_type="duckdb")

    # Draft a validation plan for the "nycflights" table
    pb.DraftValidation(data=data, model="anthropic:claude-sonnet-4-5")
    ```

    The output will be a drafted validation plan for the `"nycflights"` table and this will appear
    in the console.

    ````plaintext
    ```python
    import pointblank as pb

    # Define schema based on column names and dtypes
    schema = pb.Schema(columns=[
        ("year", "int64"),
        ("month", "int64"),
        ("day", "int64"),
        ("dep_time", "int64"),
        ("sched_dep_time", "int64"),
        ("dep_delay", "int64"),
        ("arr_time", "int64"),
        ("sched_arr_time", "int64"),
        ("arr_delay", "int64"),
        ("carrier", "string"),
        ("flight", "int64"),
        ("tailnum", "string"),
        ("origin", "string"),
        ("dest", "string"),
        ("air_time", "int64"),
        ("distance", "int64"),
        ("hour", "int64"),
        ("minute", "int64")
    ])

    # The validation plan
    validation = (
        pb.Validate(
            data=your_data,  # Replace your_data with the actual data variable
            label="Draft Validation",
            thresholds=pb.Thresholds(warning=0.10, error=0.25, critical=0.35)
        )
        .col_schema_match(schema=schema)
        .col_vals_not_null(columns=[
            "year", "month", "day", "sched_dep_time", "carrier", "flight",
            "origin", "dest", "distance", "hour", "minute"
        ])
        .col_vals_between(columns="month", left=1, right=12)
        .col_vals_between(columns="day", left=1, right=31)
        .col_vals_between(columns="sched_dep_time", left=106, right=2359)
        .col_vals_between(columns="dep_delay", left=-43, right=1301, na_pass=True)
        .col_vals_between(columns="air_time", left=20, right=695, na_pass=True)
        .col_vals_between(columns="distance", left=17, right=4983)
        .col_vals_between(columns="hour", left=1, right=23)
        .col_vals_between(columns="minute", left=0, right=59)
        .col_vals_in_set(columns="origin", set=["EWR", "LGA", "JFK"])
        .col_count_match(count=18)
        .row_count_match(count=336776)
        .rows_distinct()
        .interrogate()
    )

    validation
    ```
    ````

    The drafted validation plan can be copied and pasted into a Python script or notebook for
    further use. In other words, the generated plan can be adjusted as needed to suit the specific
    requirements of the table being validated.

    Note that the output does not know how the data was obtained, so it uses the placeholder
    `your_data` in the `data=` argument of the `Validate` class. When adapted for use, this should
    be replaced with the actual data variable.
    """

    data: Any
    model: str
    api_key: str | None = None
    verify_ssl: bool = True
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
        provider = self.model.split(sep=":", maxsplit=1)[0]

        # Validate that the provider is supported
        if provider not in MODEL_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers are {MODEL_PROVIDERS}."
            )

        # Read the API/examples text from a file
        with files("pointblank.data").joinpath("api-docs.txt").open() as f:  # pragma: no cover
            api_and_examples_text = f.read()

        # Get the model name from the `model` value
        model_name = self.model.split(sep=":", maxsplit=1)[1]  # pragma: no cover

        prompt = (  # pragma: no cover
            f"{api_and_examples_text}"
            "--------------------------"
            "Knowing what you now know about the Pointblank package in Python, can you write a "
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
            "    the `data=` arg of the `Validate` class, and, add the comment to the right of the "
            "    code line # Replace your_data with the actual data variable"
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

        # Create httpx client with SSL verification settings
        # This will be passed to the LLM provider's chat client
        try:
            import httpx  # noqa
        except ImportError:  # pragma: no cover
            raise ImportError(  # pragma: no cover
                "The `httpx` package is required for SSL configuration. "
                "Please install it using `pip install httpx`."
            )

        http_client = httpx.AsyncClient(verify=self.verify_ssl)

        if provider == "anthropic":  # pragma: no cover
            # Check that the anthropic package is installed
            try:
                import anthropic  # noqa  # type: ignore[import-not-found]
            except ImportError:  # pragma: no cover
                raise ImportError(  # pragma: no cover
                    "The `anthropic` package is required to use the `DraftValidation` class with "
                    "the `anthropic` provider. Please install it using `pip install anthropic`."
                )

            from chatlas import ChatAnthropic  # pragma: no cover

            chat = ChatAnthropic(  # pragma: no cover
                model=model_name,
                system_prompt="You are a terse assistant and a Python expert.",
                api_key=self.api_key,
                kwargs={"http_client": http_client},
            )

        if provider == "openai":  # pragma: no cover
            # Check that the openai package is installed
            try:
                import openai  # noqa
            except ImportError:  # pragma: no cover
                raise ImportError(  # pragma: no cover
                    "The `openai` package is required to use the `DraftValidation` class with the "
                    "`openai` provider. Please install it using `pip install openai`."
                )

            from chatlas import ChatOpenAI  # pragma: no cover

            chat = ChatOpenAI(  # pragma: no cover
                model=model_name,
                system_prompt="You are a terse assistant and a Python expert.",
                api_key=self.api_key,
                kwargs={"http_client": http_client},
            )

        if provider == "ollama":  # pragma: no cover
            # Check that the openai package is installed
            try:
                import openai  # noqa
            except ImportError:  # pragma: no cover
                raise ImportError(  # pragma: no cover
                    "The `openai` package is required to use the `DraftValidation` class with "
                    "`ollama`. Please install it using `pip install openai`."
                )

            from chatlas import ChatOllama

            chat = ChatOllama(  # pragma: no cover
                model=model_name,
                system_prompt="You are a terse assistant and a Python expert.",
                kwargs={"http_client": http_client},
            )

        if provider == "bedrock":  # pragma: no cover
            from chatlas import ChatBedrockAnthropic  # pragma: no cover

            chat = ChatBedrockAnthropic(  # pragma: no cover
                model=model_name,
                system_prompt="You are a terse assistant and a Python expert.",
                kwargs={"http_client": http_client},
            )

        self.response = str(chat.chat(prompt, stream=False, echo="none"))  # pragma: no cover

    def __str__(self) -> str:
        return self.response  # pragma: no cover

    def __repr__(self) -> str:
        return self.response  # pragma: no cover
