from __future__ import annotations

from typing import Any

from importlib_resources import files
from narwhals.typing import FrameT

from pointblank._constants import MODEL_PROVIDERS
from pointblank.datascan import DataScan

__all__ = [
    "assistant",
]


def assistant(
    model: str,
    data: FrameT | Any | None = None,
    tbl_name: str | None = None,
    api_key: str | None = None,
    display: str | None = None,
) -> None:
    """
    Chat with the PbA (Pointblank Assistant) about your data validation needs.

    The `assistant()` function provides an interactive chat session with the PbA (Pointblank
    Assistant) to help you with your data validation needs. The PbA can help you with constructing
    validation plans, suggesting validation methods, and providing code snippets for using the
    Pointblank Python package. Feel free to ask the PbA about any aspect of the Pointblank package
    and it will do its best to assist you.

    The PbA can also help you with constructing validation plans for your data tables. If you
    provide a data table to the PbA, it will internally generate a JSON summary of the table and
    use that information to suggest validation methods that can be used with the Pointblank package.
    If using a Polars table as the data source, the PbA will be knowledgeable about the Polars API
    and can smartly suggest validation steps that use aggregate measures with up-to-date Polars
    methods.

    The PbA can be used with models from the following providers:

    - Anthropic
    - OpenAI
    - Ollama
    - Amazon Bedrock

    The PbA can be displayed in a browser (the default) or in the terminal. You can choose one or
    the other by setting the `display=` parameter to `"browser"` or `"terminal"`.

    :::{.callout-warning}
    The `assistant()` function is still experimental. Please report any issues you encounter in
    the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).
    :::

    Parameters
    ----------
    model
        The model to be used. This should be in the form of `provider:model` (e.g.,
        `"anthropic:claude-3-5-sonnet-latest"`). Supported providers are `"anthropic"`, `"openai"`,
        `"ollama"`, and `"bedrock"`.
    data
        An optional data table to focus on during discussion with the PbA, which could be a
        DataFrame object, an Ibis table object, a CSV file path, a Parquet file path, or a database
        connection string. Read the *Supported Input Table Types* section for details on the
        supported table types.
    tbl_name : str, optional
        The name of the data table. This is optional and is only used to provide a more detailed
        prompt to the PbA.
    api_key : str, optional
        The API key to be used for the model.
    display : str, optional
        The display mode to use for the chat session. Supported values are `"browser"` and
        `"terminal"`. If not provided, the default value is `"browser"`.

    Returns
    -------
    None
        Nothing is returned. Rather, you get an an interactive chat session with the PbA, which is
        displayed in a browser or in the terminal.

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

    Notes on Data Sent to the Model Provider
    ----------------------------------------
    If `data=` is provided then that data is sent to the model provider is a JSON summary of the
    table. This data summary is generated internally by use of the `DataScan` class. The summary
    includes the following information:

    - the number of rows and columns in the table
    - the type of dataset (e.g., Polars, DuckDB, Pandas, etc.)
    - the column names and their types
    - column level statistics such as the number of missing values, min, max, mean, and median, etc.
    - a short list of data values in each column

    The JSON summary is used to provide the model with the necessary information be knowledgable
    about the data table. Compared to the size of the entire table, the JSON summary is quite small
    and can be safely sent to the model provider.

    The Amazon Bedrock provider is a special case since it is a self-hosted model and security
    controls are in place to ensure that data is kept within the user's AWS environment. If using an
    Ollama model all data is handled locally.

    Supported Input Table Types
    ---------------------------
    The `data=` parameter can be given any of the following table types:

    - Polars DataFrame (`"polars"`)
    - Pandas DataFrame (`"pandas"`)
    - PySpark table (`"pyspark"`)
    - DuckDB table (`"duckdb"`)*
    - MySQL table (`"mysql"`)*
    - PostgreSQL table (`"postgresql"`)*
    - SQLite table (`"sqlite"`)*
    - Microsoft SQL Server table (`"mssql"`)*
    - Snowflake table (`"snowflake"`)*
    - Databricks table (`"databricks"`)*
    - BigQuery table (`"bigquery"`)*
    - Parquet table (`"parquet"`)*
    - CSV files (string path or `pathlib.Path` object with `.csv` extension)
    - Parquet files (string path, `pathlib.Path` object, glob pattern, directory with `.parquet`
    extension, or partitioned dataset)
    - Database connection strings (URI format with optional table specification)

    The table types marked with an asterisk need to be prepared as Ibis tables (with type of
    `ibis.expr.types.relations.Table`). Furthermore, using `assistant()` with these types of tables
    requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a Polars or
    Pandas DataFrame, the availability of Ibis is not needed.

    To use a CSV file, ensure that a string or `pathlib.Path` object with a `.csv` extension is
    provided. The file will be automatically detected and loaded using the best available DataFrame
    library. The loading preference is Polars first, then Pandas as a fallback.
    """

    # Check that the chatlas package is installed
    try:
        import chatlas  # noqa
    except ImportError:
        raise ImportError(
            "The `chatlas` package is required to use the `Assistant` class. "
            "Please install it using `pip install chatlas`."
        )

    # Check that the shiny package is installed
    try:
        import shiny  # noqa
    except ImportError:
        raise ImportError(
            "The `shiny` package is required to use the `Assistant` class. "
            "Please install it using `pip install shiny`."
        )

    if display is None:
        display = "browser"

    prompt_tbl = ""

    # If a dataset is provided, generate a table summary in JSON format
    if data is not None:
        # Import processing functions from validate module
        from pointblank.validate import _process_data

        # Process input data to handle different data source types
        data = _process_data(data)

        scan = DataScan(data=data)

        tbl_type: str = scan.profile.implementation.name.lower()
        tbl_json = scan.to_json()

        if tbl_name is not None:
            prompt_tbl_1 = (  # pragma: no cover
                f"The user has supplied a {tbl_type} data table named {tbl_name} to chat about."
            )

        else:
            prompt_tbl_1 = f"The user has supplied a {tbl_type} data table to chat about."

        prompt_tbl = f"{prompt_tbl_1}Here is the JSON summary for the table: ```json{tbl_json}``` "
    else:
        from pointblank.validate import load_dataset  # pragma: no cover

        game_revenue = load_dataset(dataset="game_revenue", tbl_type="polars")  # pragma: no cover
        scan_game_revenue = DataScan(data=game_revenue).to_json()  # pragma: no cover

        nycflights = load_dataset(dataset="nycflights", tbl_type="polars")  # pragma: no cover
        scan_nycflights = DataScan(data=nycflights).to_json()  # pragma: no cover

        prompt_tbl = (  # pragma: no cover
            "The user has not supplied a data table to chat about but you can still draw on some "
            "examples from the API documentation to help the user. Here are JSON summaries for "
            "two example tables: "
            "`game_revenue` "
            f"```json{scan_game_revenue}``` "
            "`nycflights` "
            f"```json{scan_nycflights}``` "
        )

    # Read the API/examples text from a file
    with files("pointblank.data").joinpath("api-docs.txt").open() as f:  # pragma: no cover
        api_and_examples_text = f.read()

    with files("pointblank.data").joinpath("polars-api-docs.txt").open() as f:  # pragma: no cover
        polars_api_text = f.read()

    # Get the LLM provider from the `model` value
    provider = model.split(sep=":", maxsplit=1)[0]

    # Validate that the provider is supported
    if provider not in MODEL_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers are {MODEL_PROVIDERS}."
        )

    # Get the model name from the `model` value
    model_name = model.split(sep=":", maxsplit=1)[1]  # pragma: no cover

    system_prompt = (  # pragma: no cover
        "You are a helper to assist in using the Pointblank Python package for data validation."
        "Use the documentation from the package below to help. In your responses, be direct "
        "and do not be excessively verbose or just write large or entire parts of the "
        "Pointblank documentation."
        " "
        f"{api_and_examples_text}"
        "--------------------------"
        "Here is the Polars API in minified form (it will be useful since this API text is pulled "
        "from the latest stable release of Polars):"
        " "
        f"{polars_api_text}"
        "--------------------------"
        f"{prompt_tbl}"
        " "
        "When provides pieces of code that the user can use, ensure that is contained in "
        "```python + ``` code fences. "
        " "
        "Additional notes for helping with developing validation plans:"
        " "
        "  - some validations might be more complex than what is possible with the suite of "
        "    validation methods. In such cases, feel free to modify the table (if it is"
        "    provided) in a pre-processing step (with the `pre=` argument) to make the"
        "    validation step possible; if a table is not provided, then you can at least speak"
        "    to how the table should be modified to make the validation step possible"
        "  - if suggesting a validation that does a schema check, make sure to use the `Schema`"
        "    class and provide the column names and types from the JSON summary"
        "  - the name of the dataset might not be known to you (unless it's provided in the "
        "    JSON summary). If that's the case simply use the text your_data when referring to"
        "    the dataset in the code."
        "  - when providing column types to `Schema` make sure it is taken *exactly* from the "
        "    JSON summary; so don't use 'Datetime' when the `column_type` is"
        "    'Datetime(time_unit='us', time_zone='UTC')'"
        "  - note that the `col_vals_*()` validation methods do not work with anything other "
        "    than numeric values (except `col_vals_regex()`, which is meant for text)"
        "  - if the number of `missing_values` for a column is `0` (provided in the JSON "
        "    summary) then suggest using the `col_vals_not_null()` validation method (and "
        "    include all such columns) in the `columns=` arg"
        "  - when using the `col_vals_*()` methods that have the `na_pass=` argument, use "
        "    `na_pass=True` when the JSON summary indicates that a column has a non-zero "
        "    number of `missing_values`"
        "  - when providing long lists of columns, use line breaks at `[` and `]`, this is so "
        "    that the generated code isn't too wide (try to adhere to a 80 character limit "
        "    per line)"
    )

    if provider == "anthropic":  # pragma: no cover
        # Check that the anthropic package is installed
        try:
            import anthropic  # noqa
        except ImportError:  # pragma: no cover
            raise ImportError(  # pragma: no cover
                "The `anthropic` package is required to use the `DraftValidation` class with "
                "the `anthropic` provider. Please install it using `pip install anthropic`."
            )

        from chatlas import ChatAnthropic  # pragma: no cover

        chat = ChatAnthropic(  # pragma: no cover
            model=model_name,
            system_prompt=system_prompt,
            api_key=api_key,
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
            system_prompt=system_prompt,
            api_key=api_key,
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
            system_prompt=system_prompt,
        )

    if provider == "bedrock":  # pragma: no cover
        from chatlas import ChatBedrockAnthropic  # pragma: no cover

        chat = ChatBedrockAnthropic(  # pragma: no cover
            model=model_name,
            system_prompt=system_prompt,
        )

    if display == "browser":  # pragma: no cover
        return chat.app()  # pragma: no cover
    elif display == "terminal":  # pragma: no cover
        return chat.console()  # pragma: no cover
