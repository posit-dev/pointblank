# DraftValidation

Draft a validation plan for a given table using an LLM.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/draft.py#L25-L507)

``` python
DraftValidation(
    data,
    model,
    api_key=None,
    verify_ssl=True,
    max_reprompts=1,
    attachments=None,
)
```

By using a large language model (LLM) to draft a validation plan, you can quickly generate a starting point for validating a table. This can be useful when you have a new table and you want to get a sense of how to validate it (and adjustments could always be made later). The [DraftValidation](../reference/DraftValidation.md#pointblank.DraftValidation) class uses the `chatlas` package to draft a validation plan for a given table using an LLM from the `"anthropic"`, `"openai"`, `"ollama"`, `"bedrock"`, `"azure-openai"`, `"cohere"`, or `"databricks"` provider. You can install all requirements for the class through an optional 'generate' install of Pointblank via `pip install pointblank[generate]` (the `"databricks"` provider also needs the `databricks-sdk` package).

> **Warning**
>
> The [DraftValidation](../reference/DraftValidation.md#pointblank.DraftValidation) class is still experimental. Please report any issues you encounter in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).

## Parameters

`data: Any`  
The data to be used for drafting a validation plan.

`model: str`  
The model to be used. This should be in the form of `provider:model` (e.g., `"anthropic:claude-opus-4-6"`). Supported providers are `"anthropic"`, `"openai"`, `"ollama"`, `"bedrock"`, `"azure-openai"`, `"cohere"`, and `"databricks"`. For `"azure-openai"`, the value after the colon is the Azure *deployment id*, not an OpenAI model id, and for `"databricks"` it's the name of a serving endpoint.

`api_key: str | None = None`  
The API key to be used for the model.

`verify_ssl: bool = True`  
Whether to verify SSL certificates when making requests to the LLM provider. Set to `False` to disable SSL verification (e.g., when behind a corporate firewall with self-signed certificates). Defaults to `True`. Use with caution as disabling SSL verification can pose security risks. This isn't supported for the `"databricks"` provider.

`max_reprompts: int = 1`  
The maximum number of times to ask the model for a corrected plan when the one it returned doesn't pass a basic check: the code must parse as Python, and every method that looks like a validation method (a name starting with `col_`, `row_`, or `rows_`) must exist, since models sometimes invent method names. The retry tells the model what the problem was. Each retry is another request to the provider, so a higher value can cost more. Set to `0` to accept the first reply as is, and check it yourself with the object's [validate_syntax()](../reference/pointblank.EditValidation.validate_syntax.md) method. Defaults to `1`.

`attachments: list | None = None`  
An optional list of reference files (images or PDFs) describing what the table should contain, such as a data dictionary, a schema diagram, or a specification. Each item can be a local file path, a URL (`http://` / `https://`), a `pathlib.Path`, or a pre-built chatlas `Content` object. Supported extensions are `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, and `.pdf`. The model uses the expectations it finds in them (ranges, allowed values, required columns, etc.) in the drafted plan, even where the table's summary disagrees. See the *Drafting from Reference Files* section for more.

## Returns

`str`  
The drafted validation plan.

## Constructing The `model` Argument

The `model=` argument should be constructed using the provider and model name separated by a colon (`provider:model`). The provider text can any of:

- `"anthropic"` (Anthropic)
- `"openai"` (OpenAI)
- `"ollama"` (Ollama)
- `"bedrock"` (Amazon Bedrock)
- `"azure-openai"` (Azure OpenAI)
- `"cohere"` (Cohere)
- `"databricks"` (Databricks)

The model name should be the specific model to be used from the provider. Model names are subject to change so consult the provider's documentation for the most up-to-date model names. For `"azure-openai"`, the value after the colon is the Azure *deployment id* (the name you assigned when deploying the model in your Azure OpenAI resource). It also requires the environment variables `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` (e.g., `https://<resource>.openai.azure.com`), and `OPENAI_API_VERSION` (e.g., `"2024-06-01"`) to be set. For `"cohere"`, the API key is read from `CO_API_KEY` (or `COHERE_API_KEY`) if `api_key=` isn't given, and the requests go to Cohere's OpenAI-compatible API.

For `"databricks"`, the value after the colon is the name of a Model Serving endpoint in your workspace (e.g., `"databricks:databricks-claude-sonnet-4-6"`). It needs the `databricks-sdk` package (`pip install databricks-sdk`) and chatlas 0.13.2 or later. The `databricks-sdk` package handles signing in: through the `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables, a profile in `~/.databrickscfg`, or a Databricks notebook's own credentials. If `api_key=` is given, it's used as a personal access token.

For `"openai"`, if the `OPENAI_BASE_URL` environment variable is set, the requests go to the Chat Completions endpoint at that URL instead of to OpenAI. This is a way to use an OpenAI-compatible server or gateway (such as vLLM or LiteLLM), with `OPENAI_API_KEY` (or `api_key=`) as its key.

## Notes On Authentication

Providing a valid API key as a string in the `api_key` argument is adequate for getting started but you should consider using a more secure method for handling API keys.

One way to do this is to load the API key from an environent variable and retrieve it using the `os` module (specifically the `os.getenv()` function). Places to store the API key might include `.bashrc`, `.bash_profile`, `.zshrc`, or `.zsh_profile`.

Another solution is to store one or more model provider API keys in an `.env` file (in the root of your project). If the API keys have correct names (e.g., `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) then DraftValidation will automatically load the API key from the `.env` file and there's no need to provide the `api_key` argument. An `.env` file might look like this:

``` plaintext
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```

There's no need to have the `python-dotenv` package installed when using `.env` files in this way.

## Notes On Ssl Certificate Verification

By default, SSL certificate verification is enabled for all requests to LLM providers. However, in certain network environments (such as corporate networks with self-signed certificates or firewall proxies), you may encounter SSL certificate verification errors.

To disable SSL verification, set the `verify_ssl` parameter to `False`:

``` python
import pointblank as pb

data = pb.load_dataset(dataset="nycflights", tbl_type="duckdb")

# Disable SSL verification for networks with self-signed certificates
pb.DraftValidation(
    data=data,
    model="anthropic:claude-opus-4-6",
    verify_ssl=False
)
```

> **Warning**
>
> Disabling SSL verification (through `verify_ssl=False`) can expose your API keys and data to man-in-the-middle attacks. Only use this option in trusted network environments and when absolutely necessary.

## Notes On Data Sent To The Model Provider

The data sent to the model provider is a JSON summary of the table. This data summary is generated internally by [DraftValidation](../reference/DraftValidation.md#pointblank.DraftValidation) using the [DataScan](../reference/DataScan.md#pointblank.DataScan) class. The summary includes the following information:

- the number of rows and columns in the table
- the type of dataset (e.g., Polars, DuckDB, Pandas, etc.)
- the column names and their types
- column level statistics such as the number of missing values, min, max, mean, and median, etc.
- a short list of data values in each column

The JSON summary is used to provide the model with the necessary information to draft a validation plan. As such, even very large tables can be used with the [DraftValidation](../reference/DraftValidation.md#pointblank.DraftValidation) class since the contents of the table are not sent to the model provider.

Files given in `attachments=` are different: they are sent to the model provider as they are (local images are first downscaled to 512x512), so don't attach anything that the provider shouldn't see.

The Amazon Bedrock is a special case since it is a self-hosted model and security controls are in place to ensure that data is kept within the user's AWS environment. If using an Ollama model all data is handled locally, though only a few models are capable enough to perform the task of drafting a validation plan.

## Drafting From Reference Files

The table's summary only says what the data *is*, so a plan drafted from it alone checks that the data stays the way it is now. If there's a document saying what the data *should be* (a data dictionary, a schema diagram, a specification, or the configuration of a project), it can be given to `attachments=`, and the model will draft the plan from both. Where a reference file gives an expectation, such as a range, a set of allowed values, or a required column, the plan uses it even if the data disagrees, since catching that is the point of the validation. Each step that comes from a reference file gets a comment saying so.

A few things to keep in mind when using reference files:

- the [Schema](../reference/Schema.md#pointblank.Schema) in the plan still takes its column types from the table's summary (they need to be the table's own type names), and a different type in a reference file is noted in a comment; a column that a reference file requires but the table lacks is added to the [Schema](../reference/Schema.md#pointblank.Schema) by name only, so the schema check fails
- not every model accepts images and PDFs (many local models through Ollama accept neither), so choose one that does
- local images given as paths are downscaled to 512x512 (chatlas's `resize="low"`); for a detailed diagram, pass `chatlas.content_image_file(path, resize="high")` instead
- the files are sent with the first request; if the plan has to be corrected (see `max_reprompts=`), the retry sends them again as part of the conversation, so large files add to the cost of each retry

## Examples

Let's look at how the [DraftValidation](../reference/DraftValidation.md#pointblank.DraftValidation) class can be used to draft a validation plan for a table. The table to be used is `"nycflights"`, which is available here via the [`load_dataset()`](../reference/load_dataset.md#pointblank.load_dataset) function. The model to be used is `"anthropic:claude-opus-4-6"` (which performs very well compared to other LLMs). The example assumes that the API key is stored in an `.env` file as `ANTHROPIC_API_KEY`.

``` python
import pointblank as pb

# Load the "nycflights" dataset as a DuckDB table
data = pb.load_dataset(dataset="nycflights", tbl_type="duckdb")

# Draft a validation plan for the "nycflights" table
pb.DraftValidation(data=data, model="anthropic:claude-opus-4-6")
```

The output will be a drafted validation plan for the `"nycflights"` table and this will appear in the console.

```` plaintext
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

The drafted validation plan can be copied and pasted into a Python script or notebook for further use. In other words, the generated plan can be adjusted as needed to suit the specific requirements of the table being validated.

Note that the output does not know how the data was obtained, so it uses the placeholder `your_data` in the `data=` argument of the [Validate](../reference/Validate.md#pointblank.Validate) class. When adapted for use, this should be replaced with the actual data variable.

If there is a document describing what the table should contain, it can be attached so that the plan checks the data against it. Here, a data dictionary for the flights data (a PDF) and a diagram of its schema (an image) are given to `attachments=`:

``` python
pb.DraftValidation(
    data=data,
    model="anthropic:claude-opus-4-6",
    attachments=["flights_data_dictionary.pdf", "flights_schema.png"]
)
```

The drafted plan has the same parts as before, but its steps also use the expectations in the two files (e.g., the allowed values of `carrier` or the range of `distance`), with a comment on each step that comes from them.
