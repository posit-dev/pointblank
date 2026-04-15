## DraftValidation


Draft a validation plan for a given table using an LLM.


Usage

``` python
DraftValidation()
```


By using a large language model (LLM) to draft a validation plan, you can quickly generate a starting point for validating a table. This can be useful when you have a new table and you want to get a sense of how to validate it (and adjustments could always be made later). The [DraftValidation](DraftValidation.md#pointblank.DraftValidation) class uses the `chatlas` package to draft a validation plan for a given table using an LLM from either the `"anthropic"`, `"openai"`, `"ollama"` or `"bedrock"` provider. You can install all requirements for the class through an optional 'generate' install of Pointblank via `pip install pointblank[generate]`.

> **Warning: Warning**
>
> The [DraftValidation](DraftValidation.md#pointblank.DraftValidation) class is still experimental. Please report any issues you encounter in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).


## Parameters


`data: Any`  
The data to be used for drafting a validation plan.

`model: str`  
The model to be used. This should be in the form of `provider:model` (e.g., `"anthropic:claude-opus-4-6"`). Supported providers are `"anthropic"`, `"openai"`, `"ollama"`, and `"bedrock"`.

`api_key: str | None = None`  
The API key to be used for the model.

`verify_ssl: bool = ``True`  
Whether to verify SSL certificates when making requests to the LLM provider. Set to `False` to disable SSL verification (e.g., when behind a corporate firewall with self-signed certificates). Defaults to `True`. Use with caution as disabling SSL verification can pose security risks.


## Returns


`str`  
The drafted validation plan.


## Constructing The `model` Argument

The `model=` argument should be constructed using the provider and model name separated by a colon (`provider:model`). The provider text can any of:

- `"anthropic"` (Anthropic)
- `"openai"` (OpenAI)
- `"ollama"` (Ollama)
- `"bedrock"` (Amazon Bedrock)

The model name should be the specific model to be used from the provider. Model names are subject to change so consult the provider's documentation for the most up-to-date model names.


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

> **Warning: Warning**
>
> Disabling SSL verification (through `verify_ssl=False`) can expose your API keys and data to man-in-the-middle attacks. Only use this option in trusted network environments and when absolutely necessary.


## Notes On Data Sent To The Model Provider

The data sent to the model provider is a JSON summary of the table. This data summary is generated internally by [DraftValidation](DraftValidation.md#pointblank.DraftValidation) using the [DataScan](DataScan.md#pointblank.DataScan) class. The summary includes the following information:

- the number of rows and columns in the table
- the type of dataset (e.g., Polars, DuckDB, Pandas, etc.)
- the column names and their types
- column level statistics such as the number of missing values, min, max, mean, and median, etc.
- a short list of data values in each column

The JSON summary is used to provide the model with the necessary information to draft a validation plan. As such, even very large tables can be used with the [DraftValidation](DraftValidation.md#pointblank.DraftValidation) class since the contents of the table are not sent to the model provider.

The Amazon Bedrock is a special case since it is a self-hosted model and security controls are in place to ensure that data is kept within the user's AWS environment. If using an Ollama model all data is handled locally, though only a few models are capable enough to perform the task of drafting a validation plan.


## Examples

Let's look at how the [DraftValidation](DraftValidation.md#pointblank.DraftValidation) class can be used to draft a validation plan for a table. The table to be used is `"nycflights"`, which is available here via the <a href="load_dataset.html#pointblank.load_dataset" class="gdls-link"><code>load_dataset()</code></a> function. The model to be used is `"anthropic:claude-opus-4-6"` (which performs very well compared to other LLMs). The example assumes that the API key is stored in an `.env` file as `ANTHROPIC_API_KEY`.

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

Note that the output does not know how the data was obtained, so it uses the placeholder `your_data` in the `data=` argument of the [Validate](Validate.md#pointblank.Validate) class. When adapted for use, this should be replaced with the actual data variable.
