## Validate.prompt()


Validate rows using AI/LLM-powered analysis.


Usage

``` python
Validate.prompt(
    prompt,
    model,
    columns_subset=None,
    batch_size=1000,
    max_concurrent=3,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True
)
```


The [prompt()](Validate.prompt.md#pointblank.Validate.prompt) validation method uses Large Language Models (LLMs) to validate rows of data based on natural language criteria. Similar to other Pointblank validation methods, this generates binary test results (pass/fail) that integrate seamlessly with the standard reporting framework.

Like `col_vals_*()` methods, [prompt()](Validate.prompt.md#pointblank.Validate.prompt) evaluates data against specific criteria, but instead of using programmatic rules, it uses natural language prompts interpreted by an LLM. Like [rows_distinct()](Validate.rows_distinct.md#pointblank.Validate.rows_distinct) and [rows_complete()](Validate.rows_complete.md#pointblank.Validate.rows_complete), it operates at the row level and allows you to specify a subset of columns for evaluation using `columns_subset=`.

The system automatically combines your validation criteria from the `prompt=` parameter with the necessary technical context, data formatting instructions, and response structure requirements. This is all so you only need to focus on describing your validation logic in plain language.

Each row becomes a test unit that either passes or fails the validation criteria, producing the familiar True/False results that appear in Pointblank validation reports. This method is particularly useful for complex validation rules that are difficult to express with traditional validation methods, such as semantic checks, context-dependent validation, or subjective quality assessments.


## Parameters


`prompt: str`  
A natural language description of the validation criteria. This prompt should clearly describe what constitutes valid vs invalid rows. Some examples: `"Each row should contain a valid email address and a realistic person name"`, `"Values should indicate positive sentiment"`, `"The description should mention a country name"`.

`columns_subset: str | list[str] | None = None`  
A single column or list of columns to include in the validation. If `None`, all columns will be included. Specifying fewer columns can improve performance and reduce API costs so try to include only the columns necessary for the validation.

`model: str`  
The model to be used. This should be in the form of `provider:model` (e.g., `"anthropic:claude-sonnet-4-5"`). Supported providers are `"anthropic"`, `"openai"`, `"ollama"`, and `"bedrock"`. The model name should be the specific model to be used from the provider. Model names are subject to change so consult the provider's documentation for the most up-to-date model names.

`batch_size: int = ``1000`  
Number of rows to process in each batch. Larger batches are more efficient but may hit API limits. Default is `1000`.

`max_concurrent: int = ``3`  
Maximum number of concurrent API requests. Higher values speed up processing but may hit rate limits. Default is `3`.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment). Can be a single column name, a tuple that specifies a column name and its corresponding values to segment on, or a combination of both (provided as a list).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the <a href="Actions.html#pointblank.Actions" class="gdls-link"><code>Actions</code></a> class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = ``True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like <a href="has_columns.html#pointblank.has_columns" class="gdls-link"><code>has_columns()</code></a> and <a href="has_rows.html#pointblank.has_rows" class="gdls-link"><code>has_rows()</code></a> can be used here to conditionally activate a step based on properties of the target table.


## Returns


`Validate`  
The [Validate](Validate.md#pointblank.Validate) object with the added validation step.


## Constructing The `model` Argument

The `model=` argument should be constructed using the provider and model name separated by a colon (`provider:model`). The provider text can any of:

- `"anthropic"` (Anthropic)
- `"openai"` (OpenAI)
- `"ollama"` (Ollama)
- `"bedrock"` (Amazon Bedrock)

The model name should be the specific model to be used from the provider. Model names are subject to change so consult the provider's documentation for the most up-to-date model names.


## Notes On Authentication

API keys are automatically loaded from environment variables or `.env` files and are **not** stored in the validation object for security reasons. You should consider using a secure method for handling API keys.

One way to do this is to load the API key from an environment variable and retrieve it using the `os` module (specifically the `os.getenv()` function). Places to store the API key might include `.bashrc`, `.bash_profile`, `.zshrc`, or `.zsh_profile`.

Another solution is to store one or more model provider API keys in an `.env` file (in the root of your project). If the API keys have correct names (e.g., `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) then the AI validation will automatically load the API key from the `.env` file. An `.env` file might look like this:

``` plaintext
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```

There's no need to have the `python-dotenv` package installed when using `.env` files in this way.

**Provider-specific setup**:

- **OpenAI**: set `OPENAI_API_KEY` environment variable or create `.env` file
- **Anthropic**: set `ANTHROPIC_API_KEY` environment variable or create `.env` file
- **Ollama**: no API key required, just ensure Ollama is running locally
- **Bedrock**: configure AWS credentials through standard AWS methods


## Ai Validation Process

The AI validation process works as follows:

1.  data batching: the data is split into batches of the specified size
2.  row deduplication: duplicate rows (based on selected columns) are identified and only unique combinations are sent to the LLM for analysis
3.  json conversion: each batch of unique rows is converted to JSON format for the LLM
4.  prompt construction: the user prompt is embedded in a structured system prompt
5.  llm processing: each batch is sent to the LLM for analysis
6.  response parsing: LLM responses are parsed to extract validation results
7.  result projection: results are mapped back to all original rows using row signatures
8.  result aggregation: results from all batches are combined

**Performance Optimization**: the process uses row signature memoization to avoid redundant LLM calls. When multiple rows have identical values in the selected columns, only one representative row is validated, and the result is applied to all matching rows. This can dramatically reduce API costs and processing time for datasets with repetitive patterns.

The LLM receives data in this JSON format:

``` json
{
  "columns": ["col1", "col2", "col3"],
  "rows": [
    {"col1": "value1", "col2": "value2", "col3": "value3", "_pb_row_index": 0},
    {"col1": "value4", "col2": "value5", "col3": "value6", "_pb_row_index": 1}
  ]
}
```

The LLM returns validation results in this format:

``` json
[
  {"index": 0, "result": true},
  {"index": 1, "result": false}
]
```


## Prompt Design Tips

For best results, design prompts that are:

- boolean-oriented: frame validation criteria to elicit clear valid/invalid responses
- specific: clearly define what makes a row valid/invalid
- unambiguous: avoid subjective language that could be interpreted differently
- context-aware: include relevant business rules or domain knowledge
- example-driven: consider providing examples in the prompt when helpful

**Critical**: Prompts must be designed so the LLM can determine whether each row passes or fails the validation criteria. The system expects binary validation responses, so avoid open-ended questions or prompts that might generate explanatory text instead of clear pass/fail judgments.

Good prompt examples:

- "Each row should contain a valid email address in the 'email' column and a non-empty name in the 'name' column"
- "The 'sentiment' column should contain positive sentiment words (happy, good, excellent, etc.)"
- "Product descriptions should mention at least one technical specification"

Poor prompt examples (avoid these):

- "What do you think about this data?" (too open-ended)
- "Describe the quality of each row" (asks for description, not validation)
- "How would you improve this data?" (asks for suggestions, not pass/fail)


## Performance Considerations

AI validation is significantly slower than traditional validation methods due to API calls to LLM providers. However, performance varies dramatically based on data characteristics:

**High Memoization Scenarios** (seconds to minutes):

- data with many duplicate rows in the selected columns
- low cardinality data (repeated patterns)
- small number of unique row combinations

**Low Memoization Scenarios** (minutes to hours):

- high cardinality data with mostly unique rows
- large datasets with few repeated patterns
- all or most rows requiring individual LLM evaluation

The row signature memoization optimization can reduce processing time significantly when data has repetitive patterns. For datasets where every row is unique, expect longer processing times similar to validating each row individually.

**Strategies to Reduce Processing Time**:

- test on data slices: define a sampling function like `def sample_1000(df): return df.head(1000)` and use `pre=sample_1000` to validate on smaller samples
- filter relevant data: define filter functions like `def active_only(df): return df.filter(df["status"] == "active")` and use `pre=active_only` to focus on a specific subset
- optimize column selection: use `columns_subset=` to include only the columns necessary for validation
- start with smaller batches: begin with `batch_size=100` for testing, then increase gradually
- reduce concurrency: lower `max_concurrent=1` if hitting rate limits
- use faster/cheaper models: consider using smaller or more efficient models for initial testing before switching to more capable models


## Examples

The following examples demonstrate how to use AI validation for different types of data quality checks. These examples show both basic usage and more advanced configurations with custom thresholds and actions.

**Basic AI validation example:**

This first example shows a simple validation scenario where we want to check that customer records have both valid email addresses and non-empty names. Notice how we use `columns_subset=` to focus only on the relevant columns, which improves both performance and cost-effectiveness.

``` python
import pointblank as pb
import polars as pl

# Sample data with email and name columns
tbl = pl.DataFrame({
    "email": ["john@example.com", "invalid-email", "jane@test.org"],
    "name": ["John Doe", "", "Jane Smith"],
    "age": [25, 30, 35]
})

# Validate using AI
validation = (
    pb.Validate(data=tbl)
    .prompt(
        prompt="Each row should have a valid email address and a non-empty name",
        columns_subset=["email", "name"],  # Only check these columns
        model="openai:gpt-4o-mini",
    )
    .interrogate()
)

validation
```

In this example, the AI will identify that the second row fails validation because it has an invalid email format (`"invalid-email"`) and the third row also fails because it has an empty name field. The validation results will show 2 out of 3 rows failing the criteria.

**Advanced example with custom thresholds:**

This more sophisticated example demonstrates how to use AI validation with custom thresholds and actions. Here we're validating phone number formats to ensure they include area codes, which is a common data quality requirement for customer contact information.

``` python
customer_data = pl.DataFrame({
    "customer_id": [1, 2, 3, 4, 5],
    "name": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Davis"],
    "phone_number": [
        "(555) 123-4567",  # Valid with area code
        "555-987-6543",    # Valid with area code
        "123-4567",        # Missing area code
        "(800) 555-1234",  # Valid with area code
        "987-6543"         # Missing area code
    ]
})

validation = (
    pb.Validate(data=customer_data)
    .prompt(
        prompt="Do all the phone numbers include an area code?",
        columns_subset="phone_number",  # Only check the `phone_number` column
        model="openai:gpt-4o",
        batch_size=500,
        max_concurrent=5,
        thresholds=pb.Thresholds(warning=0.1, error=0.2, critical=0.3),
        actions=pb.Actions(error="Too many phone numbers missing area codes.")
    )
    .interrogate()
)
```

This validation will identify that 2 out of 5 phone numbers (40%) are missing area codes, which exceeds all threshold levels. The validation will trigger the specified error action since the failure rate (40%) is above the error threshold (20%). The AI can recognize various phone number formats and determine whether they include area codes.
