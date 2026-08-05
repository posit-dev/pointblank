# prompt_create_validator


Prompt to create a Pointblank Validator for a loaded DataFrame.


# Arguments


`df_id: string`  
ID of the DataFrame to validate.

Provide as a JSON string matching the following schema: {"type":"string"}

`validator_id: string`  
Optional ID for the Validator. If not provided, a new ID will be generated.

Provide as a JSON string matching the following schema: {"anyOf":\[{"type":"string"},{"type":"null"}\]}

`table_name: string`  
Optional name for the table within Pointblank reports.

Provide as a JSON string matching the following schema: {"anyOf":\[{"type":"string"},{"type":"null"}\]}

`validator_label: string`  
Optional descriptive label for the Validator.

Provide as a JSON string matching the following schema: {"anyOf":\[{"type":"string"},{"type":"null"}\]}

`thresholds_dict_example: string`  
Example thresholds for validation failures.

Provide as a JSON string matching the following schema: {"anyOf":\[{"additionalProperties":{"anyOf":\[{"type":"integer"},{"type":"number"}\]},"type":"object"},{"type":"null"}\]}


# Prompt Text

> **Note: Assistant**
>
> ``` text
> Once your data is loaded (using its `df_id`), I can create a 'Validator' object to define data quality checks.
> ```

> **Note: User message**
>
> ``` text
> Please call `create_validator` using the `df_id` of your loaded data (e.g., 'df_default').
> You can optionally provide:
> - `validator_id` (e.g., 'validator_default') to name this validator instance.
> - `table_name` (e.g., 'data_table') as a reference name for the data table in reports.
> - `validator_label` (e.g., 'Validator') for a descriptive label.
> - `thresholds_dict` (e.g., {'warning': 0.05, 'error': 10}) to set global failure thresholds for validation steps.
> - Other optional parameters like `actions_dict`, `final_actions_dict`, `brief`, `lang`, `locale` can also be specified if needed.
> Make a note of the returned `validator_id` to use when adding validation steps.
> ```
