# prompt_get_validation_step_output


Prompt to get validation output by specifying either a step index or a sundered type.


# Arguments


`validator_id: string`  
Example ID of the Validator.

Provide as a JSON string matching the following schema: {"type":"string"}

`step_index: string`  
Example step index for the first mode of operation.

Provide as a JSON string matching the following schema: {"anyOf":\[{"type":"integer"},{"type":"null"}\]}

`sundered_type: string`  
Example sundered type ('pass' or 'fail') for the second mode of operation.

Provide as a JSON string matching the following schema: {"anyOf":\[{"type":"string"},{"type":"null"}\]}


# Prompt Text

> **Note: Assistant**
>
> ``` text
> I can extract validation data in two different ways. You must choose one: either get data for a *specific step* by its index, or get *all passed or failed rows* from the entire validation run.
> ```

> **Note: User message**
>
> ``` text
> Please call the `get_validation_step_output` tool using only **one** of the following mutually exclusive options:
>
> **OPTION 1: Get data for a specific step**
> To get the data extract for step number 0, use the `step_index` parameter. For example:
> `get_validation_step_output(validator_id='validator_123', step_index=0, output_path='step_0_data.csv')`
>
> **OPTION 2: Get all passed or failed data**
> To get all rows that 'fail' across all validation steps, use the `sundered_type` parameter. For example:
> `get_validation_step_output(validator_id='validator_123', sundered_type='fail', output_path='all_fail_rows.csv')`
> ```
