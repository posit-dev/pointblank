# prompt_interrogate_validator


Prompt to run validations and generate reports with Python code.


## Arguments


`validator_id: string`  
ID of the Validator to interrogate.

Provide a value matching the following JSON schema: {"type":"string"}. Encode non-string values as JSON.


## Prompt Text

> **Note: Assistant**
>
> ``` text
> After all desired validation steps have been added to a validator, I can run the interrogation process. This will execute all checks and generate comprehensive reports.
> ```

> **Note: User message**
>
> ``` text
> Please call `interrogate_validator` with the `validator_id` (e.g., 'validator_123').
> This will:
> • Execute all validation checks and return a JSON summary
> • Generate an interactive HTML report that opens in your browser
> • Provide Python code equivalent for reproducing the validation
> • Give you the flexibility to customize and extend the validation in your own scripts
> ```
