# prompt_add_validation_step_example


Prompt to add a validation step to a Pointblank Validator.


# Prompt Text

> **Note: Assistant**
>
> ``` text
> I can add various validation steps to your validator. You'll need to specify the 'validator_id', 'validation_type', and 'params' for the step. For example, to check if values in column 'age' are less than 100 for validator 'validator_123':
> ```

> **Note: User message**
>
> ``` text
> Please call `add_validation_step` with validator_id='validator_123', validation_type='col_vals_lt', and params={'columns': 'age', 'value': 100}. Note: Parameter names within 'params' (like 'columns', 'value', 'left', 'right', 'set_', etc.) must exactly match what the specific Pointblank validation function expects.
> Other examples:
> - For 'col_vals_between': params={'columns': 'score', 'left': 0, 'right': 100, 'inclusive': [True, True]}
> - For 'col_vals_in_set': params={'columns': 'grade', 'set_': ['A', 'B', 'C']} (Note: Pointblank uses 'set_' for this method's list of values)
> - For 'col_exists': params={'columns': 'user_id'}
> Refer to the Pointblank Python API for the 'Validate' class for available `validation_type` (method names) and their specific `params`.
> ```
