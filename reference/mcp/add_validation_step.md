# add_validation_step


Add a validation step to an existing Pointblank Validator.


``` json
{
  "tool": "add_validation_step",
  "arguments": {
    "validator_id": ...,
    "validation_type": ...,
    "params": ...
  }
}
```


# Parameters


`validator_id: string`  
ID of the Validator to add a step to. required

`validation_type: string`  
Type of validation to perform (e.g., 'col_vals_lt', 'col_vals_between', 'rows_distinct'). required

`params: object`  
Parameters for the validation function matching Pointblank's API. required
