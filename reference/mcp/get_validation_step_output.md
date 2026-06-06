# get_validation_step_output


Retrieve output for a validation step and save it to a CSV file.


``` json
{
  "tool": "get_validation_step_output",
  "arguments": {
    "validator_id": ...,
    "output_path": ...,
    "sundered_type": "fail",
    "step_index": ...  // optional
  }
}
```


# Parameters


`validator_id: string`  
ID of the Validator to retrieve output from. required

`output_path: string`  
Path to save the output file. Must end with .csv. required

`sundered_type: string = "fail"`  
Retrieve all 'pass' or 'fail' rows.

`step_index: any`  
Specific step index (0-based). Overrides sundered_type.
