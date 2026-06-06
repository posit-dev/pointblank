# apply_validation_template


Apply a pre-built validation template to a validator.


``` json
{
  "tool": "apply_validation_template",
  "arguments": {
    "validator_id": ...,
    "template_name": ...,
    "column_mapping": ...
  }
}
```


# Parameters


`validator_id: string`  
ID of the Validator to apply template to. required

`template_name: string`  
Template name. Available: basic_quality, financial_data, customer_data, sensor_data, survey_data. required

`column_mapping: object`  
Mapping of template column names to actual DataFrame column names. required
