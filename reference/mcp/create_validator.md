# create_validator


Create a Pointblank Validator for a previously loaded DataFrame.


``` json
{
  "tool": "create_validator",
  "arguments": {
    "df_id": ...,
    "validator_id": ...  // optional,
    "table_name": ...  // optional,
    "validator_label": ...  // optional,
    "thresholds_dict": ...  // optional,
    "brief": ...  // optional,
    "lang": ...  // optional,
    "locale": ...  // optional
  }
}
```


# Parameters


`df_id: string`  
ID of the DataFrame to validate. required

`validator_id: any`  
Optional ID for the Validator.

`table_name: any`  
Optional name for the table within reports.

`validator_label: any`  
Optional descriptive label.

`thresholds_dict: any`  
Optional thresholds. Example: {'warning': 0.1, 'error': 5, 'critical': 0.10}.

`brief: any`  
No description.

`lang: any`  
No description.

`locale: any`  
No description.
