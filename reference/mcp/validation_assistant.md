# validation_assistant


Generate data-aware validation suggestions based on actual column types and statistics.

Analyzes your data's structure (types, null counts, value ranges, cardinality) to produce actionable Pointblank validation code tailored to your dataset.


``` json
{
  "tool": "validation_assistant",
  "arguments": {
    "dataframe_id": ...,
    "validation_goal": "general"
  }
}
```


# Parameters


`dataframe_id: string`  
required

`validation_goal: string = "general"`  
No description.
