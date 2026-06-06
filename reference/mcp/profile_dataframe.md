# profile_dataframe


Profile a loaded DataFrame using Pointblank's DataScan, returning column-level statistics.


``` json
{
  "tool": "profile_dataframe",
  "arguments": {
    "df_id": ...,
    "sample_size": 0
  }
}
```


# Parameters


`df_id: string`  
ID of the DataFrame to profile. required

`sample_size: integer = 0`  
Maximum number of rows to sample for profiling (0 = all rows).
