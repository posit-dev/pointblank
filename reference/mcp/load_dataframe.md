# load_dataframe


Load a DataFrame from a CSV, Excel or Parquet file into the server's context.


``` json
{
  "tool": "load_dataframe",
  "arguments": {
    "input_path": ...,
    "df_id": ...  // optional,
    "backend": "auto"
  }
}
```


# Parameters


`input_path: string`  
Path to the input CSV, Excel or Parquet file. required

`df_id: any`  
No description.

`backend: string = "auto"`  
DataFrame backend to use: 'auto', 'pandas', or 'polars'. Default is 'auto'.
