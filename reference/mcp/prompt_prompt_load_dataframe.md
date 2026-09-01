# prompt_load_dataframe


Prompt to load a DataFrame from a file into the server's context for validation.


## Arguments


`input_path: string`  
No description.

`df_id: string`  
Provide a value matching the following JSON schema: {"anyOf":\[{"type":"string"},{"type":"null"}\]}. Encode non-string values as JSON.


## Prompt Text

> **Note: Assistant**
>
> ``` text
> I can load your data from a file into my context for validation.
> ```

> **Note: User message**
>
> ``` text
> Please call `load_dataframe` with input_path='Path to the input CSV, Excel or Parquet file.'. You can optionally provide a `df_id` (e.g., 'Optional ID for the DataFrame. If not provided, a new ID will be generated.') to name this dataset, or I will generate one for you. Make a note of the returned `df_id` for subsequent steps.
> ```
