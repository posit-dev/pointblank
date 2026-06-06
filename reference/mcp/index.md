# MCP Reference


<span class="mcp-tile-label">Tools19</span> <span class="mcp-tile-label">Resources0</span> <span class="mcp-tile-label">Templates0</span> <span class="mcp-tile-label">Prompts0</span> <span class="mcp-tile-label">Instructions✗</span> <span class="mcp-tile-label">Completions✗</span>


## Data Management

[load_dataframe](../../reference/mcp/load_dataframe.md)  
Load a DataFrame from a CSV, Excel or Parquet file into the server's context.

[list_available_backends](../../reference/mcp/list_available_backends.md)  
List available DataFrame backends (pandas, polars) installed in the environment.

[list_loaded_dataframes](../../reference/mcp/list_loaded_dataframes.md)  
List all DataFrames currently loaded in the server context.

[delete_dataframe](../../reference/mcp/delete_dataframe.md)  
Remove a DataFrame from the server context to free up memory.


## Data Analysis

[profile_dataframe](../../reference/mcp/profile_dataframe.md)  
Profile a loaded DataFrame using Pointblank's DataScan, returning column-level statistics.


## Validation

[list_active_validators](../../reference/mcp/list_active_validators.md)  
List all validators currently active in the server context.

[delete_validator](../../reference/mcp/delete_validator.md)  
Remove a validator from the server context.

[create_validator](../../reference/mcp/create_validator.md)  
Create a Pointblank Validator for a previously loaded DataFrame.

[add_validation_step](../../reference/mcp/add_validation_step.md)  
Add a validation step to an existing Pointblank Validator.

[apply_validation_template](../../reference/mcp/apply_validation_template.md)  
Apply a pre-built validation template to a validator.

[get_validation_step_output](../../reference/mcp/get_validation_step_output.md)  
Retrieve output for a validation step and save it to a CSV file.

[interrogate_validator](../../reference/mcp/interrogate_validator.md)  
Run validations and return a JSON summary with Python code equivalent.


## Table Visualization

[preview_table](../../reference/mcp/preview_table.md)  
Display a preview of the DataFrame showing rows from top and bottom.

[missing_values_table](../../reference/mcp/missing_values_table.md)  
Generate a table showing missing values analysis for the DataFrame.

[column_summary_table](../../reference/mcp/column_summary_table.md)  
Generate a comprehensive column-level summary of the DataFrame.


## AI & Assistant

[draft_validation_plan](../../reference/mcp/draft_validation_plan.md)  
Generate an AI-powered validation plan using Pointblank's DraftValidation class.

[validation_assistant](../../reference/mcp/validation_assistant.md)  
Generate data-aware validation suggestions based on actual column types and statistics.


## Reference & Management

[server_health_check](../../reference/mcp/server_health_check.md)  
Get comprehensive server health and status information.

[get_pointblank_api_reference](../../reference/mcp/get_pointblank_api_reference.md)  
Get API reference for Pointblank validation methods and common patterns.
