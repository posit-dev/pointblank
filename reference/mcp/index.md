# MCP Reference

Tools19 Resources0 Templates0 Prompts5 Instructions✗ Completions✗

### Data Management

[load_dataframe](load_dataframe.md)  
Load a DataFrame from a CSV, Excel or Parquet file into the server's context.

[list_available_backends](list_available_backends.md)  
List available DataFrame backends (pandas, polars) installed in the environment.

[list_loaded_dataframes](list_loaded_dataframes.md)  
List all DataFrames currently loaded in the server context.

[delete_dataframe](delete_dataframe.md)  
Remove a DataFrame from the server context to free up memory.

### Data Analysis

[profile_dataframe](profile_dataframe.md)  
Profile a loaded DataFrame using Pointblank's DataScan, returning column-level statistics.

### Validation

[list_active_validators](list_active_validators.md)  
List all validators currently active in the server context.

[delete_validator](delete_validator.md)  
Remove a validator from the server context.

[create_validator](create_validator.md)  
Create a Pointblank Validator for a previously loaded DataFrame.

[add_validation_step](add_validation_step.md)  
Add a validation step to an existing Pointblank Validator.

[apply_validation_template](apply_validation_template.md)  
Apply a pre-built validation template to a validator.

[get_validation_step_output](get_validation_step_output.md)  
Retrieve output for a validation step and save it to a CSV file.

[interrogate_validator](interrogate_validator.md)  
Run validations and return a JSON summary with Python code equivalent.

### Table Visualization

[preview_table](preview_table.md)  
Display a preview of the DataFrame showing rows from top and bottom.

[missing_values_table](missing_values_table.md)  
Generate a table showing missing values analysis for the DataFrame.

[column_summary_table](column_summary_table.md)  
Generate a comprehensive column-level summary of the DataFrame.

### AI & Assistant

[draft_validation_plan](draft_validation_plan.md)  
Generate an AI-powered validation plan using Pointblank's DraftValidation class.

[validation_assistant](validation_assistant.md)  
Generate data-aware validation suggestions based on actual column types and statistics.

### Reference & Management

[server_health_check](server_health_check.md)  
Get comprehensive server health and status information.

[get_pointblank_api_reference](get_pointblank_api_reference.md)  
Get API reference for Pointblank validation methods and common patterns.

### Prompts

[prompt_load_dataframe](prompt_prompt_load_dataframe.md)  
Prompt to load a DataFrame from a file into the server's context for validation.

[prompt_create_validator](prompt_prompt_create_validator.md)  
Prompt to create a Pointblank Validator for a loaded DataFrame.

[prompt_add_validation_step_example](prompt_prompt_add_validation_step_example.md)  
Prompt to add a validation step to a Pointblank Validator.

[prompt_get_validation_step_output](prompt_prompt_get_validation_step_output.md)  
Prompt to get validation output by specifying either a step index or a sundered type.

[prompt_interrogate_validator](prompt_prompt_interrogate_validator.md)  
Prompt to run validations and generate reports with Python code.
