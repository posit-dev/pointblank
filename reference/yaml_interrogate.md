## yaml_interrogate()


Execute a YAML-based validation workflow.


Usage

``` python
yaml_interrogate(
    yaml,
    set_tbl=None,
    namespaces=None,
)
```


This is the main entry point for YAML-based validation workflows. It takes YAML configuration (as a string or file path) and returns a validated [Validate](Validate.md#pointblank.Validate) object with interrogation results.

The YAML configuration defines the data source, validation steps, and optional settings like thresholds and labels. This function automatically loads the data, builds the validation plan, executes all validation steps, and returns the interrogated results.


## Parameters


`yaml: Union[str, Path]`  
YAML configuration as string or file path. Can be: (1) a YAML string containing the validation configuration, or (2) a Path object or string path to a YAML file.

`set_tbl: Any = None`  
An optional table to override the table specified in the YAML configuration. This allows you to apply a YAML-defined validation workflow to a different table than what's specified in the configuration. If provided, this table will replace the table defined in the YAML's `tbl` field before executing the validation workflow. This can be any supported table type including DataFrame objects, Ibis table objects, CSV file paths, Parquet file paths, GitHub URLs, or database connection strings.

`namespaces: Optional[Union[Iterable[str], Mapping[str, str]]] = None`  
Optional module namespaces to make available for Python code execution in YAML configurations. Can be a dictionary mapping aliases to module names or a list of module names. See the "Using Namespaces" section below for detailed examples.


## Returns


`Validate`  
An instance of the [Validate](Validate.md#pointblank.Validate) class that has been configured based on the YAML input. This object contains the results of the validation steps defined in the YAML configuration. It includes metadata like table name, label, language, and thresholds if specified.


## Raises


`YAMLValidationError`  
If the YAML is invalid, malformed, or execution fails. This includes syntax errors, missing required fields, unknown validation methods, or data loading failures.


## Using Namespaces

The `namespaces=` parameter enables custom Python modules and functions in YAML configurations. This is particularly useful for custom action functions and advanced Python expressions.

**Namespace formats:**

- Dictionary format: `{"alias": "module.name"}` maps aliases to module names
- List format: `["module.name", "another.module"]` imports modules directly

**Option 1: Inline expressions (no namespaces needed)**


``` python
import pointblank as pb

# Simple inline custom action
yaml_config = '''
tbl: small_table
thresholds:
  warning: 0.01
actions:
  warning:
    python: "lambda: print('Custom warning triggered')"
steps:
- col_vals_gt:
    columns: [a]
    value: 1000
'''

result = pb.yaml_interrogate(yaml_config)
result
```


    Custom warning triggered


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-05-16|00:23:32</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span><span style="background-color: #AAAAAA; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; border: solid 1px #AAAAAA; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">WARNING</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #AAAAAA; padding: 2px 15px 2px 15px; font-size: smaller; margin-right: 5px;">0.01</span><span style="background-color: #EBBC14; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #EBBC14; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">ERROR</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #EBBC14; padding: 2px 15px 2px 15px; font-size: smaller; margin-right: 5px;">--</span><span style="background-color: #FF3300; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #FF3300; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">CRITICAL</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #FF3300; padding: 2px 15px 2px 15px; font-size: smaller;">--</span></span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">STEP</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALUES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASS</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FAIL</th>
<th id="pb_tbl-w_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">W</th>
<th id="pb_tbl-e_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">E</th>
<th id="pb_tbl-c_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">C</th>
<th id="pb_tbl-extract_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EXT</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #AAAAAA; color: transparent; font-size: 0px">#AAAAAA</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">1000</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">13<br />
1.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">●</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><div style="margin-top: 5px; margin-bottom: 5px;">
<span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin-left: 10px; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-05-16 00:23:32 UTC</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">< 1 s</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 1px 5px -1px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-05-16 00:23:32 UTC</span>
</div></td>
</tr>
</tfoot>

</table>


**Option 2: External functions with namespaces**


``` python
# Define a custom action function
def my_custom_action():
    print("Data validation failed: please check your data.")

# Add to current module for demo
import sys
sys.modules[__name__].my_custom_action = my_custom_action

# YAML that references the external function
yaml_config = '''
tbl: small_table
thresholds:
  warning: 0.01
actions:
  warning:
    python: actions.my_custom_action
steps:
- col_vals_gt:
    columns: [a]
    value: 1000  # This will fail
'''

# Use namespaces to make the function available
result = pb.yaml_interrogate(yaml_config, namespaces={'actions': '__main__'})
result
```


    Data validation failed: please check your data.


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-05-16|00:23:32</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span><span style="background-color: #AAAAAA; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; border: solid 1px #AAAAAA; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">WARNING</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #AAAAAA; padding: 2px 15px 2px 15px; font-size: smaller; margin-right: 5px;">0.01</span><span style="background-color: #EBBC14; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #EBBC14; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">ERROR</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #EBBC14; padding: 2px 15px 2px 15px; font-size: smaller; margin-right: 5px;">--</span><span style="background-color: #FF3300; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #FF3300; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">CRITICAL</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #FF3300; padding: 2px 15px 2px 15px; font-size: smaller;">--</span></span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">STEP</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALUES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASS</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FAIL</th>
<th id="pb_tbl-w_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">W</th>
<th id="pb_tbl-e_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">E</th>
<th id="pb_tbl-c_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">C</th>
<th id="pb_tbl-extract_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EXT</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #AAAAAA; color: transparent; font-size: 0px">#AAAAAA</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">1000</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">13<br />
1.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">●</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><div style="margin-top: 5px; margin-bottom: 5px;">
<span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin-left: 10px; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-05-16 00:23:32 UTC</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">< 1 s</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 1px 5px -1px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-05-16 00:23:32 UTC</span>
</div></td>
</tr>
</tfoot>

</table>


This approach enables modular, reusable validation workflows with custom business logic.


## Examples

For the examples here, we'll use YAML configurations to define validation workflows. Let's start with a basic YAML workflow that validates the built-in `small_table` dataset.


``` python
import pointblank as pb

# Define a basic YAML validation workflow
yaml_config = '''
tbl: small_table
steps:
- rows_distinct
- col_exists:
    columns: [date, a, b]
'''

# Execute the validation workflow
result = pb.yaml_interrogate(yaml_config)
result
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">STEP</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALUES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASS</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FAIL</th>
<th id="pb_tbl-w_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">W</th>
<th id="pb_tbl-e_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">E</th>
<th id="pb_tbl-c_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">C</th>
<th id="pb_tbl-extract_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EXT</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93c19kaXN0aW5jdDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJyb3dzX2Rpc3RpbmN0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC40ODI3NTkpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJub19nZW1pbmkiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE3LjAwMDAwMCwgMTMuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMy42NjcwNTYxOSw2LjYyMTA3NTA4IEMzLjEyNTEwMTA0LDYuNjQwNjYzODYgMi42NzQ1NTk3NCw3LjA0NzY3NDQ0IDIuNjAyNzM0MzIsNy41ODUyNzY4MiBDMi41Mjg3MzIyOCw4LjEyMjg3OTIgMi44NTMwMzUyNiw4LjYzNDM2MjcgMy4zNzEwNDg0OCw4Ljc5NzYwMjM5IEM0LjQwNDg5OTA5LDkuMTU0NTUyODYgNi43MDExMzU1Myw5Ljg3MDYzMDIxIDkuODY1ODA1OTUsMTAuMzY0NzAyIEw5Ljg2NTgwNTk1LDMwLjczNjk5NzYgQzYuNjU5NzgxMzcsMzEuMjMzMjQ1OCA0LjM3MjI1MTA0LDMxLjk2NDU1OSAzLjM3MTA0ODQ4LDMyLjMyMTUwOTUgQzIuNzg5OTE1NTUsMzIuNTI4Mjc5NyAyLjQ4NTIwMTczLDMzLjE2ODE3ODQgMi42OTE5NzE5NiwzMy43NDkzMTE0IEMyLjg5ODc0MjIsMzQuMzMwNDQ0MyAzLjUzODY0MDk0LDM0LjYzNTE1ODEgNC4xMTk3NzM4NywzNC40MjgzODc5IEM1LjU0OTc1MjE3LDMzLjkxOTA4MDggMTAuMjAzMTY3OCwzMi40NjA4MDcyIDE2LjUxNzI3MzQsMzIuNDYwODA3MiBDMjIuNzk0Mzc4MSwzMi40NjA4MDcyIDI3LjU1MDA5MDEsMzMuODkwNzg1NSAyOS4wNTQwNzA2LDM0LjQxMDk3NTcgQzI5LjYzNTIwMzYsMzQuNjEzMzkyNiAzMC4yNzA3NDk1LDM0LjMwNDMyNiAzMC40NzMxNjY0LDMzLjcyMzE5MzEgQzMwLjY3NTU4MzMsMzMuMTQyMDYwMSAzMC4zNjY1MTY3LDMyLjUwNjUxNDIgMjkuNzg1MzgzOCwzMi4zMDQwOTczIEMyOC43NDkzNTY4LDMxLjk0NDk3MDQgMjYuNDMxMzU1NCwzMS4yNDQxMjgzIDIzLjIzODM4OTcsMzAuNzU0NDA5OCBMMjMuMjM4Mzg5NywxMC4zODIxMTQzIEMyNi40NDQ0MTQzLDkuODg4MDQyNDMgMjguNzQ1MDA0LDkuMTYxMDgyNTkgMjkuNzY3OTcxNiw4Ljc5NzYwMjM5IEMzMC4zNDkxMDQ1LDguNTkwODMyMTUgMzAuNjUzODE4NCw3Ljk1MDkzMzQxIDMwLjQ0NzA0ODEsNy4zNjk4MDA0OCBDMzAuMjQwMjc3OSw2Ljc4ODY2NzU1IDI5LjYwMDM3OTEsNi40ODM5NTM3MyAyOS4wMTkyNDYyLDYuNjkwNzIzOTYgQzI3LjU1ODc5NjMsNy4yMDg3Mzc3NCAyMi45MTYyNjM3LDguNjU4MzA0NjQgMTYuNjM5MTU4OSw4LjY1ODMwNDY0IEMxMC4zNzA3NjAzLDguNjU4MzA0NjQgNS42MTI4NzE4OCw3LjIxMzA5MDUxIDQuMTAyMzYxNjYsNi42OTA3MjM5NiBDMy45NjMwNjM5MSw2LjYzODQ4NzMgMy44MTUwNjAwNSw2LjYxNDU0NTQ4IDMuNjY3MDU2MTksNi42MjEwNzUwOCBaIE0xMi4wOTQ1Njk5LDEwLjY0MzI5NzUgQzEzLjQ5NDA3NzEsMTAuNzg5MTI1IDE1LjAxOTgyMjYsMTAuODg3MDY4NiAxNi42MzkxNTg5LDEwLjg4NzA2ODYgQzE4LjE5OTcyODksMTAuODg3MDY4NiAxOS42NjIzNTUyLDEwLjc5NzgzMTEgMjEuMDA5NjI1NywxMC42NjA3MDk4IEwyMS4wMDk2MjU3LDMwLjQ1ODQwMjEgQzE5LjYyMzE3OCwzMC4zMTY5MjggMTguMTE5MTk3NSwzMC4yMzIwNDMzIDE2LjUxNzI3MzQsMzAuMjMyMDQzMyBDMTQuOTMyNzYxNSwzMC4yMzIwNDMzIDEzLjQ1NzA3NjMsMzAuMzE5MTA0NCAxMi4wOTQ1Njk5LDMwLjQ1ODQwMjEgTDEyLjA5NDU2OTksMTAuNjQzMjk3NSBaIiBpZD0iZ2VtaW5pIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTE2LjY2MDUzNTQsLTUuMDU5Mjk0OTkgQzE2LjkzNjY3NzgsLTUuMDU5Mjk0OTkgMTcuMTYwNTM1NCwtNC44MzU0MzczNyAxNy4xNjA1MzU0LC00LjU1OTI5NDk5IEwxNy4xNjA1MzU0LDQ0LjU2ODc1MDkgQzE3LjE2MDUzNTQsNDQuODQ0ODkzMyAxNi45MzY2Nzc4LDQ1LjA2ODc1MDkgMTYuNjYwNTM1NCw0NS4wNjg3NTA5IEMxNi4zODQzOTMsNDUuMDY4NzUwOSAxNi4xNjA1MzU0LDQ0Ljg0NDg5MzMgMTYuMTYwNTM1NCw0NC41Njg3NTA5IEwxNi4xNjA1MzU0LC00LjU1OTI5NDk5IEMxNi4xNjA1MzU0LC00LjgzNTQzNzM3IDE2LjM4NDM5MywtNS4wNTkyOTQ5OSAxNi42NjA1MzU0LC01LjA1OTI5NDk5IFoiIGlkPSJsaW5lX2JsYWNrIiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNi42NjA1MzUsIDIwLjAwNDcyOCkgcm90YXRlKC0zMjAuMDAwMDAwKSB0cmFuc2xhdGUoLTE2LjY2MDUzNSwgLTIwLjAwNDcyOCkgIiAvPgogICAgICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4LjU2MDAzMiwgMTkuMTU4MDMxKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMTguNTYwMDMyLCAtMTkuMTU4MDMxKSAiIHBvaW50cz0iMTguMDYwMDMxNiAtNC40NTM2NjczNSAxOS4wNjAwMzE2IC00LjQ1MzY2NzM1IDE5LjA2MDAzMTYgNDIuNzY5NzI5OSAxOC4wNjAwMzE2IDQyLjc2OTcyOTkiPjwvcG9seWdvbj4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

rows_distinct()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">ALL COLUMNS</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">9<br />
0.69</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.15</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
A3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">4</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody>
</table>


The validation table shows the results of our YAML-defined workflow. We can see that the [rows_distinct()](Validate.rows_distinct.md#pointblank.Validate.rows_distinct) validation failed (because there are duplicate rows in the table), while the column existence checks passed.

Now let's create a more comprehensive validation workflow with thresholds and metadata:


``` python
# Advanced YAML configuration with thresholds and metadata
yaml_config = '''
tbl: small_table
tbl_name: small_table_demo
label: Comprehensive data validation
thresholds:
  warning: 0.1
  error: 0.25
  critical: 0.35
steps:
- col_vals_gt:
    columns: [d]
    value: 100
- col_vals_regex:
    columns: [b]
    pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
- col_vals_not_null:
    columns: [date, a]
'''

# Execute the validation workflow
result = pb.yaml_interrogate(yaml_config)
print(f"Table name: {result.tbl_name}")
print(f"Label: {result.label}")
print(f"Total validation steps: {len(result.validation_info)}")
```


    Table name: small_table_demo
    Label: Comprehensive data validation
    Total validation steps: 4


The validation results now include our custom table name and label. The thresholds we defined will determine when validation steps are marked as warnings, errors, or critical failures.

You can also load YAML configurations from files. Here's how you would work with a YAML file:


``` python
from pathlib import Path
import tempfile

# Create a temporary YAML file for demonstration
yaml_content = '''
tbl: small_table
tbl_name: File-based Validation
steps:
- col_vals_between:
    columns: [c]
    left: 1
    right: 10
- col_vals_in_set:
    columns: [f]
    set: [low, mid, high]
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    f.write(yaml_content)
    yaml_file_path = Path(f.name)

# Load and execute validation from file
result = pb.yaml_interrogate(yaml_file_path)
result
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">STEP</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALUES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASS</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FAIL</th>
<th id="pb_tbl-w_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">W</th>
<th id="pb_tbl-e_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">E</th>
<th id="pb_tbl-c_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">C</th>
<th id="pb_tbl-extract_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EXT</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfYmV0d2VlbjwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19iZXR3ZWVuIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4yMDY4OTcpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS45OTM0ODQsMjEuOTY4NzUgQzEwLjk2MjIzNCwyMi4wODIwMzEgMTAuMTg4Nzk3LDIyLjk2NDg0NCAxMC4yMTIyMzQsMjQgTDEwLjIxMjIzNCw0MiBDMTAuMjAwNTE1LDQyLjcyMjY1NiAxMC41Nzk0MjIsNDMuMzkwNjI1IDExLjIwNDQyMiw0My43NTM5MDYgQzExLjgyNTUxNSw0NC4xMjEwOTQgMTIuNTk4OTUzLDQ0LjEyMTA5NCAxMy4yMjAwNDcsNDMuNzUzOTA2IEMxMy44NDUwNDcsNDMuMzkwNjI1IDE0LjIyMzk1Myw0Mi43MjI2NTYgMTQuMjEyMjM0LDQyIEwxNC4yMTIyMzQsMjQgQzE0LjIyMDA0NywyMy40NTcwMzEgMTQuMDA5MTA5LDIyLjkzNzUgMTMuNjI2Mjk3LDIyLjU1NDY4OCBDMTMuMjQzNDg0LDIyLjE3MTg3NSAxMi43MjM5NTMsMjEuOTYwOTM4IDEyLjE4MDk4NCwyMS45Njg3NSBDMTIuMTE4NDg0LDIxLjk2NDg0NCAxMi4wNTU5ODQsMjEuOTY0ODQ0IDExLjk5MzQ4NCwyMS45Njg3NSBaIE01NS45OTM0ODQsMjEuOTY4NzUgQzU0Ljk2MjIzNCwyMi4wODIwMzEgNTQuMTg4Nzk3LDIyLjk2NDg0NCA1NC4yMTIyMzQsMjQgTDU0LjIxMjIzNCw0MiBDNTQuMjAwNTE1LDQyLjcyMjY1NiA1NC41Nzk0MjIsNDMuMzkwNjI1IDU1LjIwNDQyMiw0My43NTM5MDYgQzU1LjgyNTUxNSw0NC4xMjEwOTQgNTYuNTk4OTUzLDQ0LjEyMTA5NCA1Ny4yMjAwNDcsNDMuNzUzOTA2IEM1Ny44NDUwNDcsNDMuMzkwNjI1IDU4LjIyMzk1Myw0Mi43MjI2NTYgNTguMjEyMjM0LDQyIEw1OC4yMTIyMzQsMjQgQzU4LjIyMDA0NywyMy40NTcwMzEgNTguMDA5MTA5LDIyLjkzNzUgNTcuNjI2Mjk3LDIyLjU1NDY4OCBDNTcuMjQzNDg0LDIyLjE3MTg3NSA1Ni43MjM5NTMsMjEuOTYwOTM4IDU2LjE4MDk4NCwyMS45Njg3NSBDNTYuMTE4NDg0LDIxLjk2NDg0NCA1Ni4wNTU5ODQsMjEuOTY0ODQ0IDU1Ljk5MzQ4NCwyMS45Njg3NSBaIE0xNi4yMTIyMzQsMjIgQzE1LjY2MTQ1MywyMiAxNS4yMTIyMzQsMjIuNDQ5MjE5IDE1LjIxMjIzNCwyMyBDMTUuMjEyMjM0LDIzLjU1MDc4MSAxNS42NjE0NTMsMjQgMTYuMjEyMjM0LDI0IEMxNi43NjMwMTUsMjQgMTcuMjEyMjM0LDIzLjU1MDc4MSAxNy4yMTIyMzQsMjMgQzE3LjIxMjIzNCwyMi40NDkyMTkgMTYuNzYzMDE1LDIyIDE2LjIxMjIzNCwyMiBaIE0yMC4yMTIyMzQsMjIgQzE5LjY2MTQ1MywyMiAxOS4yMTIyMzQsMjIuNDQ5MjE5IDE5LjIxMjIzNCwyMyBDMTkuMjEyMjM0LDIzLjU1MDc4MSAxOS42NjE0NTMsMjQgMjAuMjEyMjM0LDI0IEMyMC43NjMwMTUsMjQgMjEuMjEyMjM0LDIzLjU1MDc4MSAyMS4yMTIyMzQsMjMgQzIxLjIxMjIzNCwyMi40NDkyMTkgMjAuNzYzMDE1LDIyIDIwLjIxMjIzNCwyMiBaIE0yNC4yMTIyMzQsMjIgQzIzLjY2MTQ1MywyMiAyMy4yMTIyMzQsMjIuNDQ5MjE5IDIzLjIxMjIzNCwyMyBDMjMuMjEyMjM0LDIzLjU1MDc4MSAyMy42NjE0NTMsMjQgMjQuMjEyMjM0LDI0IEMyNC43NjMwMTUsMjQgMjUuMjEyMjM0LDIzLjU1MDc4MSAyNS4yMTIyMzQsMjMgQzI1LjIxMjIzNCwyMi40NDkyMTkgMjQuNzYzMDE1LDIyIDI0LjIxMjIzNCwyMiBaIE0yOC4yMTIyMzQsMjIgQzI3LjY2MTQ1MywyMiAyNy4yMTIyMzQsMjIuNDQ5MjE5IDI3LjIxMjIzNCwyMyBDMjcuMjEyMjM0LDIzLjU1MDc4MSAyNy42NjE0NTMsMjQgMjguMjEyMjM0LDI0IEMyOC43NjMwMTUsMjQgMjkuMjEyMjM0LDIzLjU1MDc4MSAyOS4yMTIyMzQsMjMgQzI5LjIxMjIzNCwyMi40NDkyMTkgMjguNzYzMDE1LDIyIDI4LjIxMjIzNCwyMiBaIE0zMi4yMTIyMzQsMjIgQzMxLjY2MTQ1MywyMiAzMS4yMTIyMzQsMjIuNDQ5MjE5IDMxLjIxMjIzNCwyMyBDMzEuMjEyMjM0LDIzLjU1MDc4MSAzMS42NjE0NTMsMjQgMzIuMjEyMjM0LDI0IEMzMi43NjMwMTUsMjQgMzMuMjEyMjM0LDIzLjU1MDc4MSAzMy4yMTIyMzQsMjMgQzMzLjIxMjIzNCwyMi40NDkyMTkgMzIuNzYzMDE1LDIyIDMyLjIxMjIzNCwyMiBaIE0zNi4yMTIyMzQsMjIgQzM1LjY2MTQ1MywyMiAzNS4yMTIyMzQsMjIuNDQ5MjE5IDM1LjIxMjIzNCwyMyBDMzUuMjEyMjM0LDIzLjU1MDc4MSAzNS42NjE0NTMsMjQgMzYuMjEyMjM0LDI0IEMzNi43NjMwMTUsMjQgMzcuMjEyMjM0LDIzLjU1MDc4MSAzNy4yMTIyMzQsMjMgQzM3LjIxMjIzNCwyMi40NDkyMTkgMzYuNzYzMDE1LDIyIDM2LjIxMjIzNCwyMiBaIE00MC4yMTIyMzQsMjIgQzM5LjY2MTQ1MywyMiAzOS4yMTIyMzQsMjIuNDQ5MjE5IDM5LjIxMjIzNCwyMyBDMzkuMjEyMjM0LDIzLjU1MDc4MSAzOS42NjE0NTMsMjQgNDAuMjEyMjM0LDI0IEM0MC43NjMwMTUsMjQgNDEuMjEyMjM0LDIzLjU1MDc4MSA0MS4yMTIyMzQsMjMgQzQxLjIxMjIzNCwyMi40NDkyMTkgNDAuNzYzMDE1LDIyIDQwLjIxMjIzNCwyMiBaIE00NC4yMTIyMzQsMjIgQzQzLjY2MTQ1MywyMiA0My4yMTIyMzQsMjIuNDQ5MjE5IDQzLjIxMjIzNCwyMyBDNDMuMjEyMjM0LDIzLjU1MDc4MSA0My42NjE0NTMsMjQgNDQuMjEyMjM0LDI0IEM0NC43NjMwMTUsMjQgNDUuMjEyMjM0LDIzLjU1MDc4MSA0NS4yMTIyMzQsMjMgQzQ1LjIxMjIzNCwyMi40NDkyMTkgNDQuNzYzMDE1LDIyIDQ0LjIxMjIzNCwyMiBaIE00OC4yMTIyMzQsMjIgQzQ3LjY2MTQ1MywyMiA0Ny4yMTIyMzQsMjIuNDQ5MjE5IDQ3LjIxMjIzNCwyMyBDNDcuMjEyMjM0LDIzLjU1MDc4MSA0Ny42NjE0NTMsMjQgNDguMjEyMjM0LDI0IEM0OC43NjMwMTUsMjQgNDkuMjEyMjM0LDIzLjU1MDc4MSA0OS4yMTIyMzQsMjMgQzQ5LjIxMjIzNCwyMi40NDkyMTkgNDguNzYzMDE1LDIyIDQ4LjIxMjIzNCwyMiBaIE01Mi4yMTIyMzQsMjIgQzUxLjY2MTQ1MywyMiA1MS4yMTIyMzQsMjIuNDQ5MjE5IDUxLjIxMjIzNCwyMyBDNTEuMjEyMjM0LDIzLjU1MDc4MSA1MS42NjE0NTMsMjQgNTIuMjEyMjM0LDI0IEM1Mi43NjMwMTUsMjQgNTMuMjEyMjM0LDIzLjU1MDc4MSA1My4yMTIyMzQsMjMgQzUzLjIxMjIzNCwyMi40NDkyMTkgNTIuNzYzMDE1LDIyIDUyLjIxMjIzNCwyMiBaIE0yMS40NjIyMzQsMjcuOTY4NzUgQzIxLjQxOTI2NSwyNy45NzY1NjMgMjEuMzc2Mjk3LDI3Ljk4ODI4MSAyMS4zMzcyMzQsMjggQzIxLjE3NzA3OCwyOC4wMjczNDQgMjEuMDI4NjQsMjguMDg5ODQ0IDIwLjg5OTczNCwyOC4xODc1IEwxNS42MTg0ODQsMzIuMTg3NSBDMTUuMzU2NzY1LDMyLjM3NSAxNS4yMDA1MTUsMzIuNjc5Njg4IDE1LjIwMDUxNSwzMyBDMTUuMjAwNTE1LDMzLjMyMDMxMyAxNS4zNTY3NjUsMzMuNjI1IDE1LjYxODQ4NCwzMy44MTI1IEwyMC44OTk3MzQsMzcuODEyNSBDMjEuMzQ4OTUzLDM4LjE0ODQzOCAyMS45ODU2NzIsMzguMDU4NTk0IDIyLjMyMTYwOSwzNy42MDkzNzUgQzIyLjY1NzU0NywzNy4xNjAxNTYgMjIuNTY3NzAzLDM2LjUyMzQzOCAyMi4xMTg0ODQsMzYuMTg3NSBMMTkuMjEyMjM0LDM0IEw0OS4yMTIyMzQsMzQgTDQ2LjMwNTk4NCwzNi4xODc1IEM0NS44NTY3NjUsMzYuNTIzNDM4IDQ1Ljc2NjkyMiwzNy4xNjAxNTYgNDYuMTAyODU5LDM3LjYwOTM3NSBDNDYuNDM4Nzk3LDM4LjA1ODU5NCA0Ny4wNzU1MTUsMzguMTQ4NDM4IDQ3LjUyNDczNCwzNy44MTI1IEw1Mi44MDU5ODQsMzMuODEyNSBDNTMuMDY3NzAzLDMzLjYyNSA1My4yMjM5NTMsMzMuMzIwMzEzIDUzLjIyMzk1MywzMyBDNTMuMjIzOTUzLDMyLjY3OTY4OCA1My4wNjc3MDMsMzIuMzc1IDUyLjgwNTk4NCwzMi4xODc1IEw0Ny41MjQ3MzQsMjguMTg3NSBDNDcuMzA5ODksMjguMDI3MzQ0IDQ3LjA0MDM1OSwyNy45NjA5MzggNDYuNzc0NzM0LDI4IEM0Ni43NDM0ODQsMjggNDYuNzEyMjM0LDI4IDQ2LjY4MDk4NCwyOCBDNDYuMjgyNTQ3LDI4LjA3NDIxOSA0NS45NjYxNCwyOC4zODI4MTMgNDUuODg0MTA5LDI4Ljc4MTI1IEM0NS44MDIwNzgsMjkuMTc5Njg4IDQ1Ljk3MDA0NywyOS41ODU5MzggNDYuMzA1OTg0LDI5LjgxMjUgTDQ5LjIxMjIzNCwzMiBMMTkuMjEyMjM0LDMyIEwyMi4xMTg0ODQsMjkuODEyNSBDMjIuNTIwODI4LDI5LjU2NjQwNiAyMi42OTY2MDksMjkuMDcwMzEzIDIyLjUzNjQ1MywyOC42MjUgQzIyLjM4MDIwMywyOC4xNzk2ODggMjEuOTMwOTg0LDI3LjkwNjI1IDIxLjQ2MjIzNCwyNy45Njg3NSBaIE0xNi4yMTIyMzQsNDIgQzE1LjY2MTQ1Myw0MiAxNS4yMTIyMzQsNDIuNDQ5MjE5IDE1LjIxMjIzNCw0MyBDMTUuMjEyMjM0LDQzLjU1MDc4MSAxNS42NjE0NTMsNDQgMTYuMjEyMjM0LDQ0IEMxNi43NjMwMTUsNDQgMTcuMjEyMjM0LDQzLjU1MDc4MSAxNy4yMTIyMzQsNDMgQzE3LjIxMjIzNCw0Mi40NDkyMTkgMTYuNzYzMDE1LDQyIDE2LjIxMjIzNCw0MiBaIE0yMC4yMTIyMzQsNDIgQzE5LjY2MTQ1Myw0MiAxOS4yMTIyMzQsNDIuNDQ5MjE5IDE5LjIxMjIzNCw0MyBDMTkuMjEyMjM0LDQzLjU1MDc4MSAxOS42NjE0NTMsNDQgMjAuMjEyMjM0LDQ0IEMyMC43NjMwMTUsNDQgMjEuMjEyMjM0LDQzLjU1MDc4MSAyMS4yMTIyMzQsNDMgQzIxLjIxMjIzNCw0Mi40NDkyMTkgMjAuNzYzMDE1LDQyIDIwLjIxMjIzNCw0MiBaIE0yNC4yMTIyMzQsNDIgQzIzLjY2MTQ1Myw0MiAyMy4yMTIyMzQsNDIuNDQ5MjE5IDIzLjIxMjIzNCw0MyBDMjMuMjEyMjM0LDQzLjU1MDc4MSAyMy42NjE0NTMsNDQgMjQuMjEyMjM0LDQ0IEMyNC43NjMwMTUsNDQgMjUuMjEyMjM0LDQzLjU1MDc4MSAyNS4yMTIyMzQsNDMgQzI1LjIxMjIzNCw0Mi40NDkyMTkgMjQuNzYzMDE1LDQyIDI0LjIxMjIzNCw0MiBaIE0yOC4yMTIyMzQsNDIgQzI3LjY2MTQ1Myw0MiAyNy4yMTIyMzQsNDIuNDQ5MjE5IDI3LjIxMjIzNCw0MyBDMjcuMjEyMjM0LDQzLjU1MDc4MSAyNy42NjE0NTMsNDQgMjguMjEyMjM0LDQ0IEMyOC43NjMwMTUsNDQgMjkuMjEyMjM0LDQzLjU1MDc4MSAyOS4yMTIyMzQsNDMgQzI5LjIxMjIzNCw0Mi40NDkyMTkgMjguNzYzMDE1LDQyIDI4LjIxMjIzNCw0MiBaIE0zMi4yMTIyMzQsNDIgQzMxLjY2MTQ1Myw0MiAzMS4yMTIyMzQsNDIuNDQ5MjE5IDMxLjIxMjIzNCw0MyBDMzEuMjEyMjM0LDQzLjU1MDc4MSAzMS42NjE0NTMsNDQgMzIuMjEyMjM0LDQ0IEMzMi43NjMwMTUsNDQgMzMuMjEyMjM0LDQzLjU1MDc4MSAzMy4yMTIyMzQsNDMgQzMzLjIxMjIzNCw0Mi40NDkyMTkgMzIuNzYzMDE1LDQyIDMyLjIxMjIzNCw0MiBaIE0zNi4yMTIyMzQsNDIgQzM1LjY2MTQ1Myw0MiAzNS4yMTIyMzQsNDIuNDQ5MjE5IDM1LjIxMjIzNCw0MyBDMzUuMjEyMjM0LDQzLjU1MDc4MSAzNS42NjE0NTMsNDQgMzYuMjEyMjM0LDQ0IEMzNi43NjMwMTUsNDQgMzcuMjEyMjM0LDQzLjU1MDc4MSAzNy4yMTIyMzQsNDMgQzM3LjIxMjIzNCw0Mi40NDkyMTkgMzYuNzYzMDE1LDQyIDM2LjIxMjIzNCw0MiBaIE00MC4yMTIyMzQsNDIgQzM5LjY2MTQ1Myw0MiAzOS4yMTIyMzQsNDIuNDQ5MjE5IDM5LjIxMjIzNCw0MyBDMzkuMjEyMjM0LDQzLjU1MDc4MSAzOS42NjE0NTMsNDQgNDAuMjEyMjM0LDQ0IEM0MC43NjMwMTUsNDQgNDEuMjEyMjM0LDQzLjU1MDc4MSA0MS4yMTIyMzQsNDMgQzQxLjIxMjIzNCw0Mi40NDkyMTkgNDAuNzYzMDE1LDQyIDQwLjIxMjIzNCw0MiBaIE00NC4yMTIyMzQsNDIgQzQzLjY2MTQ1Myw0MiA0My4yMTIyMzQsNDIuNDQ5MjE5IDQzLjIxMjIzNCw0MyBDNDMuMjEyMjM0LDQzLjU1MDc4MSA0My42NjE0NTMsNDQgNDQuMjEyMjM0LDQ0IEM0NC43NjMwMTUsNDQgNDUuMjEyMjM0LDQzLjU1MDc4MSA0NS4yMTIyMzQsNDMgQzQ1LjIxMjIzNCw0Mi40NDkyMTkgNDQuNzYzMDE1LDQyIDQ0LjIxMjIzNCw0MiBaIE00OC4yMTIyMzQsNDIgQzQ3LjY2MTQ1Myw0MiA0Ny4yMTIyMzQsNDIuNDQ5MjE5IDQ3LjIxMjIzNCw0MyBDNDcuMjEyMjM0LDQzLjU1MDc4MSA0Ny42NjE0NTMsNDQgNDguMjEyMjM0LDQ0IEM0OC43NjMwMTUsNDQgNDkuMjEyMjM0LDQzLjU1MDc4MSA0OS4yMTIyMzQsNDMgQzQ5LjIxMjIzNCw0Mi40NDkyMTkgNDguNzYzMDE1LDQyIDQ4LjIxMjIzNCw0MiBaIE01Mi4yMTIyMzQsNDIgQzUxLjY2MTQ1Myw0MiA1MS4yMTIyMzQsNDIuNDQ5MjE5IDUxLjIxMjIzNCw0MyBDNTEuMjEyMjM0LDQzLjU1MDc4MSA1MS42NjE0NTMsNDQgNTIuMjEyMjM0LDQ0IEM1Mi43NjMwMTUsNDQgNTMuMjEyMjM0LDQzLjU1MDc4MSA1My4yMTIyMzQsNDMgQzUzLjIxMjIzNCw0Mi40NDkyMTkgNTIuNzYzMDE1LDQyIDUyLjIxMjIzNCw0MiBaIiBpZD0iaW5zaWRlX3JhbmdlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_between()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">[1, 10]</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">11<br />
0.85</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.15</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
QwMiAyMi44NzY1OTI3LDQxLjIzNDUwMzEgMjQuNjk3NzA0LDQyLjQ2NTA4MTMgQzI2LjUxNzY0MzQsNDMuNjk1NzU3MiAyOC43MjI3NDI3LDQ0LjQxNTU4ODMgMzEuMDgxNDU2OCw0NC40MTU1ODgzIEw0NC4xMjc4NzE0LDQ0LjQxNTU4ODMgQzQ1LjAyODQ2Niw0NC40MTU1ODgzIDQ1Ljc1ODY0ODgsNDMuNjg1NDA1NSA0NS43NTg2NDg4LDQyLjc4NDgxMDkgQzQ1Ljc1ODY0ODgsNDEuODg0MjE2MyA0NS4wMjg1NjM2LDQxLjE1MzgzODIgNDQuMTI3OTY5LDQxLjE1MzgzODIgWiIgaWQ9InNldF9vZiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_in_set()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">f</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">low, mid, high</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">13<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody>
</table>


This approach is particularly useful for storing validation configurations as part of your data pipeline or version control system, allowing you to maintain validation rules alongside your code.


#### Governance Metadata

YAML workflows support governance metadata via `owner`, `consumers`, and `version` top-level keys. These are forwarded to the [Validate](Validate.md#pointblank.Validate) constructor and embedded in the validation report:


``` python
yaml_config = '''
tbl: small_table
tbl_name: sales_pipeline
owner: Data Engineering
consumers: [Analytics, Finance, Compliance]
version: "2.1.0"
steps:
- col_vals_not_null:
    columns: [a, b]
'''

result = pb.yaml_interrogate(yaml_config)
print(f"Owner: {result.owner}")
print(f"Consumers: {result.consumers}")
print(f"Version: {result.version}")
```


    Owner: Data Engineering
    Consumers: ['Analytics', 'Finance', 'Compliance']
    Version: 2.1.0


------------------------------------------------------------------------


#### Aggregate Validations

YAML supports aggregate validation methods for checking column-level statistics. These methods validate that a column's sum, average, or standard deviation meets a threshold:


``` python
yaml_config = '''
tbl: small_table
steps:
- col_sum_gt:
    columns: [d]
    value: 0
- col_avg_le:
    columns: [a]
    value: 10
'''

result = pb.yaml_interrogate(yaml_config)
result
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">STEP</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALUES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASS</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FAIL</th>
<th id="pb_tbl-w_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">W</th>
<th id="pb_tbl-e_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">E</th>
<th id="pb_tbl-c_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">C</th>
<th id="pb_tbl-extract_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EXT</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc3VtX2d0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9zdW1fZ3QiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InN1bSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTguODg5MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCw3LjE0IEMyLjY2OTMzMzMzLDcuMTQgMi4wMyw3LjAxNjMzMzMzIDEuNDcsNi43NjkgQzAuOTEsNi41MjE2NjY2NyAwLjQ2NjY2NjY2Nyw2LjE4OCAwLjE0LDUuNzY4IEMwLjA0NjY2NjY2NjcsNS42NDY2NjY2NyAwLDUuNTI1MzMzMzMgMCw1LjQwNCBDMCw1LjE5ODY2NjY3IDAuMDk4LDUuMDQ0NjY2NjcgMC4yOTQsNC45NDIgQzAuMzc4LDQuODk1MzMzMzMgMC40NjY2NjY2NjcsNC44NzIgMC41Niw0Ljg3MiBDMC43MTg2NjY2NjcsNC44NzIgMC44ODIsNC45NTYgMS4wNSw1LjEyNCBDMS4zNTgsNS40NTA2NjY2NyAxLjY5ODY2NjY3LDUuNjg4NjY2NjcgMi4wNzIsNS44MzggQzIuNDQ1MzMzMzMsNS45ODczMzMzMyAyLjg3NDY2NjY3LDYuMDYyIDMuMzYsNi4wNjIgQzMuODA4LDYuMDYyIDQuMTc0MzMzMzMsNS45ODk2NjY2NyA0LjQ1OSw1Ljg0NSBDNC43NDM2NjY2Nyw1LjcwMDMzMzMzIDQuODg2LDUuNTAyIDQuODg2LDUuMjUgQzQuODg2LDUuMDYzMzMzMzMgNC44MzIzMzMzMyw0LjkwOTMzMzMzIDQuNzI1LDQuNzg4IEM0LjYxNzY2NjY3LDQuNjY2NjY2NjcgNC40MzMzMzMzMyw0LjU1IDQuMTcyLDQuNDM4IEMzLjkxMDY2NjY3LDQuMzI2IDMuNSw0LjE3NjY2NjY3IDIuOTQsMy45OSBDMS45OTczMzMzMywzLjY5MTMzMzMzIDEuMzMyMzMzMzMsMy4zODMzMzMzMyAwLjk0NSwzLjA2NiBDMC41NTc2NjY2NjcsMi43NDg2NjY2NyAwLjM2NCwyLjMzMzMzMzMzIDAuMzY0LDEuODIgQzAuMzY0LDEuMjY5MzMzMzMgMC42MDQzMzMzMzMsMC44MjgzMzMzMzMgMS4wODUsMC40OTcgQzEuNTY1NjY2NjcsMC4xNjU2NjY2NjcgMi4yMTY2NjY2NywwIDMuMDM4LDAgQzMuNjQ0NjY2NjcsMCA0LjIwNywwLjEwMDMzMzMzMyA0LjcyNSwwLjMwMSBDNS4yNDMsMC41MDE2NjY2NjcgNS42MjgsMC43NzQ2NjY2NjcgNS44OCwxLjEyIEM1Ljk2NCwxLjIzMiA2LjAwNiwxLjM0ODY2NjY3IDYuMDA2LDEuNDcgQzYuMDA2LDEuNjI4NjY2NjcgNS45MjIsMS43NjQgNS43NTQsMS44NzYgQzUuNjMyNjY2NjcsMS45NTA2NjY2NyA1LjUxMTMzMzMzLDEuOTg4IDUuMzksMS45ODggQzUuMjAzMzMzMzMsMS45ODggNS4wMjYsMS45MDQgNC44NTgsMS43MzYgQzQuNjM0LDEuNTEyIDQuMzc1LDEuMzQ2MzMzMzMgNC4wODEsMS4yMzkgQzMuNzg3LDEuMTMxNjY2NjcgMy40MzQ2NjY2NywxLjA3OCAzLjAyNCwxLjA3OCBDMi41ODUzMzMzMywxLjA3OCAyLjI1MTY2NjY3LDEuMTQzMzMzMzMgMi4wMjMsMS4yNzQgQzEuNzk0MzMzMzMsMS40MDQ2NjY2NyAxLjY4LDEuNTkxMzMzMzMgMS42OCwxLjgzNCBDMS42OCwyLjAyMDY2NjY3IDEuNzMxMzMzMzMsMi4xNyAxLjgzNCwyLjI4MiBDMS45MzY2NjY2NywyLjM5NCAyLjExNCwyLjUwMTMzMzMzIDIuMzY2LDIuNjA0IEMyLjYxOCwyLjcwNjY2NjY3IDMuMDM4LDIuODUxMzMzMzMgMy42MjYsMy4wMzggQzQuMjc5MzMzMzMsMy4yNDMzMzMzMyA0Ljc5MDMzMzMzLDMuNDQ4NjY2NjcgNS4xNTksMy42NTQgQzUuNTI3NjY2NjcsMy44NTkzMzMzMyA1Ljc5MTMzMzMzLDQuMDg4IDUuOTUsNC4zNCBDNi4xMDg2NjY2Nyw0LjU5MiA2LjE4OCw0Ljg5NTMzMzMzIDYuMTg4LDUuMjUgQzYuMTg4LDUuODE5MzMzMzMgNS45MzM2NjY2Nyw2LjI3NjY2NjY3IDUuNDI1LDYuNjIyIEM0LjkxNjMzMzMzLDYuOTY3MzMzMzMgNC4yMzczMzMzMyw3LjE0IDMuMzg4LDcuMTQgWiIgaWQ9InMiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTAuNTU2LDcuMTQgQzkuNjIyNjY2NjcsNy4xNCA4Ljg5NDY2NjY3LDYuODcxNjY2NjcgOC4zNzIsNi4zMzUgQzcuODQ5MzMzMzMsNS43OTgzMzMzMyA3LjU4OCw1LjA1ODY2NjY3IDcuNTg4LDQuMTE2IEw3LjU4OCwwLjY1OCBDNy41ODgsMC40NTI2NjY2NjcgNy42NDYzMzMzMywwLjI5MTY2NjY2NyA3Ljc2MywwLjE3NSBDNy44Nzk2NjY2NywwLjA1ODMzMzMzMzMgOC4wMzYsMCA4LjIzMiwwIEM4LjQzNzMzMzMzLDAgOC41OTYsMC4wNTgzMzMzMzMzIDguNzA4LDAuMTc1IEM4LjgyLDAuMjkxNjY2NjY3IDguODc2LDAuNDUyNjY2NjY3IDguODc2LDAuNjU4IEw4Ljg3Niw0LjExNiBDOC44NzYsNC43MzIgOS4wNDE2NjY2Nyw1LjIxMDMzMzMzIDkuMzczLDUuNTUxIEM5LjcwNDMzMzMzLDUuODkxNjY2NjcgMTAuMTY4NjY2Nyw2LjA2MiAxMC43NjYsNi4wNjIgQzExLjQ5NCw2LjA2MiAxMi4wNzk2NjY3LDUuODI4NjY2NjcgMTIuNTIzLDUuMzYyIEMxMi45NjYzMzMzLDQuODk1MzMzMzMgMTMuMTg4LDQuMjc0NjY2NjcgMTMuMTg4LDMuNSBMMTMuMTg4LDAuNjU4IEMxMy4xODgsMC40NTI2NjY2NjcgMTMuMjQ2MzMzMywwLjI5MTY2NjY2NyAxMy4zNjMsMC4xNzUgQzEzLjQ3OTY2NjcsMC4wNTgzMzMzMzMzIDEzLjY0MDY2NjcsMCAxMy44NDYsMCBDMTQuMDUxMzMzMywwIDE0LjIwNzY2NjcsMC4wNTYgMTQuMzE1LDAuMTY4IEMxNC40MjIzMzMzLDAuMjggMTQuNDc2LDAuNDQzMzMzMzMzIDE0LjQ3NiwwLjY1OCBMMTQuNDc2LDYuNDgyIEMxNC40NzYsNi42OTY2NjY2NyAxNC40MjIzMzMzLDYuODYgMTQuMzE1LDYuOTcyIEMxNC4yMDc2NjY3LDcuMDg0IDE0LjA1MTMzMzMsNy4xNCAxMy44NDYsNy4xNCBDMTMuNjQwNjY2Nyw3LjE0IDEzLjQ4Miw3LjA4MTY2NjY3IDEzLjM3LDYuOTY1IEMxMy4yNTgsNi44NDgzMzMzMyAxMy4yMDIsNi42ODczMzMzMyAxMy4yMDIsNi40ODIgTDEzLjIwMiw1Ljc5NiBDMTIuNTAyLDYuNjkyIDExLjYyLDcuMTQgMTAuNTU2LDcuMTQgWiIgaWQ9InUiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTYuODQyLDcuMTQgQzE2LjYzNjY2NjcsNy4xNCAxNi40NzgsNy4wODE2NjY2NyAxNi4zNjYsNi45NjUgQzE2LjI1NCw2Ljg0ODMzMzMzIDE2LjE5OCw2LjY4NzMzMzMzIDE2LjE5OCw2LjQ4MiBMMTYuMTk4LDAuNjU4IEMxNi4xOTgsMC40NTI2NjY2NjcgMTYuMjU0LDAuMjkxNjY2NjY3IDE2LjM2NiwwLjE3NSBDMTYuNDc4LDAuMDU4MzMzMzMzMyAxNi42MzY2NjY3LDAgMTYuODQyLDAgQzE3LjA0NzMzMzMsMCAxNy4yMDgzMzMzLDAuMDU4MzMzMzMzMyAxNy4zMjUsMC4xNzUgQzE3LjQ0MTY2NjcsMC4yOTE2NjY2NjcgMTcuNSwwLjQ1MjY2NjY2NyAxNy41LDAuNjU4IEwxNy41LDEuMzAyIEMxNy43MzMzMzMzLDAuODkxMzMzMzMzIDE4LjAzNjY2NjcsMC41NzE2NjY2NjcgMTguNDEsMC4zNDMgQzE4Ljc4MzMzMzMsMC4xMTQzMzMzMzMgMTkuMTgsMCAxOS42LDAgQzIwLjAyOTMzMzMsMCAyMC40MDczMzMzLDAuMTE0MzMzMzMzIDIwLjczNCwwLjM0MyBDMjEuMDYwNjY2NywwLjU3MTY2NjY2NyAyMS4zMDMzMzMzLDAuODkxMzMzMzMzIDIxLjQ2MiwxLjMwMiBDMjIuMTQzMzMzMywwLjQzNCAyMi45MTMzMzMzLDAgMjMuNzcyLDAgQzI0LjUwOTMzMzMsMCAyNS4wNzE2NjY3LDAuMjM4IDI1LjQ1OSwwLjcxNCBDMjUuODQ2MzMzMywxLjE5IDI2LjA0LDEuODgwNjY2NjcgMjYuMDQsMi43ODYgTDI2LjA0LDYuNDgyIEMyNi4wNCw2LjY4NzMzMzMzIDI1Ljk4MTY2NjcsNi44NDgzMzMzMyAyNS44NjUsNi45NjUgQzI1Ljc0ODMzMzMsNy4wODE2NjY2NyAyNS41OTIsNy4xNCAyNS4zOTYsNy4xNCBDMjUuMTkwNjY2Nyw3LjE0IDI1LjAzMiw3LjA4NCAyNC45Miw2Ljk3MiBDMjQuODA4LDYuODYgMjQuNzUyLDYuNjk2NjY2NjcgMjQuNzUyLDYuNDgyIEwyNC43NTIsMi43MyBDMjQuNzUyLDIuMTc5MzMzMzMgMjQuNjU4NjY2NywxLjc2NjMzMzMzIDI0LjQ3MiwxLjQ5MSBDMjQuMjg1MzMzMywxLjIxNTY2NjY3IDIzLjk3NzMzMzMsMS4wNzggMjMuNTQ4LDEuMDc4IEMyMi44MDEzMzMzLDEuMDc4IDIyLjIzNjY2NjcsMS41MDI2NjY2NyAyMS44NTQsMi4zNTIgTDIxLjg1NCw2LjQ4MiBDMjEuODU0LDYuNjg3MzMzMzMgMjEuNzk1NjY2Nyw2Ljg0ODMzMzMzIDIxLjY3OSw2Ljk2NSBDMjEuNTYyMzMzMyw3LjA4MTY2NjY3IDIxLjQwMTMzMzMsNy4xNCAyMS4xOTYsNy4xNCBDMjAuOTkwNjY2Nyw3LjE0IDIwLjgzNDMzMzMsNy4wODQgMjAuNzI3LDYuOTcyIEMyMC42MTk2NjY3LDYuODYgMjAuNTY2LDYuNjk2NjY2NjcgMjAuNTY2LDYuNDgyIEwyMC41NjYsMi45NjggQzIwLjU2NiwyLjM1MiAyMC40NjMzMzMzLDEuODgzIDIwLjI1OCwxLjU2MSBDMjAuMDUyNjY2NywxLjIzOSAxOS43NTQsMS4wNzggMTkuMzYyLDEuMDc4IEMxOC44MTEzMzMzLDEuMDc4IDE4LjM2MzMzMzMsMS4yOTUgMTguMDE4LDEuNzI5IEMxNy42NzI2NjY3LDIuMTYzIDE3LjUsMi43MjUzMzMzMyAxNy41LDMuNDE2IEwxNy41LDYuNDgyIEMxNy41LDYuNjg3MzMzMzMgMTcuNDQxNjY2Nyw2Ljg0ODMzMzMzIDE3LjMyNSw2Ljk2NSBDMTcuMjA4MzMzMyw3LjA4MTY2NjY3IDE3LjA0NzMzMzMsNy4xNCAxNi44NDIsNy4xNCBaIiBpZD0ibSIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuOTUyMzgxLDEwIEwxOS4wNDc2MTkxLDEwIEMxNi44MTMzMzMzLDEwIDE1LDExLjgxMzMzMzMgMTUsMTQuMDQ3NjE5IEwxNSwzOS45NTIzODEgQzE1LDQyLjE4NjY2NjcgMTYuODEzMzMzMyw0NCAxOS4wNDc2MTkxLDQ0IEw0NC45NTIzODEsNDQgQzQ3LjE4NjY2NjcsNDQgNDksNDIuMTg2NjY2NyA0OSwzOS45NTIzODEgTDQ5LDE0LjA0NzYxOSBDNDksMTEuODEzMzMzMyA0Ny4xODY2NjY3LDEwIDQ0Ljk1MjM4MSwxMCBaIE0yNi43OTQ3NjE5LDM2LjU2ODU3MTQgTDI1Ljg3MTkwNDgsMzUuMjQwOTUyNCBMMzcuODYwOTUyNCwyNyBMMjUuODcxOTA0OCwxOC43NTkwNDc2IEwyNi43OTQ3NjE5LDE3LjQzMTQyODYgTDQwLjcxODU3MTQsMjcgTDI2Ljc5NDc2MTksMzYuNTY4NTcxNCBaIiBpZD0iZ3JlYXRlcl90aGFuIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_sum_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfYXZnX2xlPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9hdmdfbGUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9ImF2ZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjAuMDU4MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yLjk0LDcuMTQgQzIuMDM0NjY2NjcsNy4xNCAxLjMxODMzMzMzLDYuOTM0NjY2NjcgMC43OTEsNi41MjQgQzAuMjYzNjY2NjY3LDYuMTEzMzMzMzMgMCw1LjU1OCAwLDQuODU4IEMwLDQuMTU4IDAuMjc3NjY2NjY3LDMuNjAyNjY2NjcgMC44MzMsMy4xOTIgQzEuMzg4MzMzMzMsMi43ODEzMzMzMyAyLjE0NjY2NjY3LDIuNTc2IDMuMTA4LDIuNTc2IEMzLjYxMiwyLjU3NiA0LjA2MjMzMzMzLDIuNjQxMzMzMzMgNC40NTksMi43NzIgQzQuODU1NjY2NjcsMi45MDI2NjY2NyA1LjE3NTMzMzMzLDMuMDggNS40MTgsMy4zMDQgTDUuNDE4LDIuNjA0IEM1LjQxOCwyLjEgNS4yNTcsMS43MTUgNC45MzUsMS40NDkgQzQuNjEzLDEuMTgzIDQuMTM5MzMzMzMsMS4wNSAzLjUxNCwxLjA1IEMyLjY4MzMzMzMzLDEuMDUgMS45NjQ2NjY2NywxLjI0MTMzMzMzIDEuMzU4LDEuNjI0IEMxLjE4MDY2NjY3LDEuNzI2NjY2NjcgMS4wMjY2NjY2NywxLjc3OCAwLjg5NiwxLjc3OCBDMC43NzQ2NjY2NjcsMS43NzggMC42NTMzMzMzMzMsMS43MjY2NjY2NyAwLjUzMiwxLjYyNCBDMC40MjkzMzMzMzMsMS41MjEzMzMzMyAwLjM3OCwxLjQgMC4zNzgsMS4yNiBDMC4zNzgsMS4xMDEzMzMzMyAwLjQ1NzMzMzMzMywwLjk1NjY2NjY2NyAwLjYxNiwwLjgyNiBDMC45NTIsMC41NzQgMS4zNzY2NjY2NywwLjM3MzMzMzMzMyAxLjg5LDAuMjI0IEMyLjQwMzMzMzMzLDAuMDc0NjY2NjY2NyAyLjk0LDAgMy41LDAgQzQuNTE3MzMzMzMsMCA1LjMxMDY2NjY3LDAuMjMxIDUuODgsMC42OTMgQzYuNDQ5MzMzMzMsMS4xNTUgNi43MzQsMS44MDEzMzMzMyA2LjczNCwyLjYzMiBMNi43MzQsNi41MSBDNi43MzQsNi43MTUzMzMzMyA2LjY4MDMzMzMzLDYuODcxNjY2NjcgNi41NzMsNi45NzkgQzYuNDY1NjY2NjcsNy4wODYzMzMzMyA2LjMwOTMzMzMzLDcuMTQgNi4xMDQsNy4xNCBDNS44OTg2NjY2Nyw3LjE0IDUuNzQsNy4wODYzMzMzMyA1LjYyOCw2Ljk3OSBDNS41MTYsNi44NzE2NjY2NyA1LjQ2LDYuNzI0NjY2NjcgNS40Niw2LjUzOCBMNS40Niw2LjE0NiBDNC45LDYuODA4NjY2NjcgNC4wNiw3LjE0IDIuOTQsNy4xNCBaIE0zLjE2NCw2LjE0NiBDMy44NDUzMzMzMyw2LjE0NiA0LjM5MzY2NjY3LDYuMDIyMzMzMzMgNC44MDksNS43NzUgQzUuMjI0MzMzMzMsNS41Mjc2NjY2NyA1LjQzMiw1LjIwMzMzMzMzIDUuNDMyLDQuODAyIEM1LjQzMiw0LjQxIDUuMjMzNjY2NjcsNC4xMDQzMzMzMyA0LjgzNywzLjg4NSBDNC40NDAzMzMzMywzLjY2NTY2NjY3IDMuODkyLDMuNTU2IDMuMTkyLDMuNTU2IEMyLjU4NTMzMzMzLDMuNTU2IDIuMTE4NjY2NjcsMy42NjggMS43OTIsMy44OTIgQzEuNDY1MzMzMzMsNC4xMTYgMS4zMDIsNC40MzggMS4zMDIsNC44NTggQzEuMzAyLDUuMjY4NjY2NjcgMS40NjMsNS41ODYgMS43ODUsNS44MSBDMi4xMDcsNi4wMzQgMi41NjY2NjY2Nyw2LjE0NiAzLjE2NCw2LjE0NiBaIiBpZD0iYSIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS40OTQsNy4xNCBDMTAuNTMyNjY2Nyw3LjE0IDkuODUxMzMzMzMsNi41NjEzMzMzMyA5LjQ1LDUuNDA0IEw3Ljc5OCwwLjc4NCBDNy43NywwLjcwOTMzMzMzMyA3Ljc1NiwwLjY0NCA3Ljc1NiwwLjU4OCBDNy43NTYsMC40NjY2NjY2NjcgNy43OTgsMC4zNTcgNy44ODIsMC4yNTkgQzcuOTY2LDAuMTYxIDguMDc4LDAuMDg4NjY2NjY2NyA4LjIxOCwwLjA0MiBDOC4zMDIsMC4wMTQgOC4zODYsMCA4LjQ3LDAgQzguNjEsMCA4LjczMTMzMzMzLDAuMDM1IDguODM0LDAuMTA1IEM4LjkzNjY2NjY3LDAuMTc1IDkuMDA2NjY2NjcsMC4yNzA2NjY2NjcgOS4wNDQsMC4zOTIgTDEwLjczOCw1LjEzOCBDMTAuODU5MzMzMyw1LjQ4MzMzMzMzIDEwLjk3Niw1LjcyNiAxMS4wODgsNS44NjYgQzExLjIsNi4wMDYgMTEuMzM1MzMzMyw2LjA3NiAxMS40OTQsNi4wNzYgQzExLjY1MjY2NjcsNi4wNzYgMTEuNzg4LDYuMDA4MzMzMzMgMTEuOSw1Ljg3MyBDMTIuMDEyLDUuNzM3NjY2NjcgMTIuMTI4NjY2Nyw1LjUwMiAxMi4yNSw1LjE2NiBMMTMuOTQ0LDAuMzkyIEMxMy45OTA2NjY3LDAuMjcwNjY2NjY3IDE0LjA2MywwLjE3NzMzMzMzMyAxNC4xNjEsMC4xMTIgQzE0LjI1OSwwLjA0NjY2NjY2NjcgMTQuMzc4LDAuMDE0IDE0LjUxOCwwLjAxNCBDMTQuNzMyNjY2NywwLjAxNCAxNC45MDUzMzMzLDAuMDcyMzMzMzMzMyAxNS4wMzYsMC4xODkgQzE1LjE2NjY2NjcsMC4zMDU2NjY2NjcgMTUuMjMyLDAuNDQzMzMzMzMzIDE1LjIzMiwwLjYwMiBDMTUuMjMyLDAuNjg2IDE1LjIyMjY2NjcsMC43NDY2NjY2NjcgMTUuMjA0LDAuNzg0IEwxMy41MzgsNS40MzIgQzEzLjMyMzMzMzMsNi4wMTA2NjY2NyAxMy4wNTAzMzMzLDYuNDQgMTIuNzE5LDYuNzIgQzEyLjM4NzY2NjcsNyAxMS45NzkzMzMzLDcuMTQgMTEuNDk0LDcuMTQgWiIgaWQ9InYiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTkuODEsMTAuNTE0IEMxOS4xMTkzMzMzLDEwLjUxNCAxOC40OTE2NjY3LDEwLjQxMTMzMzMgMTcuOTI3LDEwLjIwNiBDMTcuMzYyMzMzMywxMC4wMDA2NjY3IDE2LjkwNzMzMzMsOS43MTEzMzMzMyAxNi41NjIsOS4zMzggQzE2LjQ2ODY2NjcsOS4yMzUzMzMzMyAxNi40MjIsOS4xMjMzMzMzMyAxNi40MjIsOS4wMDIgQzE2LjQyMiw4Ljg0MzMzMzMzIDE2LjUxMDY2NjcsOC42ODkzMzMzMyAxNi42ODgsOC41NCBDMTYuNzcyLDguNDY1MzMzMzMgMTYuODc5MzMzMyw4LjQyOCAxNy4wMSw4LjQyOCBDMTcuMjA2LDguNDI4IDE3LjM5MjY2NjcsOC41MjEzMzMzMyAxNy41Nyw4LjcwOCBDMTcuNzk0LDguOTQxMzMzMzMgMTguMDg1NjY2Nyw5LjEyNTY2NjY3IDE4LjQ0NSw5LjI2MSBDMTguODA0MzMzMyw5LjM5NjMzMzMzIDE5LjIyNjY2NjcsOS40NjQgMTkuNzEyLDkuNDY0IEMyMC41NjEzMzMzLDkuNDY0IDIxLjIxNDY2NjcsOS4yNDQ2NjY2NyAyMS42NzIsOC44MDYgQzIyLjEyOTMzMzMsOC4zNjczMzMzMyAyMi4zNTMzMzMzLDcuNzQyIDIyLjM0NCw2LjkzIEwyMi4zNDQsNS44NjYgQzIyLjAyNjY2NjcsNi4yNjczMzMzMyAyMS42MjUzMzMzLDYuNTg3IDIxLjE0LDYuODI1IEMyMC42NTQ2NjY3LDcuMDYzIDIwLjA4NTMzMzMsNy4xODIgMTkuNDMyLDcuMTgyIEMxOC4yNTYsNy4xODIgMTcuMzQ2LDYuODY3IDE2LjcwMiw2LjIzNyBDMTYuMDU4LDUuNjA3IDE1LjczNiw0Ljc1MDY2NjY3IDE1LjczNiwzLjY2OCBDMTUuNzM2LDIuNTU3MzMzMzMgMTYuMDYyNjY2NywxLjY2ODMzMzMzIDE2LjcxNiwxLjAwMSBDMTcuMzY5MzMzMywwLjMzMzY2NjY2NyAxOC4zMDI2NjY3LDAgMTkuNTE2LDAgQzIwLjA5NDY2NjcsMCAyMC42MjIsMC4xMTY2NjY2NjcgMjEuMDk4LDAuMzUgQzIxLjU3NCwwLjU4MzMzMzMzMyAyMS45ODkzMzMzLDAuOTE0NjY2NjY3IDIyLjM0NCwxLjM0NCBMMjIuMzQ0LDAuNjMgQzIyLjM0NCwwLjQ0MzMzMzMzMyAyMi40MDcsMC4yOTE2NjY2NjcgMjIuNTMzLDAuMTc1IEMyMi42NTksMC4wNTgzMzMzMzMzIDIyLjgxNTMzMzMsMCAyMy4wMDIsMCBDMjMuMTk4LDAgMjMuMzU0MzMzMywwLjA1NiAyMy40NzEsMC4xNjggQzIzLjU4NzY2NjcsMC4yOCAyMy42NDYsMC40MzQgMjMuNjQ2LDAuNjMgTDIzLjY0Niw2LjkzIEMyMy42NDYsNy42NDg2NjY2NyAyMy40ODczMzMzLDguMjc4NjY2NjcgMjMuMTcsOC44MiBDMjIuODUyNjY2Nyw5LjM2MTMzMzMzIDIyLjQwNDY2NjcsOS43NzkgMjEuODI2LDEwLjA3MyBDMjEuMjQ3MzMzMywxMC4zNjcgMjAuNTc1MzMzMywxMC41MTQgMTkuODEsMTAuNTE0IFogTTE5LjYxNCw2LjA5IEMyMC4xMDg2NjY3LDYuMDkgMjAuNTYzNjY2Nyw1Ljk5NjY2NjY3IDIwLjk3OSw1LjgxIEMyMS4zOTQzMzMzLDUuNjIzMzMzMzMgMjEuNzI1NjY2Nyw1LjM0MzMzMzMzIDIxLjk3Myw0Ljk3IEMyMi4yMjAzMzMzLDQuNTk2NjY2NjcgMjIuMzQ0LDQuMTM5MzMzMzMgMjIuMzQ0LDMuNTk4IEMyMi4zNDQsMi44NTEzMzMzMyAyMi4xMDYsMi4yNDQ2NjY2NyAyMS42MywxLjc3OCBDMjEuMTU0LDEuMzExMzMzMzMgMjAuNDk2LDEuMDc4IDE5LjY1NiwxLjA3OCBDMTguODI1MzMzMywxLjA3OCAxOC4xODEzMzMzLDEuMzA0MzMzMzMgMTcuNzI0LDEuNzU3IEMxNy4yNjY2NjY3LDIuMjA5NjY2NjcgMTcuMDM4LDIuODQ2NjY2NjcgMTcuMDM4LDMuNjY4IEMxNy4wMzgsNC40MTQ2NjY2NyAxNy4yNTk2NjY3LDUuMDA1IDE3LjcwMyw1LjQzOSBDMTguMTQ2MzMzMyw1Ljg3MyAxOC43ODMzMzMzLDYuMDkgMTkuNjE0LDYuMDkgWiIgaWQ9ImciIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPHBhdGggZD0iTTQ2LjY4MTgxODIsMTAgTDE3LjMxODE4MTgsMTAgQzE2LjA0MTM3MDYsMTAgMTUsMTEuMDQxMzcwNSAxNSwxMi4zMTgxODE4IEwxNSw0MS42ODE4MTgyIEMxNSw0Mi45NTg2Mjk1IDE2LjA0MTM3MDYsNDQgMTcuMzE4MTgxOCw0NCBMNDYuNjgxODE4Miw0NCBDNDcuOTU4NjI5NSw0NCA0OSw0Mi45NTg2Mjk1IDQ5LDQxLjY4MTgxODIgTDQ5LDEyLjMxODE4MTggQzQ5LDExLjA0MTM3MDUgNDcuOTU4NjI5NSwxMCA0Ni42ODE4MTgyLDEwIFogTTM4LjU4MDI1NTcsMTYuMjc4NDA5MSBMMzkuMzI4ODM1MiwxNy42MzA2ODE4IEwyNi42MzYxODYzLDI0LjY4MTgxODIgTDM5LjMyODgzNTIsMzEuNzMyOTU0NSBMMzguNTgwMjU1NywzMy4wODUyMjczIEwyMy40NTQ3MjI4LDI0LjY4MTgxODIgTDM4LjU4MDI1NTcsMTYuMjc4NDA5MSBaIE00MC41LDM3LjgxODE4MTggTDIzLjUsMzcuODE4MTgxOCBMMjMuNSwzNi4yNzI3MjczIEw0MC41LDM2LjI3MjcyNzMgTDQwLjUsMzcuODE4MTgxOCBaIiBpZD0ibGVzc190aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_avg_le()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">10</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody>
</table>


The 15 available aggregate methods follow the pattern `col_{stat}_{comparator}` where `{stat}` is `sum`, `avg`, or `sd` and `{comparator}` is `gt`, `lt`, `ge`, `le`, or `eq`.

------------------------------------------------------------------------


#### Data Freshness

Check that a date/datetime column has recent data using [data_freshness](Validate.data_freshness.md#pointblank.Validate.data_freshness):

``` yaml
tbl: events.csv
steps:
- data_freshness:
    columns: event_date
    freshness: "24h"
```

------------------------------------------------------------------------


#### Active Parameter Shortcut

The `active=` parameter controls whether a validation step runs. It supports boolean values and Python expression shortcuts:

``` yaml
steps:
- col_vals_gt:
    columns: [d]
    value: 100
    active: false            # Skip this step

- col_vals_not_null:
    columns: [a]
    active: true             # Always run (default)
```

------------------------------------------------------------------------


#### Null Percentage Check

Use [col_pct_null](Validate.col_pct_null.md#pointblank.Validate.col_pct_null) to validate that the percentage of null values in a column is within bounds:

``` yaml
steps:
- col_pct_null:
    columns: [a, b]
    value: 0.05
```

------------------------------------------------------------------------


#### Using `set_tbl=` to Override the Table

The `set_tbl=` parameter allows you to override the table specified in the YAML configuration. This is useful when you have a template validation workflow but want to apply it to different tables:


``` python
import polars as pl

# Create a test table with similar structure to small_table
test_table = pl.DataFrame({
    "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    "a": [1, 2, 3],
    "b": ["1-abc-123", "2-def-456", "3-ghi-789"],
    "d": [150, 200, 250]
})

# Use the same YAML config but apply it to our test table
yaml_config = '''
tbl: small_table  # This will be overridden
tbl_name: Test Table  # This name will be used
steps:
- col_exists:
    columns: [date, a, b, d]
- col_vals_gt:
    columns: [d]
    value: 100
'''

# Execute with table override
result = pb.yaml_interrogate(yaml_config, set_tbl=test_table)
print(f"Validation applied to: {result.tbl_name}")
result
```


    Validation applied to: Test Table


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">STEP</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALUES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASS</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FAIL</th>
<th id="pb_tbl-w_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">W</th>
<th id="pb_tbl-e_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">E</th>
<th id="pb_tbl-c_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">C</th>
<th id="pb_tbl-extract_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EXT</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">4</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">5</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">3</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody>
</table>


This feature makes YAML configurations more reusable and flexible, allowing you to define validation logic once and apply it to multiple similar tables.
