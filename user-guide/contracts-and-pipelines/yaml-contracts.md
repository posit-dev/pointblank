# YAML Contracts & Pipelines

Contracts and pipelines are designed to be **data, not code**. They serialize cleanly to YAML, making them useful for version control, team collaboration, and configuration-driven workflows. This guide covers how to write, load, and use YAML-based contracts and pipelines.

When you define a contract in Python, it lives inside your application code. That's fine for quick validation tasks, but in production data platforms you often want the *specification* of what data should look like to live separately from the *code* that processes it. YAML contracts give you exactly that separation with a human-readable, machine-parseable definition of data expectations that anyone on your team can read, review, and modify.


# Why YAML?

YAML contracts offer several advantages over defining contracts purely in Python:

- **Version-controllable**: track contract changes alongside your code in Git, with clear diffs when expectations evolve
- **Language-agnostic**: contracts can be understood (and even authored) by non-Python team members such as data analysts, product managers, or compliance officers
- **Separates concerns**: validation rules live apart from transform logic, meaning you can update expectations without touching application code
- **CI/CD friendly**: load and enforce contracts in automated pipelines and gate deployments on contract compliance
- **Reviewable**: pull requests that change data contracts are easy to review because the YAML format highlights exactly what changed in a readable way

The YAML approach doesn't replace the Python API but, rather, it complements it. Use YAML for stable, well-defined contracts that change infrequently, and Python for exploratory validation or complex logic that benefits from programmatic construction.


# Contract YAML Format

Every contract YAML file uses a `contract:` top-level key as its root element. Inside this key you define all the metadata, schema, and validation steps that make up your contract. The structure is designed to be self-documenting so someone reading the file should immediately understand what data this contract governs, who owns it, and what rules are enforced.

Here's a complete example showing all the available fields:

``` yaml
contract:
  name: customer_records
  direction: source
  version: "1.2.0"
  owner: data-platform-team
  consumers:
    - analytics-team
    - ml-team
  description: "Raw customer data from the CRM system"
  on_violation: warn

  schema:
    customer_id: String
    email: String
    name: String
    signup_date: String
    plan: String

  steps:
    - col_vals_not_null:
        columns:
          - customer_id
          - email
    - col_vals_regex:
        columns: email
        pattern: "^[^@]+@[^@]+\\.[^@]+$"
    - rows_distinct:
        columns_subset:
          - customer_id
    - col_vals_in_set:
        columns: plan
        set:
          - free
          - pro
          - enterprise

  thresholds:
    warning: 0.01
    error: 0.05
    critical: 0.10
```


## Structure Reference

The table below summarizes every key available inside the `contract:` block. Only `name` is required and everything else has sensible defaults or can be omitted if not applicable to your use case.

| Key | Required | Description |
|----|----|----|
| `name` | Yes | Human-readable identifier for this contract |
| [direction](../../reference/Contract.md#pointblank.Contract.direction) | No | `"source"` (default) or `"target"`: indicates where in a pipeline this contract applies |
| `version` | No | Semantic version string (e.g., `"1.2.0"`) for tracking contract evolution over time |
| `owner` | No | Team or individual responsible for maintaining this contract |
| `consumers` | No | List of teams or systems that depend on data conforming to this contract |
| `description` | No | Longer prose describing the contract's purpose and context |
| [on_violation](../../reference/Contract.md#pointblank.Contract.on_violation) | No | Behavior when validation fails: `"warn"` (default), `"raise"`, or `"log"` |
| `schema` | No | Column name -\> data type mapping; triggers a [col_schema_match()](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) check |
| `steps` | No | Ordered list of validation steps to execute |
| `thresholds` | No | Threshold levels ([warning](../../reference/Validate.warning.md#pointblank.Validate.warning), [error](../../reference/Validate.error.md#pointblank.Validate.error), [critical](../../reference/Validate.critical.md#pointblank.Validate.critical)) as fractions or absolute counts |


## Step Syntax in YAML

Steps are the heart of a contract. They define the actual validation rules that will be checked against your data. In YAML, each step is a list item whose key is the name of a Pointblank validation method (like [col_vals_gt](../../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) or [rows_distinct](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct)), and whose value is a dictionary of the arguments to pass to that method.

This mapping is direct: the YAML step `col_vals_gt: {columns: amount, value: 0}` is exactly equivalent to calling `.col_vals_gt(columns="amount", value=0)` in Python. Here are examples covering the most common patterns:

``` yaml
steps:
  # Step with keyword arguments
  - col_vals_gt:
      columns: amount
      value: 0

  # Step with a list of columns
  - col_vals_not_null:
      columns:
        - id
        - name
        - email

  # Step with multiple arguments
  - col_vals_between:
      columns: age
      left: 0
      right: 150

  # Step with a set/list of values
  - col_vals_in_set:
      columns: status
      set:
        - active
        - inactive
        - suspended

  # Step with no arguments
  - rows_complete: {}
```


# Writing and Loading Contracts

Pointblank supports a complete round-trip workflow: define contracts in Python, save them to YAML for version control and sharing, then load them back in any environment. This section walks through each step of that workflow.


## Saving to YAML

The `.to_yaml()` method serializes a [Contract](../../reference/Contract.md#pointblank.Contract) object to a YAML string. You can optionally pass a file path to write directly to disk. The resulting YAML is clean, readable, and contains all the information needed to reconstruct the contract exactly.


``` python
import pointblank as pb
from pathlib import Path

# Create a contract in Python
contract = pb.Contract(
    name="product_catalog",
    direction="source",
    version="1.0.0",
    owner="catalog-team",
    schema=pb.Schema(
        product_id="String",
        name="String",
        price="Float64",
        category="String",
        in_stock="Boolean",
    ),
    steps=[
        pb.Step("col_vals_not_null", columns=["product_id", "name", "price"]),
        pb.Step("col_vals_gt", columns="price", value=0),
        pb.Step("col_vals_in_set", columns="category", set=["electronics", "clothing", "food", "home"]),
        pb.Step("rows_distinct", columns_subset=["product_id"]),
    ],
    thresholds=pb.Thresholds(warning=0.01, error=0.05),
)

# Save to YAML
yaml_output = contract.to_yaml("product_contract.yaml")
print(yaml_output)
```


    contract:
      name: product_catalog
      version: 1.0.0
      owner: catalog-team
      schema:
        product_id: String
        name: String
        price: Float64
        category: String
        in_stock: Boolean
      steps:
      - col_vals_not_null:
          columns:
          - product_id
          - name
          - price
      - col_vals_gt:
          columns: price
          value: 0
      - col_vals_in_set:
          columns: category
          set:
          - electronics
          - clothing
          - food
          - home
      - rows_distinct:
          columns_subset:
          - product_id
      thresholds:
        warning: 0.01
        error: 0.05


## Loading from YAML

The [Contract.from_yaml()](../../reference/Contract.md#pointblank.Contract.from_yaml) class method reads a YAML file and reconstructs the full [Contract](../../reference/Contract.md#pointblank.Contract) object, including schema, steps, thresholds, and all metadata. The loaded contract is identical in behavior to one created directly in Python (there's no functional difference).


``` python
# Load the contract back
loaded_contract = pb.Contract.from_yaml("product_contract.yaml")
print(loaded_contract)
print(f"Steps: {len(loaded_contract.steps)}")
print(f"Version: {loaded_contract.version}")
```


    Contract(name='product_catalog', direction='source', version='1.0.0', schema=<defined>, steps=4)
    Steps: 4
    Version: 1.0.0


## Validating with a Loaded Contract

Once loaded, a contract works exactly like one defined in Python. Call `.validate(data)` to compile it into a [Validate](../../reference/Validate.md#pointblank.Validate) object, execute all checks, and get the standard Pointblank validation report. This is the core value proposition of YAML contracts: define once, validate anywhere.


``` python
import polars as pl

products = pl.DataFrame(
    {
        "product_id": ["P001", "P002", "P003", "P004", "P005"],
        "name": ["Laptop", "T-Shirt", "Coffee Beans", "Desk Lamp", "Headphones"],
        "price": [999.99, 29.99, 12.50, 45.00, 79.99],
        "category": ["electronics", "clothing", "food", "home", "electronics"],
        "in_stock": [True, True, False, True, True],
    }
)

loaded_contract.validate(products)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Contract: product_catalog v1.0.0</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">Polars</span><span style="background-color: none; color: #222222; padding: 0.5em 0.5em; position: inherit; margin: 5px 10px 5px -4px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">product_catalog</span><span><span style="background-color: #AAAAAA; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; border: solid 1px #AAAAAA; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">WARNING</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #AAAAAA; padding: 2px 15px 2px 15px; font-size: smaller; margin-right: 5px;">0.01</span><span style="background-color: #EBBC14; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #EBBC14; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">ERROR</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #EBBC14; padding: 2px 15px 2px 15px; font-size: smaller; margin-right: 5px;">0.05</span><span style="background-color: #FF3300; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #FF3300; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">CRITICAL</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #FF3300; padding: 2px 15px 2px 15px; font-size: smaller;">--</span></span>

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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3NjaGVtYV9tYXRjaDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfc2NoZW1hX21hdGNoIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4zMTAzNDUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01My43MTIyMzQsMzkuNzg4NTI2OCBMNTQuMjEyMjM0LDU2LjI4ODUyNjggTDQyLjIxMjIzNCw1Ni43ODg1MjY4IEw0Mi4yMTIyMzQsMzkuNzg4NTI2OCBMNTMuNzEyMjM0LDM5Ljc4ODUyNjggWiBNMzkuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCw1Ni43ODg1MjY4IEwyNy43MTIyMzQsNTYuNzg4NTI2OCBMMjcuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCwzOS43ODg1MjY4IFogTTI1LjIxMjIzNCwzOS43ODg1MjY4IEwyNS4yMTIyMzQsNTYuNzg4NTI2OCBMMTMuNzEyMjM0LDU2Ljc4ODUyNjggTDEzLjIxMjIzNCw0MC4yODg1MjY4IEwyNS4yMTIyMzQsMzkuNzg4NTI2OCBaIiBpZD0iY29sdW1uc19zY2hlbWEiIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8ZyBpZD0idmVydGljYWxfZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLjAwMDAwMCwgMjkuMDAwMDAwKSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIyLjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSIyLjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUiPjwvbGluZT4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI1LjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSI1LjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUtQ29weSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MS43MTIyMzQsOS4wMTQ2NjkzNSBMNDEuNzEyMjM0LDI3LjAxNDY2OTQgTDUzLjcxMjIzNCwyNy4wMTQ2Njk0IEM1NC4yNjIyMzQsMjcuMDE0NjY5NCA1NC43MTIyMzQsMjYuNTY0NjY5NCA1NC43MTIyMzQsMjYuMDE0NjY5NCBMNTQuNzEyMjM0LDEwLjAxNDY2OTQgQzU0LjcxMjIzNCw5LjQ2NDY2OTM1IDU0LjI2MjIzNCw5LjAxNDY2OTM1IDUzLjcxMjIzNCw5LjAxNDY2OTM1IEw0MS43MTIyMzQsOS4wMTQ2NjkzNSBaIE0yNy4yMTIyMzQsOS4wMTQ2NjkzNSBDMjcuMjEyMjM0LDkuMDE0NjY5MzUgMjcuMjEyMjM0LDE1LjAxNDY2OTQgMjcuMjEyMjM0LDI3LjAxNDY2OTQgTDQwLjIxMjIzNCwyNy4wMTQ2Njk0IEw0MC4yMTIyMzQsOS4wMTQ2NjkzNSBDMzEuNTQ1NTY3Myw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IFogTTEzLjcxMjIzNCw5LjAxNDY2OTM1IEMxMy4xNjIyMzQsOS4wMTQ2NjkzNSAxMi43MTIyMzQsOS40NjQ2NjkzNSAxMi43MTIyMzQsMTAuMDE0NjY5NCBMMTIuNzEyMjM0LDI2LjAxNDY2OTQgQzEyLjcxMjIzNCwyNi41NjQ2Njk0IDEzLjE2MjIzNCwyNy4wMTQ2Njk0IDEzLjcxMjIzNCwyNy4wMTQ2Njk0IEwyNS43MTIyMzQsMjcuMDE0NjY5NCBMMjUuNzEyMjM0LDkuMDE0NjY5MzUgTDEzLjcxMjIzNCw5LjAxNDY2OTM1IFoiIGlkPSJjb2x1bW5zX3JlYWwiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_schema_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">SCHEMA</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">product_id</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">4</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">price</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">price</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">6</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
g4LDQyLjc4NDgxMDkgQzQ1Ljc1ODY0ODgsNDEuODg0MjE2MyA0NS4wMjg1NjM2LDQxLjE1MzgzODIgNDQuMTI3OTY5LDQxLjE1MzgzODIgWiIgaWQ9InNldF9vZiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_in_set()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">category</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">electronics, clothing, food, home</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">7</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
2ODYgQzE4LjE5OTcyODksMTAuODg3MDY4NiAxOS42NjIzNTUyLDEwLjc5NzgzMTEgMjEuMDA5NjI1NywxMC42NjA3MDk4IEwyMS4wMDk2MjU3LDMwLjQ1ODQwMjEgQzE5LjYyMzE3OCwzMC4zMTY5MjggMTguMTE5MTk3NSwzMC4yMzIwNDMzIDE2LjUxNzI3MzQsMzAuMjMyMDQzMyBDMTQuOTMyNzYxNSwzMC4yMzIwNDMzIDEzLjQ1NzA3NjMsMzAuMzE5MTA0NCAxMi4wOTQ1Njk5LDMwLjQ1ODQwMjEgTDEyLjA5NDU2OTksMTAuNjQzMjk3NSBaIiBpZD0iZ2VtaW5pIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTE2LjY2MDUzNTQsLTUuMDU5Mjk0OTkgQzE2LjkzNjY3NzgsLTUuMDU5Mjk0OTkgMTcuMTYwNTM1NCwtNC44MzU0MzczNyAxNy4xNjA1MzU0LC00LjU1OTI5NDk5IEwxNy4xNjA1MzU0LDQ0LjU2ODc1MDkgQzE3LjE2MDUzNTQsNDQuODQ0ODkzMyAxNi45MzY2Nzc4LDQ1LjA2ODc1MDkgMTYuNjYwNTM1NCw0NS4wNjg3NTA5IEMxNi4zODQzOTMsNDUuMDY4NzUwOSAxNi4xNjA1MzU0LDQ0Ljg0NDg5MzMgMTYuMTYwNTM1NCw0NC41Njg3NTA5IEwxNi4xNjA1MzU0LC00LjU1OTI5NDk5IEMxNi4xNjA1MzU0LC00LjgzNTQzNzM3IDE2LjM4NDM5MywtNS4wNTkyOTQ5OSAxNi42NjA1MzU0LC01LjA1OTI5NDk5IFoiIGlkPSJsaW5lX2JsYWNrIiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNi42NjA1MzUsIDIwLjAwNDcyOCkgcm90YXRlKC0zMjAuMDAwMDAwKSB0cmFuc2xhdGUoLTE2LjY2MDUzNSwgLTIwLjAwNDcyOCkgIiAvPgogICAgICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4LjU2MDAzMiwgMTkuMTU4MDMxKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMTguNTYwMDMyLCAtMTkuMTU4MDMxKSAiIHBvaW50cz0iMTguMDYwMDMxNiAtNC40NTM2NjczNSAxOS4wNjAwMzE2IC00LjQ1MzY2NzM1IDE5LjA2MDAzMTYgNDIuNzY5NzI5OSAxOC4wNjAwMzE2IDQyLjc2OTcyOTkiPjwvcG9seWdvbj4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

rows_distinct()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">product_id</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">

<span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px; font-size: 11px;"><span style="color: #777; font-weight: bold; font-size: 9px; text-transform: uppercase; margin-right: 3px;">Owner:</span> catalog-team</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px; font-size: 11px;"><span style="color: #777; font-weight: bold; font-size: 9px; text-transform: uppercase; margin-right: 3px;">Version:</span> 1.0.0</span>
</div></td>
</tr>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">
<hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(schema_check)</span> <span style="color:#4CA64C;">✓</span> Schema validation <strong>passed</strong>.</p>
Schema Comparison


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">product_id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">product_id</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">price</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">price</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">category</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">category</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">5</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">in_stock</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Boolean</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">5</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">in_stock</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Boolean</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('product_id', 'String'), ('name', 'String'), ('price', 'Float64'), ('category', 'String'), ('in_stock', 'Boolean')]</code>
</div></td>
</tr>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote"> 

Schema Match Settings

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64


</div></td>
</tr>
</tfoot>

</table>

</div></td>
</tr>
</tfoot>

</table>


# Pipeline YAML Format

While individual contracts validate data at a single point, **pipelines** enforce quality at both the input and output boundaries of a transformation. A pipeline YAML file packages the source contract, target contract, and pipeline-level settings (like global thresholds and short-circuit behavior) into a single, self-contained document.

This is especially powerful for team workflows: a data engineer defines the pipeline specification in YAML, commits it to version control, and any downstream system can load and execute it without knowing anything about the implementation details.

Let's build a pipeline in Python and export it to see what the YAML looks like:


``` python
# Create a pipeline and export to YAML
source = pb.Contract(
    name="raw_events",
    direction="source",
    version="1.0.0",
    schema=pb.Schema(
        event_id="String",
        user_id="String",
        event_type="String",
        value_cents="Int64",
    ),
    steps=[
        pb.Step("col_vals_not_null", columns=["event_id", "user_id", "event_type"]),
        pb.Step("col_vals_ge", columns="value_cents", value=0),
        pb.Step("col_vals_in_set", columns="event_type", set=["click", "purchase", "view", "signup"]),
        pb.Step("rows_distinct", columns_subset=["event_id"]),
    ],
)

target = pb.Contract(
    name="enriched_events",
    direction="target",
    version="1.0.0",
    schema=pb.Schema(
        event_id="String",
        user_id="String",
        event_type="String",
        value="Float64",
    ),
    steps=[
        pb.Step("col_vals_not_null", columns=["event_id", "user_id", "event_type", "value"]),
        pb.Step("col_vals_gt", columns="value", value=0),
        pb.Step("rows_distinct", columns_subset=["event_id"]),
    ],
)

pipeline = pb.Pipeline(
    source=source,
    target=target,
    label="Event Processing Pipeline",
    thresholds=pb.Thresholds(warning=0.02, error=0.10),
)

pipeline_yaml = pipeline.to_yaml("event_pipeline.yaml")
print(pipeline_yaml)
```


    pipeline:
      label: Event Processing Pipeline
      thresholds:
        warning: 0.02
        error: 0.1
    source:
      name: raw_events
      version: 1.0.0
      schema:
        event_id: String
        user_id: String
        event_type: String
        value_cents: Int64
      steps:
      - col_vals_not_null:
          columns:
          - event_id
          - user_id
          - event_type
      - col_vals_ge:
          columns: value_cents
          value: 0
      - col_vals_in_set:
          columns: event_type
          set:
          - click
          - purchase
          - view
          - signup
      - rows_distinct:
          columns_subset:
          - event_id
    target:
      name: enriched_events
      direction: target
      version: 1.0.0
      schema:
        event_id: String
        user_id: String
        event_type: String
        value: Float64
      steps:
      - col_vals_not_null:
          columns:
          - event_id
          - user_id
          - event_type
          - value
      - col_vals_gt:
          columns: value
          value: 0
      - rows_distinct:
          columns_subset:
          - event_id


## Pipeline YAML Structure

The generated YAML has three top-level sections that correspond to the three components of boundary enforcement: the `pipeline:` section for global settings, the `source:` section for the inbound contract, and the `target:` section for the outbound contract. Each section can be understood independently, which makes reviewing changes straightforward.

Here's the annotated structure:

``` yaml
# Pipeline metadata and global settings
pipeline:
  label: "Event Processing Pipeline"
  thresholds:
    warning: 0.02
    error: 0.10
  # short_circuit: true   (default, omitted when true)

# Source contract (inbound boundary)
source:
  name: raw_events
  version: "1.0.0"
  schema:
    event_id: String
    user_id: String
    event_type: String
    value_cents: Int64
  steps:
    - col_vals_not_null:
        columns: [event_id, user_id, event_type]
    - col_vals_ge:
        columns: value_cents
        value: 0

# Target contract (outbound boundary)
target:
  name: enriched_events
  version: "1.0.0"
  schema:
    event_id: String
    user_id: String
    event_type: String
    value: Float64
  steps:
    - col_vals_not_null:
        columns: [event_id, user_id, event_type, value]
    - col_vals_gt:
        columns: value
        value: 0
```


## Loading and Running a Pipeline from YAML

Loading a pipeline from YAML works the same way as loading a contract: call the [from_yaml()](../../reference/Contract.md#pointblank.Contract.from_yaml) class method with the file path. The returned [Pipeline](../../reference/Pipeline.md#pointblank.Pipeline) object is fully functional and ready to validate data or run a complete boundary enforcement workflow.


``` python
# Load the pipeline from YAML
loaded_pipeline = pb.Pipeline.from_yaml("event_pipeline.yaml")
print(loaded_pipeline)
```


    Pipeline(
      label='Event Processing Pipeline',
      source='raw_events',
      target='enriched_events',
      short_circuit=True
    )


With the pipeline loaded, you provide data and a transform function just as you would with a pipeline defined in Python. The YAML only stores the *contract specifications* whereas the transform logic always lives in your Python code (since transforms are arbitrary functions that can't be meaningfully serialized to YAML).


``` python
# Create test data and run the pipeline
events = pl.DataFrame(
    {
        "event_id": ["EVT-001", "EVT-002", "EVT-003", "EVT-004"],
        "user_id": ["U100", "U200", "U100", "U300"],
        "event_type": ["purchase", "click", "view", "signup"],
        "value_cents": [2500, 0, 0, 0],
    }
)

def enrich_events(df: pl.DataFrame) -> pl.DataFrame:
    """Convert cents to dollars, defaulting zero to a minimum value."""
    return df.with_columns(
        pl.when(pl.col("value_cents") > 0)
        .then(pl.col("value_cents") / 100)
        .otherwise(0.01)  # Minimum trackable value
        .alias("value")
    ).drop("value_cents")

result = loaded_pipeline.run(data=events, transform=enrich_events)
print(f"Pipeline passed: {result.passed}")
```


    Pipeline passed: True


``` python
result.target_validation
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Event Processing Pipeline [target]</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">Polars</span><span style="background-color: none; color: #222222; padding: 0.5em 0.5em; position: inherit; margin: 5px 10px 5px -4px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">enriched_events</span><span><span style="background-color: #AAAAAA; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; border: solid 1px #AAAAAA; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">WARNING</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #AAAAAA; padding: 2px 15px 2px 15px; font-size: smaller; margin-right: 5px;">0.02</span><span style="background-color: #EBBC14; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #EBBC14; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">ERROR</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #EBBC14; padding: 2px 15px 2px 15px; font-size: smaller; margin-right: 5px;">0.1</span><span style="background-color: #FF3300; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #FF3300; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">CRITICAL</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #FF3300; padding: 2px 15px 2px 15px; font-size: smaller;">--</span></span>

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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3NjaGVtYV9tYXRjaDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfc2NoZW1hX21hdGNoIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4zMTAzNDUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01My43MTIyMzQsMzkuNzg4NTI2OCBMNTQuMjEyMjM0LDU2LjI4ODUyNjggTDQyLjIxMjIzNCw1Ni43ODg1MjY4IEw0Mi4yMTIyMzQsMzkuNzg4NTI2OCBMNTMuNzEyMjM0LDM5Ljc4ODUyNjggWiBNMzkuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCw1Ni43ODg1MjY4IEwyNy43MTIyMzQsNTYuNzg4NTI2OCBMMjcuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCwzOS43ODg1MjY4IFogTTI1LjIxMjIzNCwzOS43ODg1MjY4IEwyNS4yMTIyMzQsNTYuNzg4NTI2OCBMMTMuNzEyMjM0LDU2Ljc4ODUyNjggTDEzLjIxMjIzNCw0MC4yODg1MjY4IEwyNS4yMTIyMzQsMzkuNzg4NTI2OCBaIiBpZD0iY29sdW1uc19zY2hlbWEiIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8ZyBpZD0idmVydGljYWxfZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLjAwMDAwMCwgMjkuMDAwMDAwKSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIyLjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSIyLjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUiPjwvbGluZT4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI1LjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSI1LjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUtQ29weSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MS43MTIyMzQsOS4wMTQ2NjkzNSBMNDEuNzEyMjM0LDI3LjAxNDY2OTQgTDUzLjcxMjIzNCwyNy4wMTQ2Njk0IEM1NC4yNjIyMzQsMjcuMDE0NjY5NCA1NC43MTIyMzQsMjYuNTY0NjY5NCA1NC43MTIyMzQsMjYuMDE0NjY5NCBMNTQuNzEyMjM0LDEwLjAxNDY2OTQgQzU0LjcxMjIzNCw5LjQ2NDY2OTM1IDU0LjI2MjIzNCw5LjAxNDY2OTM1IDUzLjcxMjIzNCw5LjAxNDY2OTM1IEw0MS43MTIyMzQsOS4wMTQ2NjkzNSBaIE0yNy4yMTIyMzQsOS4wMTQ2NjkzNSBDMjcuMjEyMjM0LDkuMDE0NjY5MzUgMjcuMjEyMjM0LDE1LjAxNDY2OTQgMjcuMjEyMjM0LDI3LjAxNDY2OTQgTDQwLjIxMjIzNCwyNy4wMTQ2Njk0IEw0MC4yMTIyMzQsOS4wMTQ2NjkzNSBDMzEuNTQ1NTY3Myw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IFogTTEzLjcxMjIzNCw5LjAxNDY2OTM1IEMxMy4xNjIyMzQsOS4wMTQ2NjkzNSAxMi43MTIyMzQsOS40NjQ2NjkzNSAxMi43MTIyMzQsMTAuMDE0NjY5NCBMMTIuNzEyMjM0LDI2LjAxNDY2OTQgQzEyLjcxMjIzNCwyNi41NjQ2Njk0IDEzLjE2MjIzNCwyNy4wMTQ2Njk0IDEzLjcxMjIzNCwyNy4wMTQ2Njk0IEwyNS43MTIyMzQsMjcuMDE0NjY5NCBMMjUuNzEyMjM0LDkuMDE0NjY5MzUgTDEzLjcxMjIzNCw5LjAxNDY2OTM1IFoiIGlkPSJjb2x1bW5zX3JlYWwiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_schema_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">SCHEMA</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">event_id</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">4<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">user_id</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">4<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">4</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">event_type</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
DAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">4<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">5</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">4<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">6</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">4<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">7</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93c19kaXN0aW5jdDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJyb3dzX2Rpc3RpbmN0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC40ODI3NTkpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJub19nZW1pbmkiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE3LjAwMDAwMCwgMTMuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMy42NjcwNTYxOSw2LjYyMTA3NTA4IEMzLjEyNTEwMTA0LDYuNjQwNjYzODYgMi42NzQ1NTk3NCw3LjA0NzY3NDQ0IDIuNjAyNzM0MzIsNy41ODUyNzY4MiBDMi41Mjg3MzIyOCw4LjEyMjg3OTIgMi44NTMwMzUyNiw4LjYzNDM2MjcgMy4zNzEwNDg0OCw4Ljc5NzYwMjM5IEM0LjQwNDg5OTA5LDkuMTU0NTUyODYgNi43MDExMzU1Myw5Ljg3MDYzMDIxIDkuODY1ODA1OTUsMTAuMzY0NzAyIEw5Ljg2NTgwNTk1LDMwLjczNjk5NzYgQzYuNjU5NzgxMzcsMzEuMjMzMjQ1OCA0LjM3MjI1MTA0LDMxLjk2NDU1OSAzLjM3MTA0ODQ4LDMyLjMyMTUwOTUgQzIuNzg5OTE1NTUsMzIuNTI4Mjc5NyAyLjQ4NTIwMTczLDMzLjE2ODE3ODQgMi42OTE5NzE5NiwzMy43NDkzMTE0IEMyLjg5ODc0MjIsMzQuMzMwNDQ0MyAzLjUzODY0MDk0LDM0LjYzNTE1ODEgNC4xMTk3NzM4NywzNC40MjgzODc5IEM1LjU0OTc1MjE3LDMzLjkxOTA4MDggMTAuMjAzMTY3OCwzMi40NjA4MDcyIDE2LjUxNzI3MzQsMzIuNDYwODA3MiBDMjIuNzk0Mzc4MSwzMi40NjA4MDcyIDI3LjU1MDA5MDEsMzMuODkwNzg1NSAyOS4wNTQwNzA2LDM0LjQxMDk3NTcgQzI5LjYzNTIwMzYsMzQuNjEzMzkyNiAzMC4yNzA3NDk1LDM0LjMwNDMyNiAzMC40NzMxNjY0LDMzLjcyMzE5MzEgQzMwLjY3NTU4MzMsMzMuMTQyMDYwMSAzMC4zNjY1MTY3LDMyLjUwNjUxNDIgMjkuNzg1MzgzOCwzMi4zMDQwOTczIEMyOC43NDkzNTY4LDMxLjk0NDk3MDQgMjYuNDMxMzU1NCwzMS4yNDQxMjgzIDIzLjIzODM4OTcsMzAuNzU0NDA5OCBMMjMuMjM4Mzg5NywxMC4zODIxMTQzIEMyNi40NDQ0MTQzLDkuODg4MDQyNDMgMjguNzQ1MDA0LDkuMTYxMDgyNTkgMjkuNzY3OTcxNiw4Ljc5NzYwMjM5IEMzMC4zNDkxMDQ1LDguNTkwODMyMTUgMzAuNjUzODE4NCw3Ljk1MDkzMzQxIDMwLjQ0NzA0ODEsNy4zNjk4MDA0OCBDMzAuMjQwMjc3OSw2Ljc4ODY2NzU1IDI5LjYwMDM3OTEsNi40ODM5NTM3MyAyOS4wMTkyNDYyLDYuNjkwNzIzOTYgQzI3LjU1ODc5NjMsNy4yMDg3Mzc3NCAyMi45MTYyNjM3LDguNjU4MzA0NjQgMTYuNjM5MTU4OSw4LjY1ODMwNDY0IEMxMC4zNzA3NjAzLDguNjU4MzA0NjQgNS42MTI4NzE4OCw3LjIxMzA5MDUxIDQuMTAyMzYxNjYsNi42OTA3MjM5NiBDMy45NjMwNjM5MSw2LjYzODQ4NzMgMy44MTUwNjAwNSw2LjYxNDU0NTQ4IDMuNjY3MDU2MTksNi42MjEwNzUwOCBaIE0xMi4wOTQ1Njk5LDEwLjY0MzI5NzUgQzEzLjQ5NDA3NzEsMTAuNzg5MTI1IDE1LjAxOTgyMjYsMTAuODg3MDY4NiAxNi42MzkxNTg5LDEwLjg4NzA2ODYgQzE4LjE5OTcyODksMTAuODg3MDY4NiAxOS42NjIzNTUyLDEwLjc5NzgzMTEgMjEuMDA5NjI1NywxMC42NjA3MDk4IEwyMS4wMDk2MjU3LDMwLjQ1ODQwMjEgQzE5LjYyMzE3OCwzMC4zMTY5MjggMTguMTE5MTk3NSwzMC4yMzIwNDMzIDE2LjUxNzI3MzQsMzAuMjMyMDQzMyBDMTQuOTMyNzYxNSwzMC4yMzIwNDMzIDEzLjQ1NzA3NjMsMzAuMzE5MTA0NCAxMi4wOTQ1Njk5LDMwLjQ1ODQwMjEgTDEyLjA5NDU2OTksMTAuNjQzMjk3NSBaIiBpZD0iZ2VtaW5pIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTE2LjY2MDUzNTQsLTUuMDU5Mjk0OTkgQzE2LjkzNjY3NzgsLTUuMDU5Mjk0OTkgMTcuMTYwNTM1NCwtNC44MzU0MzczNyAxNy4xNjA1MzU0LC00LjU1OTI5NDk5IEwxNy4xNjA1MzU0LDQ0LjU2ODc1MDkgQzE3LjE2MDUzNTQsNDQuODQ0ODkzMyAxNi45MzY2Nzc4LDQ1LjA2ODc1MDkgMTYuNjYwNTM1NCw0NS4wNjg3NTA5IEMxNi4zODQzOTMsNDUuMDY4NzUwOSAxNi4xNjA1MzU0LDQ0Ljg0NDg5MzMgMTYuMTYwNTM1NCw0NC41Njg3NTA5IEwxNi4xNjA1MzU0LC00LjU1OTI5NDk5IEMxNi4xNjA1MzU0LC00LjgzNTQzNzM3IDE2LjM4NDM5MywtNS4wNTkyOTQ5OSAxNi42NjA1MzU0LC01LjA1OTI5NDk5IFoiIGlkPSJsaW5lX2JsYWNrIiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNi42NjA1MzUsIDIwLjAwNDcyOCkgcm90YXRlKC0zMjAuMDAwMDAwKSB0cmFuc2xhdGUoLTE2LjY2MDUzNSwgLTIwLjAwNDcyOCkgIiAvPgogICAgICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4LjU2MDAzMiwgMTkuMTU4MDMxKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMTguNTYwMDMyLCAtMTkuMTU4MDMxKSAiIHBvaW50cz0iMTguMDYwMDMxNiAtNC40NTM2NjczNSAxOS4wNjAwMzE2IC00LjQ1MzY2NzM1IDE5LjA2MDAzMTYgNDIuNzY5NzI5OSAxOC4wNjAwMzE2IDQyLjc2OTcyOTkiPjwvcG9seWdvbj4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

rows_distinct()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">event_id</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">4<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">

<span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px; font-size: 11px;"><span style="color: #777; font-weight: bold; font-size: 9px; text-transform: uppercase; margin-right: 3px;">Version:</span> 1.0.0</span>
</div></td>
</tr>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">
<hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(schema_check)</span> <span style="color:#4CA64C;">✓</span> Schema validation <strong>passed</strong>.</p>
Schema Comparison


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">event_id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">event_id</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">user_id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">user_id</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">event_type</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">event_type</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('event_id', 'String'), ('user_id', 'String'), ('event_type', 'String'), ('value', 'Float64')]</code>
</div></td>
</tr>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote"> 

Schema Match Settings

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64


</div></td>
</tr>
</tfoot>

</table>

</div></td>
</tr>
</tfoot>

</table>


# Practical Patterns

The following patterns illustrate common ways teams use YAML contracts and pipelines in real-world data platforms. These aren't exhaustive: they're starting points that you can adapt to your own workflow.


## Pattern 1: Environment-Specific Pipelines

A common need is to apply different tolerance levels in different environments. In development, you might accept 20% failure rates while iterating on a transform. In production, anything above 1% is a critical issue. YAML contracts make this easy: store the same validation logic in a shared contract, but load different pipeline configurations that set environment-appropriate thresholds.


``` python
# Same contracts, different thresholds
dev_pipeline = pb.Pipeline(
    source=source,
    target=target,
    thresholds=pb.Thresholds(warning=0.20, error=0.50),  # Lenient for dev
    label="DEV: Event Pipeline",
)

prod_pipeline = pb.Pipeline(
    source=source,
    target=target,
    thresholds=pb.Thresholds(warning=0.01, error=0.05, critical=0.10),  # Strict
    label="PROD: Event Pipeline",
)

# In practice, you'd select based on an env var:
# import os
# pipeline = pb.Pipeline.from_yaml(f"contracts/{os.environ['ENV']}_pipeline.yaml")
```


## Pattern 2: Contract Library

As your data platform grows, you'll accumulate many contracts. A recommended practice is to organize them into a directory structure that mirrors your data architecture. This creates a discoverable, browsable "library" of data quality expectations that serves as living documentation of your system's data contracts.

Here's a typical layout:

    contracts/
    ├── sources/
    │   ├── raw_events.yaml
    │   ├── raw_transactions.yaml
    │   └── raw_customers.yaml
    ├── targets/
    │   ├── clean_events.yaml
    │   ├── clean_transactions.yaml
    │   └── clean_customers.yaml
    └── pipelines/
        ├── event_pipeline.yaml
        └── transaction_pipeline.yaml

``` python
# Load any contract from the library
events_source = pb.Contract.from_yaml("contracts/sources/raw_events.yaml")
events_target = pb.Contract.from_yaml("contracts/targets/clean_events.yaml")

# Build pipeline dynamically
pipeline = pb.Pipeline(source=events_source, target=events_target)
```


## Pattern 3: CI/CD Integration

YAML contracts integrate naturally into CI/CD pipelines. Because contracts are plain files, you can load them in test suites, gate deployments on contract compliance, and fail builds when data quality regresses. This turns data quality from a manual spot-check into an automated, enforceable standard.

Here's what a pytest-based contract test might look like:

``` python
# In your test suite or CI script:
import pointblank as pb

def test_pipeline_contract():
    pipeline = pb.Pipeline.from_yaml("contracts/pipelines/event_pipeline.yaml")
    test_data = load_test_fixture("events_sample.parquet")

    result = pipeline.run(data=test_data, transform=my_transform)
    assert result.passed, result.get_report()
```


## Pattern 4: Composing Steps from Multiple Sources

Sometimes you want to define a set of "base" validation rules that apply to all tables (like "every table must have a non-null `id` column that's unique"), and then layer on table-specific rules. YAML makes this easy because you can parse step definitions from YAML fragments and combine them programmatically.

This pattern is especially useful when you have organization-wide data quality standards that every contract should inherit.


``` python
# Base steps that apply to all tables
base_steps_yaml = """
- col_vals_not_null:
    columns: [id]
- rows_distinct:
    columns_subset: [id]
"""

import yaml
base_steps = [pb.Step.from_dict(s) for s in yaml.safe_load(base_steps_yaml)]

# Compose with table-specific steps
full_contract = pb.Contract(
    name="composed_contract",
    steps=base_steps + [
        pb.Step("col_vals_gt", columns="score", value=0),
        pb.Step("col_vals_le", columns="score", value=100),
    ],
)

print(f"Total steps: {len(full_contract.steps)}")
for step in full_contract.steps:
    print(f"  - {step}")
```


    Total steps: 4
      - Step('col_vals_not_null', columns=['id'])
      - Step('rows_distinct', columns_subset=['id'])
      - Step('col_vals_gt', columns='score', value=0)
      - Step('col_vals_le', columns='score', value=100)


# Round-Trip Fidelity

A critical property of Pointblank's YAML serialization is **round-trip fidelity**: when you save a contract to YAML and load it back, you get an identical object. No information is lost, no defaults are silently changed, and no steps are reordered. This guarantee means you can confidently use YAML as your source of truth without worrying about subtle differences between the YAML representation and the in-memory object.

Let's verify this with a fully-specified contract:


``` python
# Original contract
original = pb.Contract(
    name="fidelity_test",
    direction="target",
    version="3.0.0",
    owner="qa-team",
    consumers=["reporting", "dashboards"],
    on_violation="raise",
    schema=pb.Schema(x="Int64", y="Float64", z="String"),
    steps=[
        pb.Step("col_vals_not_null", columns=["x", "y", "z"]),
        pb.Step("col_vals_between", columns="y", left=-1.0, right=1.0),
    ],
    thresholds=pb.Thresholds(warning=0.01, error=0.05, critical=0.10),
)

# Round-trip through YAML
original.to_yaml("fidelity_test.yaml")
restored = pb.Contract.from_yaml("fidelity_test.yaml")

# Verify fidelity
print(f"Name matches: {original.name == restored.name}")
print(f"Direction matches: {original.direction == restored.direction}")
print(f"Version matches: {original.version == restored.version}")
print(f"Owner matches: {original.owner == restored.owner}")
print(f"Steps match: {original.steps == restored.steps}")
print(f"Thresholds match: {original.thresholds.warning == restored.thresholds.warning}")
```


    Name matches: True
    Direction matches: True
    Version matches: True
    Owner matches: True
    Steps match: True
    Thresholds match: True


# Cleanup

The examples in this guide created several YAML files on disk to demonstrate the serialization workflow. The following hidden cell removes them so they don't clutter your working directory. In your own projects, these files would live in version control and persist across sessions. You'd never need to clean them up like this.


# Conclusion

The table below summarizes the key operations and how they map between the Python API and YAML:

| Operation | Python API | YAML |
|----|----|----|
| Define contract | `pb.Contract(...)` | `contract:` block |
| Define pipeline | `pb.Pipeline(...)` | `pipeline:` + `source:` + `target:` |
| Save | `.to_yaml("path.yaml")` | -- |
| Load contract | `pb.Contract.from_yaml("path.yaml")` | -- |
| Load pipeline | `pb.Pipeline.from_yaml("path.yaml")` | -- |
| Validate | `.validate(data)` | Load -\> validate in Python |
| Run pipeline | `.run(data, transform)` | Load -\> run in Python |

YAML contracts and pipelines give you the best of both worlds: declarative, reviewable specifications that integrate seamlessly with Pointblank's Python execution engine. The separation of *what* (YAML contracts) from *how* (Python transforms) creates a clean architecture where data quality expectations can evolve independently from processing logic: reviewed by different people, on different timelines, with clear accountability at each boundary.
