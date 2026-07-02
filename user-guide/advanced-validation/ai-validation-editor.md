# AI Validation Editor

Where [Draft Validation](../../user-guide/advanced-validation/draft-validation.md) uses a large language model (LLM) to *generate* a validation plan from scratch, the AI Validation Editor uses one to *edit* a plan you already have. You give it an existing plan plus a plain-English instruction, such as "add a not-null check on `user_id`, tighten the email regex, and drop the `price > 0` check", and it returns a revised plan. You review the change as a diff and explicitly accept it before anything runs, keeping a human in the loop.

The Editor is built on the [plan serialization](../../user-guide/advanced-validation/plan-serialization.md) primitives described in the previous section: it renders your current plan to code with <a href="../../reference/Validate.to_code.html#pointblank.Validate.to_code" class="gdls-link"><code>to_code()</code></a>, sends that code and your instruction to the model, and returns the model's revised code for your review.

The main entry point is the [EditValidation](../../reference/EditValidation.md#pointblank.EditValidation) class. Two convenience methods on [Validate](../../reference/Validate.md#pointblank.Validate), <a href="../../reference/Validate.suggest_improvements.html#pointblank.Validate.suggest_improvements" class="gdls-link"><code>suggest_improvements()</code></a> and <a href="../../reference/Validate.from_prompt.html#pointblank.Validate.from_prompt" class="gdls-link"><code>from_prompt()</code></a>, wrap the same flow, and the `pb edit` CLI command brings it to the terminal.

> **Note: About the examples on this page**
>
> The [EditValidation](../../reference/EditValidation.md#pointblank.EditValidation) calls below need a live LLM, so they're shown as **non-executable code** (what you'd write once your API key is set up). The output beneath each one is rendered from a *representative* stand-in model response, so you can see the shape of the results here without a key. A real model may word things slightly differently, but the mechanics (diffing, reviewing, accepting) are exactly what you'll experience.


# How the Editor Works

When you edit a plan, the process works through these steps:

1.  your current plan is normalized to canonical Pointblank code via <a href="../../reference/Validate.to_code.html#pointblank.Validate.to_code" class="gdls-link"><code>to_code()</code></a>
2.  that code, your instruction, and (optionally) a [DataScan](../../reference/DataScan.md#pointblank.DataScan) summary of the table are assembled into a prompt and sent to your selected LLM provider
3.  the model returns a complete revised plan as Python code
4.  the revised code is checked for syntax and unknown methods; on failure the model is automatically re-prompted to fix it
5.  you inspect the proposed change as a diff and, when satisfied, accept it to obtain a runnable [Validate](../../reference/Validate.md#pointblank.Validate) object

As with [DraftValidation](../../reference/DraftValidation.md#pointblank.DraftValidation), the entire dataset is never sent to the model, only a summary. See [Data Privacy](#data-privacy) below.


# Requirements and Setup

The Editor shares its requirements with [Draft Validation](../../user-guide/advanced-validation/draft-validation.md): an API key from a supported LLM provider and the optional dependencies, installed with:

``` bash
pip install pointblank[generate]
```

The same five providers are supported (`anthropic`, `openai`, `ollama`, `bedrock`, and `azure-openai`), the model is specified in `provider:model` form, and API keys are resolved from the `api_key=` argument, environment variables, or a `.env` file exactly as described on the Draft Validation page.


# Editing a Plan with [EditValidation](../../reference/EditValidation.md#pointblank.EditValidation)

Suppose you have an existing plan:


``` python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset("small_table"), label="Small table checks")
    .col_vals_gt(columns="d", value=100)
    .col_vals_regex(columns="b", pattern="[0-9]-[a-z]{3}")
)
```


Describe the change you want in plain English and hand both to [EditValidation](../../reference/EditValidation.md#pointblank.EditValidation):


``` python
edited = pb.EditValidation(
    validation=validation,
    instruction="Loosen the d check to 50, drop the b regex, and add a not-null check on a.",
    model="anthropic:claude-opus-4-8",
    data=pb.load_dataset("small_table"),
)
```


The `validation=` argument is flexible. It accepts a live [Validate](../../reference/Validate.md#pointblank.Validate) object (as above), a code string (such as the output of <a href="../../reference/Validate.to_code.html#pointblank.Validate.to_code" class="gdls-link"><code>to_code()</code></a>), a YAML configuration string, or a path to a `.py` or `.yaml`/`.yml` file. All of these are normalized to code before being sent to the model, so you can edit a plan wherever it happens to live.

The optional `data=` argument is worth providing when you can: it includes a [DataScan](../../reference/DataScan.md#pointblank.DataScan) profile of the table in the prompt so the model can make *informed* edits (for example, choosing a realistic range from the observed min and max). It's also used as the default data source when you accept the edit.


# Reviewing the Proposed Change

The Editor proposes; you dispose. Before anything runs, inspect what the model changed using the `edited` object from above.


## As a Unified Diff

<a href="../../reference/EditValidation.html#pointblank.EditValidation.diff" class="gdls-link"><code>diff()</code></a> returns a textual, line-level diff of the original plan versus the revised plan:


``` python
print(edited.diff())
```


    --- original_plan.py
    +++ edited_plan.py
    @@ -5,8 +5,8 @@
             data=your_data,  # Replace your_data with the actual data variable
             label="Small table checks",
         )
    -    .col_vals_gt(columns="d", value=100)
    -    .col_vals_regex(columns="b", pattern="[0-9]-[a-z]{3}")
    +    .col_vals_gt(columns="d", value=50)
    +    .col_vals_not_null(columns="a")
     )
     
     validation


## As a Structured Change List

<a href="../../reference/EditValidation.html#pointblank.EditValidation.changed_steps" class="gdls-link"><code>changed_steps()</code></a> compares the two plans at the level of validation steps and returns a structured list. Unlike the textual diff, this is robust to reformatting (it compares whole steps rather than lines):


``` python
edited.changed_steps()
```


    [{'action': 'modify',
      'method': 'col_vals_gt',
      'old': "col_vals_gt(columns='d', value=100)",
      'new': "col_vals_gt(columns='d', value=50)"},
     {'action': 'remove',
      'method': 'col_vals_regex',
      'old': "col_vals_regex(columns='b', pattern='[0-9]-[a-z]{3}')"},
     {'action': 'add',
      'method': 'col_vals_not_null',
      'new': "col_vals_not_null(columns='a')"}]


Each record has an `"action"` of `"add"`, `"remove"`, or `"modify"`, the `"method"` involved, and the relevant `"old"`/`"new"` step text.


## As a Table

<a href="../../reference/EditValidation.html#pointblank.EditValidation.review" class="gdls-link"><code>review()</code></a> renders the same change list as a [Great Tables](https://posit-dev.github.io/great-tables/) object, color-coding additions, removals, and modifications for quick visual review in a notebook:


``` python
edited.review()
```


<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="4" class="gt_heading gt_title gt_font_normal">Proposed plan changes</th>
</tr>
<tr class="gt_heading">
<th colspan="4" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Loosen the d check to 50, drop the b regex, and add a not-null check on a.</th>
</tr>
<tr class="gt_col_headings">
<th id="action" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Change</th>
<th id="method" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Method</th>
<th id="before" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Before</th>
<th id="after" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">After</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="background-color: #fef7e0">~ modified</td>
<td class="gt_row gt_left" style="background-color: #fef7e0">col_vals_gt</td>
<td class="gt_row gt_left" style="background-color: #fef7e0">col_vals_gt(columns='d', value=100)</td>
<td class="gt_row gt_left" style="background-color: #fef7e0">col_vals_gt(columns='d', value=50)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #fce8e6">− removed</td>
<td class="gt_row gt_left" style="background-color: #fce8e6">col_vals_regex</td>
<td class="gt_row gt_left" style="background-color: #fce8e6">col_vals_regex(columns='b', pattern='[0-9]-[a-z]{3}')</td>
<td class="gt_row gt_left" style="background-color: #fce8e6"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #e6f4ea">+ added</td>
<td class="gt_row gt_left" style="background-color: #e6f4ea">col_vals_not_null</td>
<td class="gt_row gt_left" style="background-color: #e6f4ea"></td>
<td class="gt_row gt_left" style="background-color: #e6f4ea">col_vals_not_null(columns='a')</td>
</tr>
</tbody>
</table>


# Accepting the Edit

Once you're satisfied with the proposed change, <a href="../../reference/EditValidation.html#pointblank.EditValidation.accept" class="gdls-link"><code>accept()</code></a> executes the revised plan and returns a [Validate](../../reference/Validate.md#pointblank.Validate) object, ready to interrogate:


``` python
plan = edited.accept()          # a Validate (not yet interrogated)
plan.interrogate()              # run it
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Small table checks</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span>

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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">50</td>
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
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
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
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">

<span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin-left: 10px; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-07-02 14:11:27 UTC</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">< 1 s</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 1px 5px -1px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-07-02 14:11:27 UTC</span>
</div></td>
</tr>
</tfoot>

</table>


The data bound to the plan is the `data=` you passed to [EditValidation](../../reference/EditValidation.md#pointblank.EditValidation); you can override it with `edited.accept(data=other_table)`. Nothing runs until you call [accept()](../../reference/EditValidation.md#pointblank.EditValidation.accept), and the plan isn't interrogated until you call [interrogate()](../../reference/Validate.interrogate.md#pointblank.Validate.interrogate), so there's a clear, deliberate boundary between the model's proposal and execution.


# Comparing Two Plans Without a Model

You don't need an LLM to review the difference between two plans. When you already have a "before" and "after" (for example, two versions of a plan from source control), <a href="../../reference/EditValidation.html#pointblank.EditValidation.from_plans" class="gdls-link"><code>EditValidation.from_plans()</code></a> builds the same comparison object directly, with all of [diff()](../../reference/EditValidation.md#pointblank.EditValidation.diff), [changed_steps()](../../reference/EditValidation.md#pointblank.EditValidation.changed_steps), and [review()](../../reference/EditValidation.md#pointblank.EditValidation.review) available and no API key required. The following example is fully live:


``` python
before = pb.Validate(data=pb.load_dataset("small_table")).col_vals_gt(columns="d", value=100)
after = (
    pb.Validate(data=pb.load_dataset("small_table"))
    .col_vals_gt(columns="d", value=50)
    .col_vals_not_null(columns="a")
)

pb.EditValidation.from_plans(before, after).review()
```


<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="4" class="gt_heading gt_title gt_font_normal">Proposed plan changes</th>
</tr>
<tr class="gt_heading">
<th colspan="4" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Manual plan comparison</th>
</tr>
<tr class="gt_col_headings">
<th id="action" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Change</th>
<th id="method" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Method</th>
<th id="before" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Before</th>
<th id="after" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">After</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="background-color: #fef7e0">~ modified</td>
<td class="gt_row gt_left" style="background-color: #fef7e0">col_vals_gt</td>
<td class="gt_row gt_left" style="background-color: #fef7e0">col_vals_gt(columns='d', value=100)</td>
<td class="gt_row gt_left" style="background-color: #fef7e0">col_vals_gt(columns='d', value=50)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #e6f4ea">+ added</td>
<td class="gt_row gt_left" style="background-color: #e6f4ea">col_vals_not_null</td>
<td class="gt_row gt_left" style="background-color: #e6f4ea"></td>
<td class="gt_row gt_left" style="background-color: #e6f4ea">col_vals_not_null(columns='a')</td>
</tr>
</tbody>
</table>


# Guardrails

Two guardrails keep the flow from handing you broken code.

- **Syntax and lint check.** The revised code is parsed with `ast.parse()` and checked for validation-method typos before it's returned. <a href="../../reference/EditValidation.html#pointblank.EditValidation.validate_syntax" class="gdls-link"><code>validate_syntax()</code></a> reports whether the current revision passes.
- **Automatic re-prompt.** If the check fails, the model is automatically re-prompted with the error message to correct itself, up to [max_reprompts](../../reference/EditValidation.md#pointblank.EditValidation.max_reprompts) times (default `1`). If the code still doesn't pass, <a href="../../reference/EditValidation.html#pointblank.EditValidation.accept" class="gdls-link"><code>accept()</code></a> refuses to run it and points you at <a href="../../reference/EditValidation.html#pointblank.EditValidation.to_code" class="gdls-link"><code>to_code()</code></a> so you can fix it by hand.

These same guardrails are also applied to [`DraftValidation`](../../user-guide/advanced-validation/draft-validation.md).


# Natural-Language Refinements

Two methods on [Validate](../../reference/Validate.md#pointblank.Validate) wrap the edit flow for common cases.


## Suggesting Improvements

<a href="../../reference/Validate.suggest_improvements.html#pointblank.Validate.suggest_improvements" class="gdls-link"><code>suggest_improvements()</code></a> profiles the table, derives an instruction targeting gaps in the current plan (columns with no coverage, missing thresholds), and asks the model to extend the plan. It returns an [EditValidation](../../reference/EditValidation.md#pointblank.EditValidation) you review and accept as usual:


``` python
proposal = validation.suggest_improvements(model="anthropic:claude-opus-4-8")
```


``` python
print(proposal.diff())
```


    --- original_plan.py
    +++ edited_plan.py
    @@ -4,9 +4,13 @@
         pb.Validate(
             data=your_data,  # Replace your_data with the actual data variable
             label="Small table checks",
    +        thresholds=pb.Thresholds(warning=0.1, error=0.25),
         )
         .col_vals_gt(columns="d", value=100)
         .col_vals_regex(columns="b", pattern="[0-9]-[a-z]{3}")
    +    .col_vals_not_null(columns=["a", "c", "e"])
    +    .col_vals_in_set(columns="f", set=["low", "mid", "high"])
    +    .rows_distinct()
     )
     
     validation


## Building a Plan from a Prompt

<a href="../../reference/Validate.from_prompt.html#pointblank.Validate.from_prompt" class="gdls-link"><code>from_prompt()</code></a> is the same flow starting from an *empty* plan: the model authors steps that satisfy your prompt, using a [DataScan](../../reference/DataScan.md#pointblank.DataScan) profile of the data and any table name, label, and thresholds carried by the object you call it on:


``` python
base = pb.Validate(data=pb.load_dataset("small_table"), tbl_name="small_table")

authored = base.from_prompt(
    "ensure column a has no nulls and column d is always positive",
    model="anthropic:claude-opus-4-8",
)
```


``` python
print(authored.to_code())
```


    import pointblank as pb

    validation = (
        pb.Validate(
            data=your_data,  # Replace your_data with the actual data variable
            tbl_name="small_table",
        )
        .col_vals_not_null(columns="a")
        .col_vals_gt(columns="d", value=0)
    )

    validation


# Editing from the Command Line

The `pb edit` command brings the Editor to the terminal. Point it at a validation file, describe the change, and choose a model:

``` bash
pb edit plan.py -i "add a not-null check on user_id" -m anthropic:claude-opus-4-8
```

The command prints the proposed change as a color diff along with a summary (for example, `1 added, 1 removed, 1 modified`). Provide a data source for profile-informed edits with `--data`, and write the revised plan to a file with `--output` (add `--yes` to skip the confirmation prompt):

``` bash
pb edit plan.yaml -i "tighten the price range to 0-1000" -m openai:gpt-4o --data sales.csv
pb edit plan.py   -i "drop the email regex check"        -m anthropic:claude-opus-4-8 -o plan2.py -y
```


# Data Privacy

The Editor follows the same privacy posture as [Draft Validation](../../user-guide/advanced-validation/draft-validation.md): your actual dataset is never sent to the model. When you provide `data=`, only a [DataScan](../../reference/DataScan.md#pointblank.DataScan) summary is transmitted, which includes column names and types, per-column statistics (min, max, mean, median, missing counts), and a small sample of values from each column. This is enough context for informed edits without exposing the full table.

If you prefer to send no data at all, omit `data=` on [EditValidation](../../reference/EditValidation.md#pointblank.EditValidation); the model then edits the plan using only the plan code and your instruction. (Because their whole purpose is to profile coverage gaps, [suggest_improvements()](../../reference/Validate.suggest_improvements.md#pointblank.Validate.suggest_improvements) and [from_prompt()](../../reference/Validate.from_prompt.md#pointblank.Validate.from_prompt) always include the summary.)


# Best Practices

- **Make small, targeted edits.** The Editor is designed to change only what your instruction implies and preserve unrelated steps verbatim. Keeping instructions focused produces minimal, reviewable diffs.
- **Always review before accepting.** Read the diff or the change list; the model proposes, but you decide what runs.
- **Provide data when you can.** A [DataScan](../../reference/DataScan.md#pointblank.DataScan) profile lets the model choose realistic bounds and set members rather than guessing.
- **Version-control the accepted plan.** Once you accept an edit, serialize the result with <a href="../../reference/Validate.to_code.html#pointblank.Validate.to_code" class="gdls-link"><code>to_code()</code></a> or <a href="../../reference/Validate.to_yaml.html#pointblank.Validate.to_yaml" class="gdls-link"><code>to_yaml()</code></a> and commit it, so the plan's evolution is tracked alongside your code.


# Conclusion

The AI Validation Editor extends Pointblank's AI capabilities from generating plans to iterating on them. By combining plan serialization, a natural-language edit prompt, structural diffing, and syntax guardrails, it lets you refine validation plans conversationally while keeping a human firmly in control of what actually runs. Together with [Draft Validation](../../user-guide/advanced-validation/draft-validation.md), it covers both halves of an AI-assisted validation workflow: create a plan, then keep improving it.
