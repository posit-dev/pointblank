# Briefs

When validating data with Pointblank, it's often helpful to have descriptive labels for each validation step. This is where *briefs* come in. A brief is a short description of what a validation step is checking and it appears in the `STEP` column of the validation report table. Briefs make your validation reports more readable and they help others understand what each step is verifying without needing to look at the code.

Briefs can be set in two ways:

1.  Globally: applied to all validation steps via the `brief=` parameter in [Validate](../../reference/Validate.md#pointblank.Validate)
2.  Locally: set for individual validation steps via the `brief=` parameter in each validation method

Understanding these two approaches to adding briefs gives you flexibility in how you document your validation process. Global briefs provide consistency across all steps and save time when you want similar descriptions throughout, while step-level briefs allow for precise customization when specific validations need more detailed or unique explanations. In practice, many validation workflows will combine both approaches (i.e., setting a useful global brief template while overriding it for steps that require special attention).


# Global Briefs

To set a global brief that applies to all validation steps, use the `Validate(brief=)` parameter when creating a [Validate](../../reference/Validate.md#pointblank.Validate) object:


``` python
import pointblank as pb
import polars as pl

# Sample data
data = pl.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "value": [10, 20, 30, 40, 50],
    "category": ["A", "B", "C", "A", "B"]
})

# Create a validation with a global brief
(
    pb.Validate(
        data=data,

        # Global brief template ---
        brief="Step {step}: {auto}"
    )
    .col_vals_gt(columns="value", value=5)
    .col_vals_in_set(columns="category", set=["A", "B", "C"])
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-04-22|21:15:44</span>

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


<p>Step 1: Expect that values in <code>value</code> should be > <code>5</code>.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">5</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfaW5fc2V0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF92YWxzX2luX3NldCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMTcyNDE0KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuMTI3OTY5LDQxLjE1MzgzODIgTDMxLjA4MTQ1NjgsNDEuMTUzODM4MiBDMjkuOTUxMDc0OCw0MS4xNTM2NDI5IDI4Ljg4MjcwNTIsNDAuOTI1NjEzNCAyNy45MDc5ODg4LDQwLjUxMzY5NTMgQzI2LjQ0Njc0NDIsMzkuODk2MDEzNiAyNS4xOTg0OSwzOC44NTk5Njg1IDI0LjMxODk4OTQsMzcuNTU3NzA5OSBDMjMuODc5MjM5MSwzNi45MDY3MjcgMjMuNTMxNDgxOCwzNi4xODk5MjMzIDIzLjI5MzY4NjYsMzUuNDI1MjY3NSBDMjMuMjEzMDIxNywzNS4xNjU4OSAyMy4xNDYwMjg5LDM0LjkwMDU1NTQgMjMuMDkxMzQwOSwzNC42MzA3Mjg2IEw0NC4xMjc4NzE0LDM0LjYzMDcyODYgQzQ1LjAyODQ2NiwzNC42MzA2MzA5IDQ1Ljc1ODY0ODgsMzMuOTAwNDQ4MSA0NS43NTg2NDg4LDMyLjk5OTg1MzUgQzQ1Ljc1ODY0ODgsMzIuMDk5MjU4OSA0NS4wMjg0NjYsMzEuMzY5MDc2MSA0NC4xMjc4NzE0LDMxLjM2OTA3NjEgTDIzLjA5MDU1OTYsMzEuMzY5MDc2MSBDMjMuMTk5MDU2NywzMC44MzM3MTk0IDIzLjM1OTcwMjgsMzAuMzE4MDg5NCAyMy41Njc1MTczLDI5LjgyNjQ4MzEgQzI0LjE4NTE5OSwyOC4zNjUyMzg2IDI1LjIyMTI0NDIsMjcuMTE2OTg0NCAyNi41MjM2MDA0LDI2LjIzNzQ4MzggQzI3LjE3NDU4MzMsMjUuNzk3NzMzNCAyNy44OTEzODcsMjUuNDQ5OTc2MiAyOC42NTYwNDI4LDI1LjIxMjI3ODYgQzI5LjQyMDg5MzksMjQuOTc0NDgzMyAzMC4yMzM0OTk0LDI0Ljg0NTk2NjUgMzEuMDgxMzU5MSwyNC44NDU5NjY1IEw0NC4xMjc3NzM3LDI0Ljg0NTk2NjUgQzQ1LjAyODM2ODMsMjQuODQ1OTY2NSA0NS43NTg2NDg4LDI0LjExNTc4MzcgNDUuNzU4NjQ4OCwyMy4yMTUxODkxIEM0NS43NTg2NDg4LDIyLjMxNDU5NDUgNDUuMDI4MzY4MywyMS41ODQ0MTE3IDQ0LjEyNzc3MzcsMjEuNTg0NDExNyBMMzEuMDgxMzU5MSwyMS41ODQ0MTE3IEMyOS41MDk2NjQzLDIxLjU4NDQxMTcgMjguMDAzOTg1OCwyMS45MDM4NDgzIDI2LjYzNzM3MTEsMjIuNDgyMDc2NSBDMjQuNTg2NjY3OCwyMy4zNDk4NTgzIDIyLjg0NjkwNDksMjQuNzk1MDg3MSAyMS42MTYzMjY3LDI2LjYxNjI5NiBDMjAuMzg1NjUwOCwyOC40MzYyMzU0IDE5LjY2NTEzNiwzMC42NDEzMzQ3IDE5LjY2NTgxOTEsMzMuMDAwMDQ4OCBDMTkuNjY1ODE5MSwzNC41NzE3NDM2IDE5Ljk4NTI1NjMsMzYuMDc3NDIyMiAyMC41NjM1ODIyLDM3LjQ0NDAzNjkgQzIxLjQzMTI2NjMsMzkuNDk0NzQwMiAyMi44NzY1OTI3LDQxLjIzNDUwMzEgMjQuNjk3NzA0LDQyLjQ2NTA4MTMgQzI2LjUxNzY0MzQsNDMuNjk1NzU3MiAyOC43MjI3NDI3LDQ0LjQxNTU4ODMgMzEuMDgxNDU2OCw0NC40MTU1ODgzIEw0NC4xMjc4NzE0LDQ0LjQxNTU4ODMgQzQ1LjAyODQ2Niw0NC40MTU1ODgzIDQ1Ljc1ODY0ODgsNDMuNjg1NDA1NSA0NS43NTg2NDg4LDQyLjc4NDgxMDkgQzQ1Ljc1ODY0ODgsNDEuODg0MjE2MyA0NS4wMjg1NjM2LDQxLjE1MzgzODIgNDQuMTI3OTY5LDQxLjE1MzgzODIgWiIgaWQ9InNldF9vZiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_in_set()


<p>Step 2: Expect that values in <code>category</code> should be in the set of <code>A</code>, <code>B</code>, <code>C</code>.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">category</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">A, B, C</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
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


In this example, every validation step will have a brief description that follows the pattern `"Step X: [auto-generated description]"`.

This is a simple example of template-based briefs. Later in this guide, we'll explore the full range of templating elements available for creating custom brief descriptions that precisely communicate what each validation step is checking.


# Step-level Briefs

You can also set briefs for individual validation steps:


``` python
(
    pb.Validate(data=data)
    .col_vals_gt(
        columns="value", value=5,
        brief="Check if values exceed minimum threshold of 5"
    )
    .col_vals_in_set(
        columns="category", set=["A", "B", "C"],
        brief="Verify categories are valid"
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-04-22|21:15:44</span>

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


<p>Check if values exceed minimum threshold of 5</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">5</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfaW5fc2V0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF92YWxzX2luX3NldCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMTcyNDE0KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuMTI3OTY5LDQxLjE1MzgzODIgTDMxLjA4MTQ1NjgsNDEuMTUzODM4MiBDMjkuOTUxMDc0OCw0MS4xNTM2NDI5IDI4Ljg4MjcwNTIsNDAuOTI1NjEzNCAyNy45MDc5ODg4LDQwLjUxMzY5NTMgQzI2LjQ0Njc0NDIsMzkuODk2MDEzNiAyNS4xOTg0OSwzOC44NTk5Njg1IDI0LjMxODk4OTQsMzcuNTU3NzA5OSBDMjMuODc5MjM5MSwzNi45MDY3MjcgMjMuNTMxNDgxOCwzNi4xODk5MjMzIDIzLjI5MzY4NjYsMzUuNDI1MjY3NSBDMjMuMjEzMDIxNywzNS4xNjU4OSAyMy4xNDYwMjg5LDM0LjkwMDU1NTQgMjMuMDkxMzQwOSwzNC42MzA3Mjg2IEw0NC4xMjc4NzE0LDM0LjYzMDcyODYgQzQ1LjAyODQ2NiwzNC42MzA2MzA5IDQ1Ljc1ODY0ODgsMzMuOTAwNDQ4MSA0NS43NTg2NDg4LDMyLjk5OTg1MzUgQzQ1Ljc1ODY0ODgsMzIuMDk5MjU4OSA0NS4wMjg0NjYsMzEuMzY5MDc2MSA0NC4xMjc4NzE0LDMxLjM2OTA3NjEgTDIzLjA5MDU1OTYsMzEuMzY5MDc2MSBDMjMuMTk5MDU2NywzMC44MzM3MTk0IDIzLjM1OTcwMjgsMzAuMzE4MDg5NCAyMy41Njc1MTczLDI5LjgyNjQ4MzEgQzI0LjE4NTE5OSwyOC4zNjUyMzg2IDI1LjIyMTI0NDIsMjcuMTE2OTg0NCAyNi41MjM2MDA0LDI2LjIzNzQ4MzggQzI3LjE3NDU4MzMsMjUuNzk3NzMzNCAyNy44OTEzODcsMjUuNDQ5OTc2MiAyOC42NTYwNDI4LDI1LjIxMjI3ODYgQzI5LjQyMDg5MzksMjQuOTc0NDgzMyAzMC4yMzM0OTk0LDI0Ljg0NTk2NjUgMzEuMDgxMzU5MSwyNC44NDU5NjY1IEw0NC4xMjc3NzM3LDI0Ljg0NTk2NjUgQzQ1LjAyODM2ODMsMjQuODQ1OTY2NSA0NS43NTg2NDg4LDI0LjExNTc4MzcgNDUuNzU4NjQ4OCwyMy4yMTUxODkxIEM0NS43NTg2NDg4LDIyLjMxNDU5NDUgNDUuMDI4MzY4MywyMS41ODQ0MTE3IDQ0LjEyNzc3MzcsMjEuNTg0NDExNyBMMzEuMDgxMzU5MSwyMS41ODQ0MTE3IEMyOS41MDk2NjQzLDIxLjU4NDQxMTcgMjguMDAzOTg1OCwyMS45MDM4NDgzIDI2LjYzNzM3MTEsMjIuNDgyMDc2NSBDMjQuNTg2NjY3OCwyMy4zNDk4NTgzIDIyLjg0NjkwNDksMjQuNzk1MDg3MSAyMS42MTYzMjY3LDI2LjYxNjI5NiBDMjAuMzg1NjUwOCwyOC40MzYyMzU0IDE5LjY2NTEzNiwzMC42NDEzMzQ3IDE5LjY2NTgxOTEsMzMuMDAwMDQ4OCBDMTkuNjY1ODE5MSwzNC41NzE3NDM2IDE5Ljk4NTI1NjMsMzYuMDc3NDIyMiAyMC41NjM1ODIyLDM3LjQ0NDAzNjkgQzIxLjQzMTI2NjMsMzkuNDk0NzQwMiAyMi44NzY1OTI3LDQxLjIzNDUwMzEgMjQuNjk3NzA0LDQyLjQ2NTA4MTMgQzI2LjUxNzY0MzQsNDMuNjk1NzU3MiAyOC43MjI3NDI3LDQ0LjQxNTU4ODMgMzEuMDgxNDU2OCw0NC40MTU1ODgzIEw0NC4xMjc4NzE0LDQ0LjQxNTU4ODMgQzQ1LjAyODQ2Niw0NC40MTU1ODgzIDQ1Ljc1ODY0ODgsNDMuNjg1NDA1NSA0NS43NTg2NDg4LDQyLjc4NDgxMDkgQzQ1Ljc1ODY0ODgsNDEuODg0MjE2MyA0NS4wMjg1NjM2LDQxLjE1MzgzODIgNDQuMTI3OTY5LDQxLjE1MzgzODIgWiIgaWQ9InNldF9vZiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_in_set()


<p>Verify categories are valid</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">category</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">A, B, C</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
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


Local briefs override any global briefs that might be set.


# Brief Templating

Briefs support templating elements that get replaced with specific values:

- `{auto}`: an auto-generated description of the validation
- `{step}`: the step number in the validation plan
- `{col}`: the column name(s) being validated
- `{value}`: the comparison value used in the validation (when applicable)
- `{thresholds}`: a short summary of thresholds levels set (or unset) for the step
- `{segment}`, `{segment_column}`, `{segment_value}`: information on the step's segment

Here's how to use these templates:


``` python
(
    pb.Validate(data=data)
    .col_vals_gt(
        columns="value", value=5,
        brief="Step {step}: Checking column '{col}' for values `> 5`"
    )
    .col_vals_in_set(
        columns="category", set=["A", "B", "C"],
        brief="{auto} **(Step {step})**"
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-04-22|21:15:45</span>

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


<p>Step 1: Checking column 'value' for values <code>> 5</code></p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">5</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
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
I5MzY4NjYsMzUuNDI1MjY3NSBDMjMuMjEzMDIxNywzNS4xNjU4OSAyMy4xNDYwMjg5LDM0LjkwMDU1NTQgMjMuMDkxMzQwOSwzNC42MzA3Mjg2IEw0NC4xMjc4NzE0LDM0LjYzMDcyODYgQzQ1LjAyODQ2NiwzNC42MzA2MzA5IDQ1Ljc1ODY0ODgsMzMuOTAwNDQ4MSA0NS43NTg2NDg4LDMyLjk5OTg1MzUgQzQ1Ljc1ODY0ODgsMzIuMDk5MjU4OSA0NS4wMjg0NjYsMzEuMzY5MDc2MSA0NC4xMjc4NzE0LDMxLjM2OTA3NjEgTDIzLjA5MDU1OTYsMzEuMzY5MDc2MSBDMjMuMTk5MDU2NywzMC44MzM3MTk0IDIzLjM1OTcwMjgsMzAuMzE4MDg5NCAyMy41Njc1MTczLDI5LjgyNjQ4MzEgQzI0LjE4NTE5OSwyOC4zNjUyMzg2IDI1LjIyMTI0NDIsMjcuMTE2OTg0NCAyNi41MjM2MDA0LDI2LjIzNzQ4MzggQzI3LjE3NDU4MzMsMjUuNzk3NzMzNCAyNy44OTEzODcsMjUuNDQ5OTc2MiAyOC42NTYwNDI4LDI1LjIxMjI3ODYgQzI5LjQyMDg5MzksMjQuOTc0NDgzMyAzMC4yMzM0OTk0LDI0Ljg0NTk2NjUgMzEuMDgxMzU5MSwyNC44NDU5NjY1IEw0NC4xMjc3NzM3LDI0Ljg0NTk2NjUgQzQ1LjAyODM2ODMsMjQuODQ1OTY2NSA0NS43NTg2NDg4LDI0LjExNTc4MzcgNDUuNzU4NjQ4OCwyMy4yMTUxODkxIEM0NS43NTg2NDg4LDIyLjMxNDU5NDUgNDUuMDI4MzY4MywyMS41ODQ0MTE3IDQ0LjEyNzc3MzcsMjEuNTg0NDExNyBMMzEuMDgxMzU5MSwyMS41ODQ0MTE3IEMyOS41MDk2NjQzLDIxLjU4NDQxMTcgMjguMDAzOTg1OCwyMS45MDM4NDgzIDI2LjYzNzM3MTEsMjIuNDgyMDc2NSBDMjQuNTg2NjY3OCwyMy4zNDk4NTgzIDIyLjg0NjkwNDksMjQuNzk1MDg3MSAyMS42MTYzMjY3LDI2LjYxNjI5NiBDMjAuMzg1NjUwOCwyOC40MzYyMzU0IDE5LjY2NTEzNiwzMC42NDEzMzQ3IDE5LjY2NTgxOTEsMzMuMDAwMDQ4OCBDMTkuNjY1ODE5MSwzNC41NzE3NDM2IDE5Ljk4NTI1NjMsMzYuMDc3NDIyMiAyMC41NjM1ODIyLDM3LjQ0NDAzNjkgQzIxLjQzMTI2NjMsMzkuNDk0NzQwMiAyMi44NzY1OTI3LDQxLjIzNDUwMzEgMjQuNjk3NzA0LDQyLjQ2NTA4MTMgQzI2LjUxNzY0MzQsNDMuNjk1NzU3MiAyOC43MjI3NDI3LDQ0LjQxNTU4ODMgMzEuMDgxNDU2OCw0NC40MTU1ODgzIEw0NC4xMjc4NzE0LDQ0LjQxNTU4ODMgQzQ1LjAyODQ2Niw0NC40MTU1ODgzIDQ1Ljc1ODY0ODgsNDMuNjg1NDA1NSA0NS43NTg2NDg4LDQyLjc4NDgxMDkgQzQ1Ljc1ODY0ODgsNDEuODg0MjE2MyA0NS4wMjg1NjM2LDQxLjE1MzgzODIgNDQuMTI3OTY5LDQxLjE1MzgzODIgWiIgaWQ9InNldF9vZiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_in_set()


<p>Expect that values in <code>category</code> should be in the set of <code>A</code>, <code>B</code>, <code>C</code>. <strong>(Step 2)</strong></p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">category</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">A, B, C</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
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


These template elements make briefs highly flexible and customizable. You can combine multiple templating elements in a single brief to create descriptive yet concise validation step descriptions. The templates help maintain consistency across your validation reports while providing enough detail to understand what each step is checking.

Note that not all templating elements will be relevant for every validation step. For instance, `{value}` is only applicable to validation functions that hold a comparison value like <a href="../../reference/Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt" class="gdls-link"><code>Validate.col_vals_gt()</code></a>. If you include a templating element that isn't relevant to a particular step, it will not be replaced with a corresponding value.

Briefs support the use of Markdown formatting, allowing you to add emphasis with **bold** or *italic* text, include `inline code` formatting, or other Markdown elements to make your briefs more visually distinctive and informative. This can be especially helpful when you want certain parts of your briefs to stand out in the validation report.


# Automatic Briefs

If you want Pointblank to generate briefs for you automatically, you can set `brief=True`. Here, we'll make that setting at the global level (by using `Validate(brief=True)`):


``` python
(
    pb.Validate(
        data=data,

        # Setting for automatically generated briefs ---
        brief=True
    )
    .col_vals_gt(columns="value", value=5)
    .col_vals_in_set(columns="category", set=["A", "B", "C"])
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-04-22|21:15:45</span>

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


<p>Expect that values in <code>value</code> should be > <code>5</code>.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">5</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfaW5fc2V0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF92YWxzX2luX3NldCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMTcyNDE0KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuMTI3OTY5LDQxLjE1MzgzODIgTDMxLjA4MTQ1NjgsNDEuMTUzODM4MiBDMjkuOTUxMDc0OCw0MS4xNTM2NDI5IDI4Ljg4MjcwNTIsNDAuOTI1NjEzNCAyNy45MDc5ODg4LDQwLjUxMzY5NTMgQzI2LjQ0Njc0NDIsMzkuODk2MDEzNiAyNS4xOTg0OSwzOC44NTk5Njg1IDI0LjMxODk4OTQsMzcuNTU3NzA5OSBDMjMuODc5MjM5MSwzNi45MDY3MjcgMjMuNTMxNDgxOCwzNi4xODk5MjMzIDIzLjI5MzY4NjYsMzUuNDI1MjY3NSBDMjMuMjEzMDIxNywzNS4xNjU4OSAyMy4xNDYwMjg5LDM0LjkwMDU1NTQgMjMuMDkxMzQwOSwzNC42MzA3Mjg2IEw0NC4xMjc4NzE0LDM0LjYzMDcyODYgQzQ1LjAyODQ2NiwzNC42MzA2MzA5IDQ1Ljc1ODY0ODgsMzMuOTAwNDQ4MSA0NS43NTg2NDg4LDMyLjk5OTg1MzUgQzQ1Ljc1ODY0ODgsMzIuMDk5MjU4OSA0NS4wMjg0NjYsMzEuMzY5MDc2MSA0NC4xMjc4NzE0LDMxLjM2OTA3NjEgTDIzLjA5MDU1OTYsMzEuMzY5MDc2MSBDMjMuMTk5MDU2NywzMC44MzM3MTk0IDIzLjM1OTcwMjgsMzAuMzE4MDg5NCAyMy41Njc1MTczLDI5LjgyNjQ4MzEgQzI0LjE4NTE5OSwyOC4zNjUyMzg2IDI1LjIyMTI0NDIsMjcuMTE2OTg0NCAyNi41MjM2MDA0LDI2LjIzNzQ4MzggQzI3LjE3NDU4MzMsMjUuNzk3NzMzNCAyNy44OTEzODcsMjUuNDQ5OTc2MiAyOC42NTYwNDI4LDI1LjIxMjI3ODYgQzI5LjQyMDg5MzksMjQuOTc0NDgzMyAzMC4yMzM0OTk0LDI0Ljg0NTk2NjUgMzEuMDgxMzU5MSwyNC44NDU5NjY1IEw0NC4xMjc3NzM3LDI0Ljg0NTk2NjUgQzQ1LjAyODM2ODMsMjQuODQ1OTY2NSA0NS43NTg2NDg4LDI0LjExNTc4MzcgNDUuNzU4NjQ4OCwyMy4yMTUxODkxIEM0NS43NTg2NDg4LDIyLjMxNDU5NDUgNDUuMDI4MzY4MywyMS41ODQ0MTE3IDQ0LjEyNzc3MzcsMjEuNTg0NDExNyBMMzEuMDgxMzU5MSwyMS41ODQ0MTE3IEMyOS41MDk2NjQzLDIxLjU4NDQxMTcgMjguMDAzOTg1OCwyMS45MDM4NDgzIDI2LjYzNzM3MTEsMjIuNDgyMDc2NSBDMjQuNTg2NjY3OCwyMy4zNDk4NTgzIDIyLjg0NjkwNDksMjQuNzk1MDg3MSAyMS42MTYzMjY3LDI2LjYxNjI5NiBDMjAuMzg1NjUwOCwyOC40MzYyMzU0IDE5LjY2NTEzNiwzMC42NDEzMzQ3IDE5LjY2NTgxOTEsMzMuMDAwMDQ4OCBDMTkuNjY1ODE5MSwzNC41NzE3NDM2IDE5Ljk4NTI1NjMsMzYuMDc3NDIyMiAyMC41NjM1ODIyLDM3LjQ0NDAzNjkgQzIxLjQzMTI2NjMsMzkuNDk0NzQwMiAyMi44NzY1OTI3LDQxLjIzNDUwMzEgMjQuNjk3NzA0LDQyLjQ2NTA4MTMgQzI2LjUxNzY0MzQsNDMuNjk1NzU3MiAyOC43MjI3NDI3LDQ0LjQxNTU4ODMgMzEuMDgxNDU2OCw0NC40MTU1ODgzIEw0NC4xMjc4NzE0LDQ0LjQxNTU4ODMgQzQ1LjAyODQ2Niw0NC40MTU1ODgzIDQ1Ljc1ODY0ODgsNDMuNjg1NDA1NSA0NS43NTg2NDg4LDQyLjc4NDgxMDkgQzQ1Ljc1ODY0ODgsNDEuODg0MjE2MyA0NS4wMjg1NjM2LDQxLjE1MzgzODIgNDQuMTI3OTY5LDQxLjE1MzgzODIgWiIgaWQ9InNldF9vZiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_in_set()


<p>Expect that values in <code>category</code> should be in the set of <code>A</code>, <code>B</code>, <code>C</code>.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">category</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">A, B, C</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
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


Automatic briefs are descriptive and include information about what's being validated, including the column names and the validation conditions.


# Briefs Localized to a Specified Language

When using the `lang=` parameter in [Validate](../../reference/Validate.md#pointblank.Validate), automatically generated briefs will be created in the specified language (along with other elements of the validation report table):


``` python
(
    pb.Validate(
        data=data,

        # Setting the language as Spanish ---
        lang="es",

        # Automatically generate all briefs in Spanish
        brief=True
    )
    .col_vals_gt(columns="value", value=5)
    .col_vals_in_set(columns="category", set=["A", "B", "C"])
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Validación de Pointblank</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-04-22|21:15:45</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">PASO</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNAS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALORES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNID.</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASA</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FALLO</th>
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


<p>Se espera que los valores en <code>value</code> sean > <code>5</code>.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">5</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1,00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0,00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfaW5fc2V0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF92YWxzX2luX3NldCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMTcyNDE0KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuMTI3OTY5LDQxLjE1MzgzODIgTDMxLjA4MTQ1NjgsNDEuMTUzODM4MiBDMjkuOTUxMDc0OCw0MS4xNTM2NDI5IDI4Ljg4MjcwNTIsNDAuOTI1NjEzNCAyNy45MDc5ODg4LDQwLjUxMzY5NTMgQzI2LjQ0Njc0NDIsMzkuODk2MDEzNiAyNS4xOTg0OSwzOC44NTk5Njg1IDI0LjMxODk4OTQsMzcuNTU3NzA5OSBDMjMuODc5MjM5MSwzNi45MDY3MjcgMjMuNTMxNDgxOCwzNi4xODk5MjMzIDIzLjI5MzY4NjYsMzUuNDI1MjY3NSBDMjMuMjEzMDIxNywzNS4xNjU4OSAyMy4xNDYwMjg5LDM0LjkwMDU1NTQgMjMuMDkxMzQwOSwzNC42MzA3Mjg2IEw0NC4xMjc4NzE0LDM0LjYzMDcyODYgQzQ1LjAyODQ2NiwzNC42MzA2MzA5IDQ1Ljc1ODY0ODgsMzMuOTAwNDQ4MSA0NS43NTg2NDg4LDMyLjk5OTg1MzUgQzQ1Ljc1ODY0ODgsMzIuMDk5MjU4OSA0NS4wMjg0NjYsMzEuMzY5MDc2MSA0NC4xMjc4NzE0LDMxLjM2OTA3NjEgTDIzLjA5MDU1OTYsMzEuMzY5MDc2MSBDMjMuMTk5MDU2NywzMC44MzM3MTk0IDIzLjM1OTcwMjgsMzAuMzE4MDg5NCAyMy41Njc1MTczLDI5LjgyNjQ4MzEgQzI0LjE4NTE5OSwyOC4zNjUyMzg2IDI1LjIyMTI0NDIsMjcuMTE2OTg0NCAyNi41MjM2MDA0LDI2LjIzNzQ4MzggQzI3LjE3NDU4MzMsMjUuNzk3NzMzNCAyNy44OTEzODcsMjUuNDQ5OTc2MiAyOC42NTYwNDI4LDI1LjIxMjI3ODYgQzI5LjQyMDg5MzksMjQuOTc0NDgzMyAzMC4yMzM0OTk0LDI0Ljg0NTk2NjUgMzEuMDgxMzU5MSwyNC44NDU5NjY1IEw0NC4xMjc3NzM3LDI0Ljg0NTk2NjUgQzQ1LjAyODM2ODMsMjQuODQ1OTY2NSA0NS43NTg2NDg4LDI0LjExNTc4MzcgNDUuNzU4NjQ4OCwyMy4yMTUxODkxIEM0NS43NTg2NDg4LDIyLjMxNDU5NDUgNDUuMDI4MzY4MywyMS41ODQ0MTE3IDQ0LjEyNzc3MzcsMjEuNTg0NDExNyBMMzEuMDgxMzU5MSwyMS41ODQ0MTE3IEMyOS41MDk2NjQzLDIxLjU4NDQxMTcgMjguMDAzOTg1OCwyMS45MDM4NDgzIDI2LjYzNzM3MTEsMjIuNDgyMDc2NSBDMjQuNTg2NjY3OCwyMy4zNDk4NTgzIDIyLjg0NjkwNDksMjQuNzk1MDg3MSAyMS42MTYzMjY3LDI2LjYxNjI5NiBDMjAuMzg1NjUwOCwyOC40MzYyMzU0IDE5LjY2NTEzNiwzMC42NDEzMzQ3IDE5LjY2NTgxOTEsMzMuMDAwMDQ4OCBDMTkuNjY1ODE5MSwzNC41NzE3NDM2IDE5Ljk4NTI1NjMsMzYuMDc3NDIyMiAyMC41NjM1ODIyLDM3LjQ0NDAzNjkgQzIxLjQzMTI2NjMsMzkuNDk0NzQwMiAyMi44NzY1OTI3LDQxLjIzNDUwMzEgMjQuNjk3NzA0LDQyLjQ2NTA4MTMgQzI2LjUxNzY0MzQsNDMuNjk1NzU3MiAyOC43MjI3NDI3LDQ0LjQxNTU4ODMgMzEuMDgxNDU2OCw0NC40MTU1ODgzIEw0NC4xMjc4NzE0LDQ0LjQxNTU4ODMgQzQ1LjAyODQ2Niw0NC40MTU1ODgzIDQ1Ljc1ODY0ODgsNDMuNjg1NDA1NSA0NS43NTg2NDg4LDQyLjc4NDgxMDkgQzQ1Ljc1ODY0ODgsNDEuODg0MjE2MyA0NS4wMjg1NjM2LDQxLjE1MzgzODIgNDQuMTI3OTY5LDQxLjE1MzgzODIgWiIgaWQ9InNldF9vZiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_in_set()


<p>Se espera que los valores en <code>category</code> estén en el conjunto de <code>A</code>, <code>B</code>, <code>C</code>.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">category</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">A, B, C</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1,00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0,00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody>
</table>


When using the `lang=` parameter in combination with the `{auto}` templating element, the auto-generated portion of the brief will also be translated to the specified language. This makes it possible to create fully localized validation reports where both custom text and auto-generated descriptions appear in the same language.

Pointblank supports several languages for localized briefs, including French (`"fr"`), German (`"de"`), Spanish (`"es"`), Italian (`"it"`), and Portuguese (`"pt"`). For the complete list of supported languages, refer to the [Validate](../../reference/Validate.md#pointblank.Validate) documentation.


# Disabling Briefs

If you've set a global brief but want to disable it for specific validation steps, you can set `brief=False`:


``` python
(
    pb.Validate(
        data=data,

        # Global brief template ---
        brief="Step {step}: {auto}"
    )
    .col_vals_gt(columns="value", value=5)  # This step uses the global brief setting
    .col_vals_in_set(
        columns="category",
        set=["A", "B", "C"],

        # No brief for this step ---
        brief=False
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-04-22|21:15:45</span>

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


<p>Step 1: Expect that values in <code>value</code> should be > <code>5</code>.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">5</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfaW5fc2V0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF92YWxzX2luX3NldCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMTcyNDE0KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuMTI3OTY5LDQxLjE1MzgzODIgTDMxLjA4MTQ1NjgsNDEuMTUzODM4MiBDMjkuOTUxMDc0OCw0MS4xNTM2NDI5IDI4Ljg4MjcwNTIsNDAuOTI1NjEzNCAyNy45MDc5ODg4LDQwLjUxMzY5NTMgQzI2LjQ0Njc0NDIsMzkuODk2MDEzNiAyNS4xOTg0OSwzOC44NTk5Njg1IDI0LjMxODk4OTQsMzcuNTU3NzA5OSBDMjMuODc5MjM5MSwzNi45MDY3MjcgMjMuNTMxNDgxOCwzNi4xODk5MjMzIDIzLjI5MzY4NjYsMzUuNDI1MjY3NSBDMjMuMjEzMDIxNywzNS4xNjU4OSAyMy4xNDYwMjg5LDM0LjkwMDU1NTQgMjMuMDkxMzQwOSwzNC42MzA3Mjg2IEw0NC4xMjc4NzE0LDM0LjYzMDcyODYgQzQ1LjAyODQ2NiwzNC42MzA2MzA5IDQ1Ljc1ODY0ODgsMzMuOTAwNDQ4MSA0NS43NTg2NDg4LDMyLjk5OTg1MzUgQzQ1Ljc1ODY0ODgsMzIuMDk5MjU4OSA0NS4wMjg0NjYsMzEuMzY5MDc2MSA0NC4xMjc4NzE0LDMxLjM2OTA3NjEgTDIzLjA5MDU1OTYsMzEuMzY5MDc2MSBDMjMuMTk5MDU2NywzMC44MzM3MTk0IDIzLjM1OTcwMjgsMzAuMzE4MDg5NCAyMy41Njc1MTczLDI5LjgyNjQ4MzEgQzI0LjE4NTE5OSwyOC4zNjUyMzg2IDI1LjIyMTI0NDIsMjcuMTE2OTg0NCAyNi41MjM2MDA0LDI2LjIzNzQ4MzggQzI3LjE3NDU4MzMsMjUuNzk3NzMzNCAyNy44OTEzODcsMjUuNDQ5OTc2MiAyOC42NTYwNDI4LDI1LjIxMjI3ODYgQzI5LjQyMDg5MzksMjQuOTc0NDgzMyAzMC4yMzM0OTk0LDI0Ljg0NTk2NjUgMzEuMDgxMzU5MSwyNC44NDU5NjY1IEw0NC4xMjc3NzM3LDI0Ljg0NTk2NjUgQzQ1LjAyODM2ODMsMjQuODQ1OTY2NSA0NS43NTg2NDg4LDI0LjExNTc4MzcgNDUuNzU4NjQ4OCwyMy4yMTUxODkxIEM0NS43NTg2NDg4LDIyLjMxNDU5NDUgNDUuMDI4MzY4MywyMS41ODQ0MTE3IDQ0LjEyNzc3MzcsMjEuNTg0NDExNyBMMzEuMDgxMzU5MSwyMS41ODQ0MTE3IEMyOS41MDk2NjQzLDIxLjU4NDQxMTcgMjguMDAzOTg1OCwyMS45MDM4NDgzIDI2LjYzNzM3MTEsMjIuNDgyMDc2NSBDMjQuNTg2NjY3OCwyMy4zNDk4NTgzIDIyLjg0NjkwNDksMjQuNzk1MDg3MSAyMS42MTYzMjY3LDI2LjYxNjI5NiBDMjAuMzg1NjUwOCwyOC40MzYyMzU0IDE5LjY2NTEzNiwzMC42NDEzMzQ3IDE5LjY2NTgxOTEsMzMuMDAwMDQ4OCBDMTkuNjY1ODE5MSwzNC41NzE3NDM2IDE5Ljk4NTI1NjMsMzYuMDc3NDIyMiAyMC41NjM1ODIyLDM3LjQ0NDAzNjkgQzIxLjQzMTI2NjMsMzkuNDk0NzQwMiAyMi44NzY1OTI3LDQxLjIzNDUwMzEgMjQuNjk3NzA0LDQyLjQ2NTA4MTMgQzI2LjUxNzY0MzQsNDMuNjk1NzU3MiAyOC43MjI3NDI3LDQ0LjQxNTU4ODMgMzEuMDgxNDU2OCw0NC40MTU1ODgzIEw0NC4xMjc4NzE0LDQ0LjQxNTU4ODMgQzQ1LjAyODQ2Niw0NC40MTU1ODgzIDQ1Ljc1ODY0ODgsNDMuNjg1NDA1NSA0NS43NTg2NDg4LDQyLjc4NDgxMDkgQzQ1Ljc1ODY0ODgsNDEuODg0MjE2MyA0NS4wMjg1NjM2LDQxLjE1MzgzODIgNDQuMTI3OTY5LDQxLjE1MzgzODIgWiIgaWQ9InNldF9vZiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_in_set()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">category</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">A, B, C</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
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


# Practical Example: Comprehensive Validation with Briefs

In real-world data validation scenarios, you'll likely work with more complex datasets and apply various types of validation checks. This final example brings together many of the brief-generating techniques we've covered, showing how you can mix different approaches in a single validation workflow.


``` python
# Create a slightly larger dataset
data_2 = pl.DataFrame({
    "id": [1, 2, 3, 4, 5, 6, 7, 8],
    "value": [10, 20, 30, 40, 50, 60, 70, 80],
    "ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "category": ["A", "B", "C", "A", "B", "C", "A", "B"],
    "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04",
             "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08"]
})

(
    pb.Validate(data=data_2)
    .col_vals_gt(
        columns="value", value=0,

        # Plaintext brief ---
        brief="All values must be positive."
    )
    .col_vals_between(
        columns="ratio", left=0, right=1,

        # Template-based brief ---
        brief="**Step {step}**: Ratios should be between `0` and `1`."
    )
    .col_vals_in_set(
        columns="category", set=["A", "B", "C"],

        # Automatically generated brief ---
        brief=True
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-04-22|21:15:45</span>

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


<p>All values must be positive.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">value</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">8</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">8<br />
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfYmV0d2VlbjwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19iZXR3ZWVuIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4yMDY4OTcpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS45OTM0ODQsMjEuOTY4NzUgQzEwLjk2MjIzNCwyMi4wODIwMzEgMTAuMTg4Nzk3LDIyLjk2NDg0NCAxMC4yMTIyMzQsMjQgTDEwLjIxMjIzNCw0MiBDMTAuMjAwNTE1LDQyLjcyMjY1NiAxMC41Nzk0MjIsNDMuMzkwNjI1IDExLjIwNDQyMiw0My43NTM5MDYgQzExLjgyNTUxNSw0NC4xMjEwOTQgMTIuNTk4OTUzLDQ0LjEyMTA5NCAxMy4yMjAwNDcsNDMuNzUzOTA2IEMxMy44NDUwNDcsNDMuMzkwNjI1IDE0LjIyMzk1Myw0Mi43MjI2NTYgMTQuMjEyMjM0LDQyIEwxNC4yMTIyMzQsMjQgQzE0LjIyMDA0NywyMy40NTcwMzEgMTQuMDA5MTA5LDIyLjkzNzUgMTMuNjI2Mjk3LDIyLjU1NDY4OCBDMTMuMjQzNDg0LDIyLjE3MTg3NSAxMi43MjM5NTMsMjEuOTYwOTM4IDEyLjE4MDk4NCwyMS45Njg3NSBDMTIuMTE4NDg0LDIxLjk2NDg0NCAxMi4wNTU5ODQsMjEuOTY0ODQ0IDExLjk5MzQ4NCwyMS45Njg3NSBaIE01NS45OTM0ODQsMjEuOTY4NzUgQzU0Ljk2MjIzNCwyMi4wODIwMzEgNTQuMTg4Nzk3LDIyLjk2NDg0NCA1NC4yMTIyMzQsMjQgTDU0LjIxMjIzNCw0MiBDNTQuMjAwNTE1LDQyLjcyMjY1NiA1NC41Nzk0MjIsNDMuMzkwNjI1IDU1LjIwNDQyMiw0My43NTM5MDYgQzU1LjgyNTUxNSw0NC4xMjEwOTQgNTYuNTk4OTUzLDQ0LjEyMTA5NCA1Ny4yMjAwNDcsNDMuNzUzOTA2IEM1Ny44NDUwNDcsNDMuMzkwNjI1IDU4LjIyMzk1Myw0Mi43MjI2NTYgNTguMjEyMjM0LDQyIEw1OC4yMTIyMzQsMjQgQzU4LjIyMDA0NywyMy40NTcwMzEgNTguMDA5MTA5LDIyLjkzNzUgNTcuNjI2Mjk3LDIyLjU1NDY4OCBDNTcuMjQzNDg0LDIyLjE3MTg3NSA1Ni43MjM5NTMsMjEuOTYwOTM4IDU2LjE4MDk4NCwyMS45Njg3NSBDNTYuMTE4NDg0LDIxLjk2NDg0NCA1Ni4wNTU5ODQsMjEuOTY0ODQ0IDU1Ljk5MzQ4NCwyMS45Njg3NSBaIE0xNi4yMTIyMzQsMjIgQzE1LjY2MTQ1MywyMiAxNS4yMTIyMzQsMjIuNDQ5MjE5IDE1LjIxMjIzNCwyMyBDMTUuMjEyMjM0LDIzLjU1MDc4MSAxNS42NjE0NTMsMjQgMTYuMjEyMjM0LDI0IEMxNi43NjMwMTUsMjQgMTcuMjEyMjM0LDIzLjU1MDc4MSAxNy4yMTIyMzQsMjMgQzE3LjIxMjIzNCwyMi40NDkyMTkgMTYuNzYzMDE1LDIyIDE2LjIxMjIzNCwyMiBaIE0yMC4yMTIyMzQsMjIgQzE5LjY2MTQ1MywyMiAxOS4yMTIyMzQsMjIuNDQ5MjE5IDE5LjIxMjIzNCwyMyBDMTkuMjEyMjM0LDIzLjU1MDc4MSAxOS42NjE0NTMsMjQgMjAuMjEyMjM0LDI0IEMyMC43NjMwMTUsMjQgMjEuMjEyMjM0LDIzLjU1MDc4MSAyMS4yMTIyMzQsMjMgQzIxLjIxMjIzNCwyMi40NDkyMTkgMjAuNzYzMDE1LDIyIDIwLjIxMjIzNCwyMiBaIE0yNC4yMTIyMzQsMjIgQzIzLjY2MTQ1MywyMiAyMy4yMTIyMzQsMjIuNDQ5MjE5IDIzLjIxMjIzNCwyMyBDMjMuMjEyMjM0LDIzLjU1MDc4MSAyMy42NjE0NTMsMjQgMjQuMjEyMjM0LDI0IEMyNC43NjMwMTUsMjQgMjUuMjEyMjM0LDIzLjU1MDc4MSAyNS4yMTIyMzQsMjMgQzI1LjIxMjIzNCwyMi40NDkyMTkgMjQuNzYzMDE1LDIyIDI0LjIxMjIzNCwyMiBaIE0yOC4yMTIyMzQsMjIgQzI3LjY2MTQ1MywyMiAyNy4yMTIyMzQsMjIuNDQ5MjE5IDI3LjIxMjIzNCwyMyBDMjcuMjEyMjM0LDIzLjU1MDc4MSAyNy42NjE0NTMsMjQgMjguMjEyMjM0LDI0IEMyOC43NjMwMTUsMjQgMjkuMjEyMjM0LDIzLjU1MDc4MSAyOS4yMTIyMzQsMjMgQzI5LjIxMjIzNCwyMi40NDkyMTkgMjguNzYzMDE1LDIyIDI4LjIxMjIzNCwyMiBaIE0zMi4yMTIyMzQsMjIgQzMxLjY2MTQ1MywyMiAzMS4yMTIyMzQsMjIuNDQ5MjE5IDMxLjIxMjIzNCwyMyBDMzEuMjEyMjM0LDIzLjU1MDc4MSAzMS42NjE0NTMsMjQgMzIuMjEyMjM0LDI0IEMzMi43NjMwMTUsMjQgMzMuMjEyMjM0LDIzLjU1MDc4MSAzMy4yMTIyMzQsMjMgQzMzLjIxMjIzNCwyMi40NDkyMTkgMzIuNzYzMDE1LDIyIDMyLjIxMjIzNCwyMiBaIE0zNi4yMTIyMzQsMjIgQzM1LjY2MTQ1MywyMiAzNS4yMTIyMzQsMjIuNDQ5MjE5IDM1LjIxMjIzNCwyMyBDMzUuMjEyMjM0LDIzLjU1MDc4MSAzNS42NjE0NTMsMjQgMzYuMjEyMjM0LDI0IEMzNi43NjMwMTUsMjQgMzcuMjEyMjM0LDIzLjU1MDc4MSAzNy4yMTIyMzQsMjMgQzM3LjIxMjIzNCwyMi40NDkyMTkgMzYuNzYzMDE1LDIyIDM2LjIxMjIzNCwyMiBaIE00MC4yMTIyMzQsMjIgQzM5LjY2MTQ1MywyMiAzOS4yMTIyMzQsMjIuNDQ5MjE5IDM5LjIxMjIzNCwyMyBDMzkuMjEyMjM0LDIzLjU1MDc4MSAzOS42NjE0NTMsMjQgNDAuMjEyMjM0LDI0IEM0MC43NjMwMTUsMjQgNDEuMjEyMjM0LDIzLjU1MDc4MSA0MS4yMTIyMzQsMjMgQzQxLjIxMjIzNCwyMi40NDkyMTkgNDAuNzYzMDE1LDIyIDQwLjIxMjIzNCwyMiBaIE00NC4yMTIyMzQsMjIgQzQzLjY2MTQ1MywyMiA0My4yMTIyMzQsMjIuNDQ5MjE5IDQzLjIxMjIzNCwyMyBDNDMuMjEyMjM0LDIzLjU1MDc4MSA0My42NjE0NTMsMjQgNDQuMjEyMjM0LDI0IEM0NC43NjMwMTUsMjQgNDUuMjEyMjM0LDIzLjU1MDc4MSA0NS4yMTIyMzQsMjMgQzQ1LjIxMjIzNCwyMi40NDkyMTkgNDQuNzYzMDE1LDIyIDQ0LjIxMjIzNCwyMiBaIE00OC4yMTIyMzQsMjIgQzQ3LjY2MTQ1MywyMiA0Ny4yMTIyMzQsMjIuNDQ5MjE5IDQ3LjIxMjIzNCwyMyBDNDcuMjEyMjM0LDIzLjU1MDc4MSA0Ny42NjE0NTMsMjQgNDguMjEyMjM0LDI0IEM0OC43NjMwMTUsMjQgNDkuMjEyMjM0LDIzLjU1MDc4MSA0OS4yMTIyMzQsMjMgQzQ5LjIxMjIzNCwyMi40NDkyMTkgNDguNzYzMDE1LDIyIDQ4LjIxMjIzNCwyMiBaIE01Mi4yMTIyMzQsMjIgQzUxLjY2MTQ1MywyMiA1MS4yMTIyMzQsMjIuNDQ5MjE5IDUxLjIxMjIzNCwyMyBDNTEuMjEyMjM0LDIzLjU1MDc4MSA1MS42NjE0NTMsMjQgNTIuMjEyMjM0LDI0IEM1Mi43NjMwMTUsMjQgNTMuMjEyMjM0LDIzLjU1MDc4MSA1My4yMTIyMzQsMjMgQzUzLjIxMjIzNCwyMi40NDkyMTkgNTIuNzYzMDE1LDIyIDUyLjIxMjIzNCwyMiBaIE0yMS40NjIyMzQsMjcuOTY4NzUgQzIxLjQxOTI2NSwyNy45NzY1NjMgMjEuMzc2Mjk3LDI3Ljk4ODI4MSAyMS4zMzcyMzQsMjggQzIxLjE3NzA3OCwyOC4wMjczNDQgMjEuMDI4NjQsMjguMDg5ODQ0IDIwLjg5OTczNCwyOC4xODc1IEwxNS42MTg0ODQsMzIuMTg3NSBDMTUuMzU2NzY1LDMyLjM3NSAxNS4yMDA1MTUsMzIuNjc5Njg4IDE1LjIwMDUxNSwzMyBDMTUuMjAwNTE1LDMzLjMyMDMxMyAxNS4zNTY3NjUsMzMuNjI1IDE1LjYxODQ4NCwzMy44MTI1IEwyMC44OTk3MzQsMzcuODEyNSBDMjEuMzQ4OTUzLDM4LjE0ODQzOCAyMS45ODU2NzIsMzguMDU4NTk0IDIyLjMyMTYwOSwzNy42MDkzNzUgQzIyLjY1NzU0NywzNy4xNjAxNTYgMjIuNTY3NzAzLDM2LjUyMzQzOCAyMi4xMTg0ODQsMzYuMTg3NSBMMTkuMjEyMjM0LDM0IEw0OS4yMTIyMzQsMzQgTDQ2LjMwNTk4NCwzNi4xODc1IEM0NS44NTY3NjUsMzYuNTIzNDM4IDQ1Ljc2NjkyMiwzNy4xNjAxNTYgNDYuMTAyODU5LDM3LjYwOTM3NSBDNDYuNDM4Nzk3LDM4LjA1ODU5NCA0Ny4wNzU1MTUsMzguMTQ4NDM4IDQ3LjUyNDczNCwzNy44MTI1IEw1Mi44MDU5ODQsMzMuODEyNSBDNTMuMDY3NzAzLDMzLjYyNSA1My4yMjM5NTMsMzMuMzIwMzEzIDUzLjIyMzk1MywzMyBDNTMuMjIzOTUzLDMyLjY3OTY4OCA1My4wNjc3MDMsMzIuMzc1IDUyLjgwNTk4NCwzMi4xODc1IEw0Ny41MjQ3MzQsMjguMTg3NSBDNDcuMzA5ODksMjguMDI3MzQ0IDQ3LjA0MDM1OSwyNy45NjA5MzggNDYuNzc0NzM0LDI4IEM0Ni43NDM0ODQsMjggNDYuNzEyMjM0LDI4IDQ2LjY4MDk4NCwyOCBDNDYuMjgyNTQ3LDI4LjA3NDIxOSA0NS45NjYxNCwyOC4zODI4MTMgNDUuODg0MTA5LDI4Ljc4MTI1IEM0NS44MDIwNzgsMjkuMTc5Njg4IDQ1Ljk3MDA0NywyOS41ODU5MzggNDYuMzA1OTg0LDI5LjgxMjUgTDQ5LjIxMjIzNCwzMiBMMTkuMjEyMjM0LDMyIEwyMi4xMTg0ODQsMjkuODEyNSBDMjIuNTIwODI4LDI5LjU2NjQwNiAyMi42OTY2MDksMjkuMDcwMzEzIDIyLjUzNjQ1MywyOC42MjUgQzIyLjM4MDIwMywyOC4xNzk2ODggMjEuOTMwOTg0LDI3LjkwNjI1IDIxLjQ2MjIzNCwyNy45Njg3NSBaIE0xNi4yMTIyMzQsNDIgQzE1LjY2MTQ1Myw0MiAxNS4yMTIyMzQsNDIuNDQ5MjE5IDE1LjIxMjIzNCw0MyBDMTUuMjEyMjM0LDQzLjU1MDc4MSAxNS42NjE0NTMsNDQgMTYuMjEyMjM0LDQ0IEMxNi43NjMwMTUsNDQgMTcuMjEyMjM0LDQzLjU1MDc4MSAxNy4yMTIyMzQsNDMgQzE3LjIxMjIzNCw0Mi40NDkyMTkgMTYuNzYzMDE1LDQyIDE2LjIxMjIzNCw0MiBaIE0yMC4yMTIyMzQsNDIgQzE5LjY2MTQ1Myw0MiAxOS4yMTIyMzQsNDIuNDQ5MjE5IDE5LjIxMjIzNCw0MyBDMTkuMjEyMjM0LDQzLjU1MDc4MSAxOS42NjE0NTMsNDQgMjAuMjEyMjM0LDQ0IEMyMC43NjMwMTUsNDQgMjEuMjEyMjM0LDQzLjU1MDc4MSAyMS4yMTIyMzQsNDMgQzIxLjIxMjIzNCw0Mi40NDkyMTkgMjAuNzYzMDE1LDQyIDIwLjIxMjIzNCw0MiBaIE0yNC4yMTIyMzQsNDIgQzIzLjY2MTQ1Myw0MiAyMy4yMTIyMzQsNDIuNDQ5MjE5IDIzLjIxMjIzNCw0MyBDMjMuMjEyMjM0LDQzLjU1MDc4MSAyMy42NjE0NTMsNDQgMjQuMjEyMjM0LDQ0IEMyNC43NjMwMTUsNDQgMjUuMjEyMjM0LDQzLjU1MDc4MSAyNS4yMTIyMzQsNDMgQzI1LjIxMjIzNCw0Mi40NDkyMTkgMjQuNzYzMDE1LDQyIDI0LjIxMjIzNCw0MiBaIE0yOC4yMTIyMzQsNDIgQzI3LjY2MTQ1Myw0MiAyNy4yMTIyMzQsNDIuNDQ5MjE5IDI3LjIxMjIzNCw0MyBDMjcuMjEyMjM0LDQzLjU1MDc4MSAyNy42NjE0NTMsNDQgMjguMjEyMjM0LDQ0IEMyOC43NjMwMTUsNDQgMjkuMjEyMjM0LDQzLjU1MDc4MSAyOS4yMTIyMzQsNDMgQzI5LjIxMjIzNCw0Mi40NDkyMTkgMjguNzYzMDE1LDQyIDI4LjIxMjIzNCw0MiBaIE0zMi4yMTIyMzQsNDIgQzMxLjY2MTQ1Myw0MiAzMS4yMTIyMzQsNDIuNDQ5MjE5IDMxLjIxMjIzNCw0MyBDMzEuMjEyMjM0LDQzLjU1MDc4MSAzMS42NjE0NTMsNDQgMzIuMjEyMjM0LDQ0IEMzMi43NjMwMTUsNDQgMzMuMjEyMjM0LDQzLjU1MDc4MSAzMy4yMTIyMzQsNDMgQzMzLjIxMjIzNCw0Mi40NDkyMTkgMzIuNzYzMDE1LDQyIDMyLjIxMjIzNCw0MiBaIE0zNi4yMTIyMzQsNDIgQzM1LjY2MTQ1Myw0MiAzNS4yMTIyMzQsNDIuNDQ5MjE5IDM1LjIxMjIzNCw0MyBDMzUuMjEyMjM0LDQzLjU1MDc4MSAzNS42NjE0NTMsNDQgMzYuMjEyMjM0LDQ0IEMzNi43NjMwMTUsNDQgMzcuMjEyMjM0LDQzLjU1MDc4MSAzNy4yMTIyMzQsNDMgQzM3LjIxMjIzNCw0Mi40NDkyMTkgMzYuNzYzMDE1LDQyIDM2LjIxMjIzNCw0MiBaIE00MC4yMTIyMzQsNDIgQzM5LjY2MTQ1Myw0MiAzOS4yMTIyMzQsNDIuNDQ5MjE5IDM5LjIxMjIzNCw0MyBDMzkuMjEyMjM0LDQzLjU1MDc4MSAzOS42NjE0NTMsNDQgNDAuMjEyMjM0LDQ0IEM0MC43NjMwMTUsNDQgNDEuMjEyMjM0LDQzLjU1MDc4MSA0MS4yMTIyMzQsNDMgQzQxLjIxMjIzNCw0Mi40NDkyMTkgNDAuNzYzMDE1LDQyIDQwLjIxMjIzNCw0MiBaIE00NC4yMTIyMzQsNDIgQzQzLjY2MTQ1Myw0MiA0My4yMTIyMzQsNDIuNDQ5MjE5IDQzLjIxMjIzNCw0MyBDNDMuMjEyMjM0LDQzLjU1MDc4MSA0My42NjE0NTMsNDQgNDQuMjEyMjM0LDQ0IEM0NC43NjMwMTUsNDQgNDUuMjEyMjM0LDQzLjU1MDc4MSA0NS4yMTIyMzQsNDMgQzQ1LjIxMjIzNCw0Mi40NDkyMTkgNDQuNzYzMDE1LDQyIDQ0LjIxMjIzNCw0MiBaIE00OC4yMTIyMzQsNDIgQzQ3LjY2MTQ1Myw0MiA0Ny4yMTIyMzQsNDIuNDQ5MjE5IDQ3LjIxMjIzNCw0MyBDNDcuMjEyMjM0LDQzLjU1MDc4MSA0Ny42NjE0NTMsNDQgNDguMjEyMjM0LDQ0IEM0OC43NjMwMTUsNDQgNDkuMjEyMjM0LDQzLjU1MDc4MSA0OS4yMTIyMzQsNDMgQzQ5LjIxMjIzNCw0Mi40NDkyMTkgNDguNzYzMDE1LDQyIDQ4LjIxMjIzNCw0MiBaIE01Mi4yMTIyMzQsNDIgQzUxLjY2MTQ1Myw0MiA1MS4yMTIyMzQsNDIuNDQ5MjE5IDUxLjIxMjIzNCw0MyBDNTEuMjEyMjM0LDQzLjU1MDc4MSA1MS42NjE0NTMsNDQgNTIuMjEyMjM0LDQ0IEM1Mi43NjMwMTUsNDQgNTMuMjEyMjM0LDQzLjU1MDc4MSA1My4yMTIyMzQsNDMgQzUzLjIxMjIzNCw0Mi40NDkyMTkgNTIuNzYzMDE1LDQyIDUyLjIxMjIzNCw0MiBaIiBpZD0iaW5zaWRlX3JhbmdlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_between()


<p><strong>Step 2</strong>: Ratios should be between <code>0</code> and <code>1</code>.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">ratio</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">[0, 1]</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">8</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">8<br />
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfaW5fc2V0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF92YWxzX2luX3NldCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMTcyNDE0KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuMTI3OTY5LDQxLjE1MzgzODIgTDMxLjA4MTQ1NjgsNDEuMTUzODM4MiBDMjkuOTUxMDc0OCw0MS4xNTM2NDI5IDI4Ljg4MjcwNTIsNDAuOTI1NjEzNCAyNy45MDc5ODg4LDQwLjUxMzY5NTMgQzI2LjQ0Njc0NDIsMzkuODk2MDEzNiAyNS4xOTg0OSwzOC44NTk5Njg1IDI0LjMxODk4OTQsMzcuNTU3NzA5OSBDMjMuODc5MjM5MSwzNi45MDY3MjcgMjMuNTMxNDgxOCwzNi4xODk5MjMzIDIzLjI5MzY4NjYsMzUuNDI1MjY3NSBDMjMuMjEzMDIxNywzNS4xNjU4OSAyMy4xNDYwMjg5LDM0LjkwMDU1NTQgMjMuMDkxMzQwOSwzNC42MzA3Mjg2IEw0NC4xMjc4NzE0LDM0LjYzMDcyODYgQzQ1LjAyODQ2NiwzNC42MzA2MzA5IDQ1Ljc1ODY0ODgsMzMuOTAwNDQ4MSA0NS43NTg2NDg4LDMyLjk5OTg1MzUgQzQ1Ljc1ODY0ODgsMzIuMDk5MjU4OSA0NS4wMjg0NjYsMzEuMzY5MDc2MSA0NC4xMjc4NzE0LDMxLjM2OTA3NjEgTDIzLjA5MDU1OTYsMzEuMzY5MDc2MSBDMjMuMTk5MDU2NywzMC44MzM3MTk0IDIzLjM1OTcwMjgsMzAuMzE4MDg5NCAyMy41Njc1MTczLDI5LjgyNjQ4MzEgQzI0LjE4NTE5OSwyOC4zNjUyMzg2IDI1LjIyMTI0NDIsMjcuMTE2OTg0NCAyNi41MjM2MDA0LDI2LjIzNzQ4MzggQzI3LjE3NDU4MzMsMjUuNzk3NzMzNCAyNy44OTEzODcsMjUuNDQ5OTc2MiAyOC42NTYwNDI4LDI1LjIxMjI3ODYgQzI5LjQyMDg5MzksMjQuOTc0NDgzMyAzMC4yMzM0OTk0LDI0Ljg0NTk2NjUgMzEuMDgxMzU5MSwyNC44NDU5NjY1IEw0NC4xMjc3NzM3LDI0Ljg0NTk2NjUgQzQ1LjAyODM2ODMsMjQuODQ1OTY2NSA0NS43NTg2NDg4LDI0LjExNTc4MzcgNDUuNzU4NjQ4OCwyMy4yMTUxODkxIEM0NS43NTg2NDg4LDIyLjMxNDU5NDUgNDUuMDI4MzY4MywyMS41ODQ0MTE3IDQ0LjEyNzc3MzcsMjEuNTg0NDExNyBMMzEuMDgxMzU5MSwyMS41ODQ0MTE3IEMyOS41MDk2NjQzLDIxLjU4NDQxMTcgMjguMDAzOTg1OCwyMS45MDM4NDgzIDI2LjYzNzM3MTEsMjIuNDgyMDc2NSBDMjQuNTg2NjY3OCwyMy4zNDk4NTgzIDIyLjg0NjkwNDksMjQuNzk1MDg3MSAyMS42MTYzMjY3LDI2LjYxNjI5NiBDMjAuMzg1NjUwOCwyOC40MzYyMzU0IDE5LjY2NTEzNiwzMC42NDEzMzQ3IDE5LjY2NTgxOTEsMzMuMDAwMDQ4OCBDMTkuNjY1ODE5MSwzNC41NzE3NDM2IDE5Ljk4NTI1NjMsMzYuMDc3NDIyMiAyMC41NjM1ODIyLDM3LjQ0NDAzNjkgQzIxLjQzMTI2NjMsMzkuNDk0NzQwMiAyMi44NzY1OTI3LDQxLjIzNDUwMzEgMjQuNjk3NzA0LDQyLjQ2NTA4MTMgQzI2LjUxNzY0MzQsNDMuNjk1NzU3MiAyOC43MjI3NDI3LDQ0LjQxNTU4ODMgMzEuMDgxNDU2OCw0NC40MTU1ODgzIEw0NC4xMjc4NzE0LDQ0LjQxNTU4ODMgQzQ1LjAyODQ2Niw0NC40MTU1ODgzIDQ1Ljc1ODY0ODgsNDMuNjg1NDA1NSA0NS43NTg2NDg4LDQyLjc4NDgxMDkgQzQ1Ljc1ODY0ODgsNDEuODg0MjE2MyA0NS4wMjg1NjM2LDQxLjE1MzgzODIgNDQuMTI3OTY5LDQxLjE1MzgzODIgWiIgaWQ9InNldF9vZiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_in_set()


<p>Expect that values in <code>category</code> should be in the set of <code>A</code>, <code>B</code>, <code>C</code>.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">category</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">A, B, C</td>
MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">8</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">8<br />
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


The example above demonstrates:

- plaintext briefs with direct messages
- template-based briefs with Markdown formatting
- automatically generated briefs (`brief=True`)

By combining these different brief styles, you can create validation reports that are informative, consistent, and tailored to your specific data quality requirements.


# Best Practices for Using Briefs

Well-crafted briefs can significantly enhance the readability and usefulness of your validation reports. Here are some guidelines to follow:

1.  Be concise: briefs should be short and to the point; they're meant to quickly communicate the purpose of a validation step

2.  Be specific: include relevant details or conditions that make the validation meaningful

3.  Use templates consistently: if you're using template elements like `"{step}"` or `"{col}"`, try to use them consistently across all briefs for a cleaner look

4.  Use auto-generated briefs as a starting point: you can start with `Validate(brief=True)` to see what Pointblank generates automatically, then customize as needed

5.  Add custom briefs for complex validations: custom briefs are especially useful for complex validations where the purpose might not be immediately obvious from the code

Following these best practices will help ensure your validation reports are easy to understand for everyone who needs to review them.


# Conclusion

Briefs help make validation reports more readable and understandable. By using global briefs, step-level briefs, or a combination of both, you can create validation reports that clearly communicate what each validation step is checking.

Whether you want automatically generated descriptions or precisely tailored custom messages, the brief system provides the flexibility to make your data validation work more transparent and easier to interpret for all stakeholders.
