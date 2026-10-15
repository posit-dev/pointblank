# CDISC Submission Conformance

Regulatory submissions require that every dataset in a study package passes CDISC conformance rules. Pointblank provides a built-in conformance engine that checks your SDTM datasets against the full SDTMIG rule catalog with no external dependencies (no subprocess, no CLI, no network). For those preparing a final submission gate, Pointblank can also delegate to the external [CDISC CORE engine](https://github.com/cdisc-org/cdisc-rules-engine).

Two engines are available:

- **Built-in** (primary): Pointblank's own engine runs 362 SDTMIG 3.4 rules against your datasets in-process. Results are immediately renderable in a notebook. No installation beyond Pointblank is required.
- **CORE** (advanced): delegates to the open-source CDISC CORE CLI, which runs the authoritative CDISC-certified rule set. Requires the CORE executable to be installed separately.


# SDTMIG Conformance

The built-in engine covers the complete SDTMIG 3.4 rule catalog, spanning per-record value checks, variable metadata, dataset-level constraints, domain presence, and Define-XML cross-references. The sections below walk through the three output surfaces (the tabular conformance report, the record-level findings table, and the programmatic findings DataFrame) and describe the rule catalog, controlled terminology handling, and optional inputs such as custom CT packages and Define-XML.


## Quick Start

<a href="../../reference/validate_sdtmig.html#pointblank.validate_sdtmig" class="gdls-link"><code>validate_sdtmig()</code></a> is the entry point for SDTMIG conformance. Pass a dictionary mapping domain names to DataFrames and it returns a <a href="../../reference/ConformanceReport.html#pointblank.ConformanceReport" class="gdls-link"><code>ConformanceReport</code></a> that renders as a color-coded summary table in Jupyter and Quarto notebooks:


``` python
import polars as pl
import pointblank as pb

dm = pl.DataFrame({
    "STUDYID": ["STUDY01"] * 4,
    "DOMAIN":  ["DM"] * 4,
    "USUBJID": ["STUDY01-01-001", "STUDY01-01-002", "STUDY01-01-003", "STUDY01-01-004"],
    "SUBJID":  ["001", "002", "003", "004"],
    "RFSTDTC": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"],
    "SEX":     ["M", "F", "M", "F"],
    "RACE":    ["WHITE", "ASIAN", "WHITE", "BLACK OR AFRICAN AMERICAN"],
    "ETHNIC":  ["NOT HISPANIC OR LATINO"] * 4,
    "ARMCD":   ["TRT", "PBO", "TRT", "PBO"],
    "ARM":     ["Treatment", "Placebo", "Treatment", "Placebo"],
    "COUNTRY": ["USA"] * 4,
    "AGE":     [45, 62, 38, 55],
    "AGEU":    ["YEARS"] * 4,
    "DMDTC":   ["2024-01-10"] * 4,
    "DMDY":    [1] * 4,
    "SITEID":  ["01"] * 4,
})

report = pb.validate_sdtmig({"DM": dm})
report
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_conformance_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_conformance_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_conformance_tbl p { margin: 0; padding: 0; }
 #pb_conformance_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 90%; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_conformance_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_conformance_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_conformance_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_conformance_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_conformance_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_conformance_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_conformance_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_conformance_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_conformance_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_conformance_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_conformance_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_conformance_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_conformance_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_conformance_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_conformance_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_conformance_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_conformance_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_conformance_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_conformance_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_conformance_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_conformance_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_conformance_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_conformance_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_conformance_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_conformance_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_conformance_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_conformance_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_conformance_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_conformance_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_conformance_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_conformance_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_conformance_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_conformance_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_conformance_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_conformance_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_conformance_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_conformance_tbl .gt_left { text-align: left; }
 #pb_conformance_tbl .gt_center { text-align: center; }
 #pb_conformance_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_conformance_tbl .gt_font_normal { font-weight: normal; }
 #pb_conformance_tbl .gt_font_bold { font-weight: bold; }
 #pb_conformance_tbl .gt_font_italic { font-style: italic; }
 #pb_conformance_tbl .gt_super { font-size: 65%; }
 #pb_conformance_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_conformance_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_conformance_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_conformance_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_conformance_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_conformance_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_conformance_tbl-title" aria-describedby="pb_conformance_tbl-subtitle" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="6" id="pb_conformance_tbl-title" class="gt_heading gt_title gt_font_normal">CDISC Conformance <span style="display:inline-block;padding:2px 9px;border-radius:10px;background:#ffebee;color:#FF3300;font-size:0.78em;font-weight:700;letter-spacing:0.04em;">FAIL</span></th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="pb_conformance_tbl-subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">SDTMIG 3-4 · 346 passed · 4 failed · 12 n/a</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_conformance_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_conformance_tbl-rule_id" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Rule</th>
<th id="pb_conformance_tbl-dataset" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Domain</th>
<th id="pb_conformance_tbl-type" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Type</th>
<th id="pb_conformance_tbl-n_issues" class="gt_col_heading gt_columns_bottom_border gt_center" scope="col">Issues</th>
<th id="pb_conformance_tbl-description" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Description</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-033</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; color: #c62828; font-weight: bold; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">1</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DM domain must contain RFSTDTC and RFENDTC (reference start/end dates).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-035</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; color: #c62828; font-weight: bold; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">1</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DM domain must contain treatment arm variables (ARMCD, ARM, ACTARMCD, ACTARM).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-121</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Domain</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; color: #c62828; font-weight: bold; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">1</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TS domain must be present in every SDTM submission.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-122</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Domain</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; color: #c62828; font-weight: bold; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">1</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TA domain must be present in every SDTM submission.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-001</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Domain</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DM (Demographics) domain is required in every SDTM submission.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-002</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">STUDYID must not be null in any domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-003</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must not be null in any SDTM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-004</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">USUBJID must not be null in any SDTM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-005</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">STUDYID must be consistent (same value) across all records in a dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-006</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must be consistent (same value) across all records in a dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-007</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">SEX in DM must use values from the CDISC controlled terminology codelist SEX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-008</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RACE in DM must use values from the CDISC controlled terminology codelist RACE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-009</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">COUNTRY in DM must use ISO 3166 alpha-3 country codes (CDISC COUNTRY codelist).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-010</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DMDTC in DM must be in ISO 8601 extended datetime format.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-011</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RFSTDTC in DM must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-012</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RFENDTC in DM must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-013</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DTHDTC in DM must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-014</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DTHFL in DM must use values from the NY codelist (Y or null).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-015</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">SUBJID must not be null in DM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-016</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AETERM must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-017</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AEDECOD must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-018</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESTDTC in AE must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-019</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AEENDTC in AE must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-020</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESER in AE must use values from the NY codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-021</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AEOUT in AE must use values from the OUT (Outcome of Event) codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-022</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBDTC in LB must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-023</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBTEST must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-024</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBTESTCD must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-025</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSTEST must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-026</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSDTC in VS must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-027</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Metadata</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">USUBJID must be present in every subject-level SDTM domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-028</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Metadata</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">STUDYID must be present in every SDTM domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-029</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Metadata</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must be present in every SDTM domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-030</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">ETHNIC in DM must use values from the ETHNIC codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-031</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DM domain must contain all required Identifier and Topic variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-032</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DM domain must contain required demographic variables (SEX, RACE, COUNTRY).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-034</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DM domain must contain SITEID, AGE, and AGEU.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-036</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AE domain must contain required Identifier and sequence variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-037</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AE domain must contain the AE term and dictionary-derived variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-038</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AE domain must contain the seriousness variable (AESER).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-039</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VS domain must contain required Identifier and test variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-040</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VS domain must contain result variables (VSORRES, VSORRESU, VSSTRESC, VSSTRESN, VSSTRESU).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-041</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LB domain must contain required Identifier and test variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-042</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LB domain must contain result variables (LBORRES, LBORRESU, LBSTRESC, LBSTRESN, LBSTRESU).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-043</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EG domain must contain required Identifier and test variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-044</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">In any domain, STUDYID must appear before DOMAIN, which must appear before USUBJID.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-045</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">In findings domains (VS, LB, EG), the sequence variable (xSEQ) must appear after USUBJID.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-046</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">In DM, AGE must be a numeric variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-047</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">In VS, VSSTRESN must be a numeric variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-048</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">In LB, LBSTRESN must be a numeric variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-061</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AGEU in DM must use AGEU codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-062</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">BRTHDTC in DM must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-063</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXTRT must not be null in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-064</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXSTDTC in EX must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-065</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXENDTC in EX must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-066</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXDOSFRQ in EX must use FREQ codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-067</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXROUTE in EX must use ROUTE codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-068</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXDOSFRM in EX must use FRM codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-069</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMTRT must not be null in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-070</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMSTDTC in CM must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-071</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMENDTC in CM must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-072</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMROUTE in CM must use ROUTE codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-073</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSTERM must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-074</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSDECOD must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-075</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSSTDTC in DS must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-076</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSSEQ must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-077</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHTERM must not be null in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-078</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHSTDTC in MH must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-079</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHENDTC in MH must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-080</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must not be null in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-081</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">SVSTDTC in SV must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-082</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">SVENDTC in SV must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-083</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGTEST must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-084</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGTESTCD must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-085</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGDTC in EG must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-086</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESEV in AE must use AESEV codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-089</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AEACN in AE must use the ACN (Action Taken with Study Treatment) codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-090</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESEQ must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-091</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSTESTCD must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-092</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSSTRESU in VS must use VSRESU codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-093</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSBLFL in VS must use NY codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-094</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSTRESU in LB must use the UNIT codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-095</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBBLFL in LB must use NY codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-096</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBNRIND in LB must use NRIND codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-097</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PETEST must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-098</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PEDTC in PE must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-099</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IETEST must not be null in IE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-100</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IECAT in IE must use IECAT codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-101</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPP--</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RDOMAIN must not be null in SUPP-- datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-102</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPP--</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QNAM must not be null in SUPP-- datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-103</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPP--</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QLABEL must not be null in SUPP-- datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-104</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TSPARMCD must not be null in TS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-105</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TSPARM must not be null in TS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-106</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">STUDYID must be consistent (same value) across all records within each domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-107</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must be consistent (same value) across all records within each dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-108</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EX domain must contain required identifier and treatment variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-109</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EX domain must contain timing variables EXSTDTC and EXENDTC.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-110</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CM domain must contain required identifier and treatment variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-112</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DS domain must contain required identifier and disposition variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-113</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MH domain must contain required identifier and history variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-114</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">SV domain must contain required identifier and visit variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-115</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TS domain must contain required parameter variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-116</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IE domain must contain required inclusion/exclusion criteria variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-117</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PE domain must contain required physical examination variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-118</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXDOSE must be a numeric variable in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-119</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must be a numeric variable in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-120</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGSTRESN must be a numeric variable in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-123</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">ARMCD must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-124</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">ARM must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-125</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TAETORD must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-126</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EPOCH must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-127</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">ETCD must not be null in TE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-128</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">ELEMENT must not be null in TE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-129</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must not be null in TV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-131</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IETESTCD must not be null in TI.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-132</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IETEST must not be null in TI.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-133</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IECAT in TI must use IECAT controlled terminology when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-136</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESDTH in AE must use NY controlled terminology when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-137</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESHOSP in AE must use NY controlled terminology when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-138</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESLIFE in AE must use NY controlled terminology when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-139</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXSEQ must not be null in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-140</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMSEQ must not be null in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-141</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHSEQ must not be null in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-142</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGSEQ must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-143</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSSEQ must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-144</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSEQ must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-145</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PESEQ must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-146</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IESEQ must not be null in IE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-147</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESEQ must be unique within USUBJID in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-148</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXSEQ must be unique within USUBJID in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-149</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMSEQ must be unique within USUBJID in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-150</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSEQ must be unique within USUBJID in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-151</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSSEQ must be unique within USUBJID in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-152</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESTDY must be numeric type in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-153</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AEENDY must be numeric type in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-154</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBDY must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-155</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSDY must be numeric type in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-156</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXSTDY must be numeric type in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-157</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMSTDY must be numeric type in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-158</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TA must contain required variables: STUDYID, DOMAIN, ARMCD, ARM, TAETORD, EPOCH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-159</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TE must contain required variables: STUDYID, DOMAIN, ETCD, ELEMENT.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-160</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TV must contain required variables: STUDYID, DOMAIN, VISITNUM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-161</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TI must contain required variables: STUDYID, DOMAIN, IETESTCD, IETEST, IECAT.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-163</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must be present in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-164</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must be present in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-165</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must be present in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-166</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPP--</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QVAL must not be null in SUPP-- supplemental qualifier datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-167</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TSSEQ must not be null in TS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-168</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSTNRLO must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-169</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSTNRHI must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-172</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESEQ must be numeric type in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-173</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXSEQ must be numeric type in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-174</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMSEQ must be numeric type in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-175</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHSEQ must be numeric type in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-176</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSSEQ must be numeric type in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-177</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSEQ must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-178</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGSEQ must be numeric type in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-179</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PESEQ must be numeric type in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-180</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSSEQ must be numeric type in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-181</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'DM' in the DM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-182</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'AE' in the AE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-183</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'LB' in the LB dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-184</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'VS' in the VS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-185</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'EX' in the EX dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-186</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'CM' in the CM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-187</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'DS' in the DS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-188</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'MH' in the MH dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-189</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'EG' in the EG dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-190</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'TA' in the TA dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-191</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RFXSTDTC in DM must conform to ISO 8601 format when present (date of first exposure to trial treatment).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-192</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RFXENDTC in DM must conform to ISO 8601 format when present (date of last exposure to trial treatment).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-193</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RFICDTC in DM must conform to ISO 8601 format when present (date of informed consent).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-194</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RFPENDTC in DM must conform to ISO 8601 format when present (date of end of participation).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-195</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AGE must not be negative in the DM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-197</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESCONG (congenital anomaly flag) must use the NY codelist when present in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-198</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESDISAB (disability flag) must use the NY codelist when present in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-199</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESMIE (important medical event flag) must use the NY codelist when present in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-200</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AECONTRT (concomitant treatment given flag) must use the NY codelist when present in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-202</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHPRESP (pre-specified) must use the NY codelist when present in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-203</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHOCCUR (occurrence) must use the NY codelist when present in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-204</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXOCCUR (occurrence) must use the NY codelist when present in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-205</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXDOSTOT (total cumulative dose) must be a numeric variable in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-206</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXDOSNO (dose number in a sequence) must be a numeric variable in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-207</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMOCCUR (occurrence) must use the NY codelist when present in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-208</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMDOSE (dose per administration) must be a numeric variable in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-210</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSSTDY (study day of disposition event) must be a numeric variable in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-211</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PETESTCD (test short name) must not be null in the PE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-212</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PESTRESN (numeric result in standard units) must be a numeric variable in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-213</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IEORRES (original result) must not be null in the IE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-214</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IEDTC (date/time of IE assessment) must conform to ISO 8601 format when present in IE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-215</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">SVSTDY (study day of visit start) must be a numeric variable in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-216</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">SVENDY (study day of visit end) must be a numeric variable in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-218</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TSSEQ (sequence number) must be a numeric variable in TS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-219</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBFAST (fasting status) must use the NY codelist when present in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-220</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBCLSIG (clinically significant) must use the NY codelist when present in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-221</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBLLOQ (lower limit of quantification) must be a numeric variable in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-222</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TESTRL (trial element start rule) must not be null in the TE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-223</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TESTRL (trial element start rule) must be present as a variable in the TE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-224</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TVSTRL (trial visit start rule) must not be null in the TV dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-225</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must be a numeric variable in TV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-226</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TAETORD (order of element within arm) must be a numeric variable in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-227</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">ETCD (element code) must not be null in the TA dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-228</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'IE' in the IE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-229</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'SV' in the SV dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-230</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'PE' in the PE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-231</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'TS' in the TS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-232</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'TE' in the TE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-233</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EG domain must contain required result variables: EGORRES, EGSTRESC, EGSTRESN, EGSTRESU.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-234</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PE domain must contain required result variables: PESTRESC, PESTRESN.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-235</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESTDTC (start date/time of adverse event) must be present as a variable in the AE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-236</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXDOSE (dose per administration) must be present as a variable in the EX dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-238</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'TV' in the TV dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-239</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN column value must equal 'TI' in the TI dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-241</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QNAM must not be null in SUPPDM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-242</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QVAL must not be null in SUPPDM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-243</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QNAM must not be null in SUPPAE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-244</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QVAL must not be null in SUPPAE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-245</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPLB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QNAM must not be null in SUPPLB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-246</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RDOMAIN in SUPPDM must equal 'DM'.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-247</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RDOMAIN in SUPPAE must equal 'AE'.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-248</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPLB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RDOMAIN in SUPPLB must equal 'LB'.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-249</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RDOMAIN in SUPPVS must equal 'VS'.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-250</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QNAM must not be null in SUPPVS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-251</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESTDTC in AE must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-255</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXSTDTC in EX must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-258</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHENDTC in MH must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-259</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSSTDTC in DS must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-261</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSTESTCD (test short name) must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-262</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBTESTCD (test short name) must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-263</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGTESTCD (test short name) must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-264</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSTEST (test name) must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-265</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBTEST (test name) must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-268</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGORRES (original result) must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-269</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PEORRES (original result) must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-270</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must be a numeric variable in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-271</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESEQ (sequence number) must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-272</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMSEQ (sequence number) must not be null in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-273</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXSEQ (sequence number) must not be null in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-274</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHSEQ (sequence number) must not be null in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-275</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSSEQ (sequence number) must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-276</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSEQ (sequence number) must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-277</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGSEQ (sequence number) must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-278</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PESEQ (sequence number) must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-279</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSSEQ (sequence number) must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-280</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IESEQ (sequence number) must not be null in IE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-281</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSTNRLO (lower limit of reference range) must be a numeric variable in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-282</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSTNRHI (upper limit of reference range) must be a numeric variable in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-283</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSSTRESN (numeric result in standard units) must be a numeric variable in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-285</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSNRIND (reference range indicator) must use values from the NRIND controlled terminology codelist when present in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-286</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGNRIND (reference range indicator) must use values from the NRIND controlled terminology codelist when present in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-287</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PENRIND (reference range indicator) must use values from the NRIND controlled terminology codelist when present in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-288</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must be a numeric variable in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-289</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESER (serious event flag) must be present in the AE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-290</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESEQ must be a numeric variable in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-291</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSSEQ must be a numeric variable in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-292</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSEQ must be a numeric variable in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-293</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGSEQ must be a numeric variable in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-294</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMSEQ must be a numeric variable in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-295</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSSEQ must be a numeric variable in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-296</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXSEQ must be a numeric variable in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-298</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHSEQ must be a numeric variable in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-299</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QVAL must not be null in SUPPVS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PETEST (physical exam test name) must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-301</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AETERM (verbatim adverse event term) must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-302</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AEDECOD (dictionary-derived preferred term) must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-304</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMTRT (verbatim medication name) must not be null in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-306</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSTERM (verbatim disposition term) must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-307</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSDECOD (dictionary-derived disposition term) must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-308</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHTERM (verbatim medical history term) must not be null in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-309</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXTRT (treatment name administered) must not be null in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-313</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGSTRESC (standardized result in character format) must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-314</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PESTRESC (standardized result in character format) must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-315</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSSTRESU column must be present in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-316</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSTRESU column must be present in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-317</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGSTRESU column must be present in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-318</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSTRESN must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-319</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGSTRESN must be numeric type in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-320</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXDOSE must be numeric type in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-321</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">ARM (arm name) must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-322</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EPOCH (epoch name) must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-328</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IETEST (inclusion/exclusion criterion name) must not be null in TI.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-329</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IETESTCD (criterion short name) must not be null in TI.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-330</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TSPARMCD (trial summary parameter short name) must not be null in TS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-331</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must not be null in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-333</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">SVSTDTC must conform to ISO 8601 date/time format when present in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-334</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">SVENDTC must conform to ISO 8601 date/time format when present in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-337</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-338</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-339</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must be numeric type in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-340</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VISITNUM must be numeric type in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-341</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QLABEL must not be null in SUPPDM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-342</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QLABEL must not be null in SUPPAE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-343</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPLB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QLABEL must not be null in SUPPLB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-344</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QNAM must not be null in SUPPEG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-345</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QVAL must not be null in SUPPEG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-346</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RDOMAIN must equal 'EG' in the SUPPEG dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-347</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPCM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QNAM must not be null in SUPPCM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-348</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPCM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RDOMAIN must equal 'CM' in the SUPPCM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-349</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPDS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QNAM must not be null in SUPPDS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-350</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPDS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RDOMAIN must equal 'DS' in the SUPPDS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-351</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPMH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QNAM must not be null in SUPPMH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-352</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPMH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RDOMAIN must equal 'MH' in the SUPPMH dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-353</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESTDY (AE start study day) must be numeric type in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-354</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AEENDY (AE end study day) must be numeric type in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-355</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMSTDY (CM start study day) must be numeric type in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-356</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXSTDY (EX start study day) must be numeric type in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-357</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHSTDY (MH start study day) must be numeric type in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-358</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSSTDY (VS study day) must be numeric type in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-359</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBSTDY (LB study day) must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-360</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGSTDY (EG study day) must be numeric type in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-361</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSDTC in VS must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-362</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBDTC in LB must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-363</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGDTC in EG must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-364</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PEDTC in PE must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-365</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AEDTC in AE must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-366</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">BRTHDTC in DM must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-368</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXENDY in EX must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-369</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">CMENDY in CM must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-370</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">MHENDY in MH must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-374</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXDOSFRM (dose form) column must be present in the EX domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-377</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">VSDY in VS must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-378</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">LBDY in LB must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-379</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EGDY in EG must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-380</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PEDY in PE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-381</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QLABEL must not be null in SUPPVS; the qualifier label is required in all supplemental datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-382</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QLABEL must not be null in SUPPEG; the qualifier label is required in all supplemental datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-383</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPCM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QLABEL must not be null in SUPPCM; the qualifier label is required in all supplemental datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-384</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPDS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QLABEL must not be null in SUPPDS; the qualifier label is required in all supplemental datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-385</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPMH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">QLABEL must not be null in SUPPMH; the qualifier label is required in all supplemental datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-394</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TSPARM must not be null in TS; the parameter name is required for every trial summary record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-395</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IECAT must not be null in TI; the inclusion/exclusion category is required for each criterion record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-396</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">ARMCD must not be null in TA; the arm code is required for every trial arm record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-397</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">TAETORD must not be null in TA; the planned element order within the arm is required.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-407</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PESEQ in PE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-408</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IESEQ in IE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-409</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">PEENDY in PE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-410</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IESTDY in IE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-411</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">IEENDY in IE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-412</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DSENDY in DS must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-421</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must equal 'SUPPDM' in the SUPPDM dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-422</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must equal 'SUPPAE' in the SUPPAE dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-423</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPLB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must equal 'SUPPLB' in the SUPPLB dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-424</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must equal 'SUPPVS' in the SUPPVS dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-425</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must equal 'SUPPEG' in the SUPPEG dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-426</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPCM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must equal 'SUPPCM' in the SUPPCM dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-427</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPDS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must equal 'SUPPDS' in the SUPPDS dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-428</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">SUPPMH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">DOMAIN must equal 'SUPPMH' in the SUPPMH dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-432</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXROUTE in EX must be a valid ROUTE codelist term when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-433</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EXDOSFRM in EX must be a valid FRM codelist term when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-434</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">EPOCH in TA must be a valid EPOCH codelist term when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-049</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">Variable labels in the submission must match the labels specified in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-050</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">Variable data types in the submission must match the types declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-051</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Define</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">Each variable in the DM dataset must be declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-052</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Define</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">Each variable in the AE dataset must be declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-053</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Define</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">Variables declared Mandatory in Define-XML must not contain null values in DM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-054</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Define</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">Variables declared Mandatory in Define-XML must not contain null values in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-055</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Define</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">Variable data types in DM must match the types declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-056</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Codelist</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">SEX values in DM must be from the codelist declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-057</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Codelist</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">RACE values in DM must be from the codelist declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-058</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Codelist</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">ETHNIC values in DM must be from the codelist declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-059</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Codelist</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AESEV values in AE must be from the codelist declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-status_color">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-rule_id">SDTM-060</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-dataset">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-type">Codelist</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-n_issues">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="pb_conformance_tbl-description">AEOUT values in AE must be from the codelist declared in Define-XML.</td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="6" class="gt_sourcenote"><span style="font-size:0.82em;color:#9e9e9e;">CT: sdtm-ct-2024-09-27</span></td>
</tr>
</tfoot>

</table>


Domain keys are matched case-insensitively (`"dm"` and `"DM"` are equivalent). Polars, pandas, and any other narwhals-compatible DataFrame are accepted. The report object itself renders the tabular summary in a notebook; read on to learn how to drill into failures and access findings programmatically.


## The Conformance Report

The tabular report shown above is produced by <a href="../../reference/ConformanceReport.html#pointblank.ConformanceReport.get_tabular_report" class="gdls-link"><code>get_tabular_report()</code></a>. Each row is one rule from the SDTMIG 3.4 catalog; the colored bar on the left edge encodes its status:

| Color | Status | Meaning |
|----|----|----|
| Green | `pass` | The rule was evaluated and no violations were found |
| Red | `fail` | One or more records or datasets violated the rule |
| Amber | [error](../../reference/Validate.error.md#pointblank.Validate.error) | The rule could not be evaluated (unexpected data state) |
| Grey | `not_applicable` | The rule requires a dataset or variable that was not supplied |
| Grey | `not_supported` | The rule type is not yet implemented in the built-in engine |

Rules are sorted by severity: failures first, then errors, then passes, then not-applicable.

The header shows a `PASS` or `FAIL` badge alongside the standard, version, and a full status breakdown (e.g., `SDTMIG 3-4 · 410 passed · 4 failed · 12 n/a`).

Call `get_tabular_report()` to get the `GT` object directly if you need to embed it in a report pipeline:


``` python
gt = report.get_tabular_report()
gt
```


The `GT` object can be passed to any Great Tables export method (for example, `gt.as_raw_html()` to embed the table in a custom HTML report, or `gt.gtsave("report.png")` to render it as an image). When failures are present, use the findings surfaces below to investigate the specific records that triggered each rule.


## Findings Drill-Down

When rules fail, the tabular report shows how many records violated each rule but not which ones. <a href="../../reference/ConformanceReport.html#pointblank.ConformanceReport.get_findings_table" class="gdls-link"><code>get_findings_table()</code></a> provides that drill-down: one row per failing record, showing the subject, the specific column that violated the rule, the offending value, and the 1-based row number in the source dataset.

To see it in action, introduce a few deliberate violations:


``` python
dm_with_issues = pl.DataFrame({
    "STUDYID": ["STUDY01"] * 4,
    "DOMAIN":  ["DM"] * 4,
    "USUBJID": ["STUDY01-01-001", "STUDY01-01-002", "STUDY01-01-003", "STUDY01-01-004"],
    "SUBJID":  ["001", "002", "003", "004"],
    "RFSTDTC": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"],
    "SEX":     ["M", "F", "UNKNOWN", "F"],      # "UNKNOWN" is not in the SEX codelist
    "RACE":    ["WHITE", "ASIAN", "WHITE", "BLACK OR AFRICAN AMERICAN"],
    "ETHNIC":  ["NOT HISPANIC OR LATINO"] * 4,
    "ARMCD":   ["TRT", "PBO", "TRT", "PBO"],
    "ARM":     ["Treatment", "Placebo", "Treatment", "Placebo"],
    "COUNTRY": ["USA"] * 4,
    "AGE":     [45, -5, 38, 55],               # -5 violates AGE >= 0
    "AGEU":    ["YEARS"] * 4,
    "DMDTC":   ["2024-01-10", "not-a-date", "2024-01-10", "2024-01-10"],  # invalid ISO date
    "DMDY":    [1] * 4,
    "SITEID":  ["01"] * 4,
})

report_with_issues = pb.validate_sdtmig({"DM": dm_with_issues})
report_with_issues.get_findings_table()
```


    /opt/hostedtoolcache/Python/3.11.17/x64/lib/python3.11/site-packages/great_tables/_render_checks.py:37: RenderWarning: Rendering table with .cols_width() in Quarto may result in unexpected behavior. This is because Quarto performs custom table processing. Either use all percentage widths, or set .tab_options(quarto_disable_processing=True) to disable Quarto table processing.
      warnings.warn(


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#vxrnbgbvix table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#vxrnbgbvix thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#vxrnbgbvix p { margin: 0; padding: 0; }
 #vxrnbgbvix .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 90%; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #vxrnbgbvix .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #vxrnbgbvix .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #vxrnbgbvix .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #vxrnbgbvix .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #vxrnbgbvix .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #vxrnbgbvix .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #vxrnbgbvix .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #vxrnbgbvix .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #vxrnbgbvix .gt_column_spanner_outer:first-child { padding-left: 0; }
 #vxrnbgbvix .gt_column_spanner_outer:last-child { padding-right: 0; }
 #vxrnbgbvix .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #vxrnbgbvix .gt_spanner_row { border-bottom-style: hidden; }
 #vxrnbgbvix .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #vxrnbgbvix .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #vxrnbgbvix .gt_from_md> :first-child { margin-top: 0; }
 #vxrnbgbvix .gt_from_md> :last-child { margin-bottom: 0; }
 #vxrnbgbvix .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #vxrnbgbvix .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #vxrnbgbvix .gt_indent_1 { text-indent: 5px; }
 #vxrnbgbvix .gt_indent_2 { text-indent: calc(5px * 2); }
 #vxrnbgbvix .gt_indent_3 { text-indent: calc(5px * 3); }
 #vxrnbgbvix .gt_indent_4 { text-indent: calc(5px * 4); }
 #vxrnbgbvix .gt_indent_5 { text-indent: calc(5px * 5); }
 #vxrnbgbvix .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #vxrnbgbvix .gt_row_group_first td { border-top-width: 2px; }
 #vxrnbgbvix .gt_row_group_first th { border-top-width: 2px; }
 #vxrnbgbvix .gt_striped { color: #333333; background-color: #F4F4F4; }
 #vxrnbgbvix .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #vxrnbgbvix .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #vxrnbgbvix .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #vxrnbgbvix .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #vxrnbgbvix .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #vxrnbgbvix .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #vxrnbgbvix .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #vxrnbgbvix .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #vxrnbgbvix .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #vxrnbgbvix .gt_left { text-align: left; }
 #vxrnbgbvix .gt_center { text-align: center; }
 #vxrnbgbvix .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #vxrnbgbvix .gt_font_normal { font-weight: normal; }
 #vxrnbgbvix .gt_font_bold { font-weight: bold; }
 #vxrnbgbvix .gt_font_italic { font-style: italic; }
 #vxrnbgbvix .gt_super { font-size: 65%; }
 #vxrnbgbvix .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #vxrnbgbvix .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #vxrnbgbvix .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #vxrnbgbvix .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #vxrnbgbvix .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #vxrnbgbvix .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" id="title" class="gt_heading gt_title gt_font_normal">Findings Report for CDISC Conformance</th>
</tr>
<tr class="gt_heading">
<th colspan="8" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">SDTMIG 3-4 · 7 failed (343 passed) · 12 n/a</th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th rowspan="2" id="status_color" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th colspan="3" id="SDTM-Rule-Definition" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">SDTM Rule Definition</th>
<th colspan="4" id="Finding" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">Finding</th>
</tr>
<tr class="gt_col_headings">
<th id="rule_id" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Rule</th>
<th id="dataset" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Domain</th>
<th id="description" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Description</th>
<th id="usubjid" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">USUBJID</th>
<th id="checked_column" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Column</th>
<th id="row_1indexed" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">Row</th>
<th id="checked_value" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Value</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: #FF3300; white-space: nowrap; font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="status_color">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="SDTM-Rule-Definition rule_id">SDTM-007</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="SDTM-Rule-Definition dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="SDTM-Rule-Definition description">SEX in DM must use values from the CDISC controlled terminology codelist SEX.</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding usubjid">STUDY01-01-003</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding checked_column">SEX</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding row_1indexed">3</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding checked_value">UNKNOWN</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: #FF3300; white-space: nowrap; font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="status_color">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="SDTM-Rule-Definition rule_id">SDTM-010</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="SDTM-Rule-Definition dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="SDTM-Rule-Definition description">DMDTC in DM must be in ISO 8601 extended datetime format.</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding usubjid">STUDY01-01-002</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding checked_column">DMDTC</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding row_1indexed">2</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding checked_value">not-a-date</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: #FF3300; white-space: nowrap; font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="status_color">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="SDTM-Rule-Definition rule_id">SDTM-195</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="SDTM-Rule-Definition dataset">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px" headers="SDTM-Rule-Definition description">AGE must not be negative in the DM dataset.</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding usubjid">STUDY01-01-002</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding checked_column">AGE</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding row_1indexed">2</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap" headers="Finding checked_value">-5</td>
</tr>
</tbody>
</table>


The findings table groups its columns into two spanners:

- **Rule**: `Domain` and `Description` identify which rule fired and where.
- **Finding**: `USUBJID` identifies the subject; `Column` and `Value` show exactly what was wrong; `Row` is the 1-based row number in the source dataset for quick lookup.

At most 100 findings per rule are shown in the table. The true total for each rule is always visible in `get_tabular_report()`. To export findings to a spreadsheet or join them back to your source data, use [findings_df()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.findings_df) instead.


## Programmatic Access with findings_df()

<a href="../../reference/ConformanceReport.html#pointblank.ConformanceReport.findings_df" class="gdls-link"><code>findings_df()</code></a> returns the same findings as a Polars DataFrame, which is better suited for filtering, grouping, exporting to CSV, or joining back to the source data:


``` python
df = report_with_issues.findings_df()
df
```


shape: (3, 7)

| rule_id | dataset | row_index | usubjid | checked_column | checked_value | description |
|----|----|----|----|----|----|----|
| str | str | i64 | str | str | str | str |
| "SDTM-007" | "DM" | 2 | "STUDY01-01-003" | "SEX" | "UNKNOWN" | "SEX in DM must use values from… |
| "SDTM-010" | "DM" | 1 | "STUDY01-01-002" | "DMDTC" | "not-a-date" | "DMDTC in DM must be in ISO 860… |
| "SDTM-195" | "DM" | 1 | "STUDY01-01-002" | "AGE" | "-5" | "AGE must not be negative in th… |


The DataFrame schema is:

| Column | Description |
|----|----|
| `rule_id` | CDISC CORE rule identifier (e.g., `"SDTM-007"`) |
| `dataset` | SDTM domain the failing record belongs to |
| `row_index` | 0-based row position in the source dataset |
| `usubjid` | Unique Subject Identifier (`USUBJID`) for the failing record |
| `checked_column` | The variable that violated the rule (e.g., `"SEX"`) |
| `checked_value` | The actual value found in `checked_column` |
| `description` | Human-readable rule description |


``` python
import polars as pl

# Filter to a specific rule
df.filter(pl.col("rule_id") == "SDTM-007")
```


shape: (1, 7)

| rule_id | dataset | row_index | usubjid | checked_column | checked_value | description |
|----|----|----|----|----|----|----|
| str | str | i64 | str | str | str | str |
| "SDTM-007" | "DM" | 2 | "STUDY01-01-003" | "SEX" | "UNKNOWN" | "SEX in DM must use values from… |


``` python
# Group by rule to count violations
df.group_by("rule_id").agg(pl.len().alias("n_violations")).sort("n_violations", descending=True)
```


shape: (3, 2)

| rule_id    | n_violations |
|------------|--------------|
| str        | u32          |
| "SDTM-195" | 1            |
| "SDTM-007" | 1            |
| "SDTM-010" | 1            |


The DataFrame is empty (with the same schema) when all rules pass, so it is safe to call unconditionally in a script. Use it to triage failures before running the more time-intensive CDISC CORE engine, or to generate a custom summary report tailored to your team's workflow.


## What the Rule Catalog Covers

The bundled SDTMIG 3.4 catalog contains 362 rules across seven types:

| Type | Description | Examples |
|----|----|----|
| `RECORD_CHECK` | Per-row value checks | Codelist membership, ISO 8601 dates, numeric ranges |
| `VARIABLE_METADATA_CHECK` | Variable presence and column ordering | USUBJID must precede domain-specific variables |
| `DATASET_METADATA_CHECK` | Dataset-level attributes | Required sort key order |
| `DATASET_CONTENTS_CHECK` | Dataset-level value constraints | All rows in a domain must share the same STUDYID |
| `DOMAIN_PRESENCE_CHECK` | Required or prohibited domain presence | DM must be present |
| `DEFINE_ITEM_METADATA_CHECK` | Variable declarations against Define-XML | Activated when `define_xml` is supplied |
| `DEFINE_CODELIST_CHECK` | Codelist values against Define-XML | Activated when `define_xml` is supplied |

Only `RECORD_CHECK` and `DATASET_CONTENTS_CHECK` rules produce row-level findings accessible via [findings_df()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.findings_df) and [get_findings_table()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.get_findings_table). The other types report a violation count in `get_tabular_report()` but do not have individual record detail.

Rules that require a domain or variable not present in your datasets are automatically marked `not_applicable` (they are not counted as failures). For example, a rule that checks `AESTDTC` in the AE domain is `not_applicable` when no AE dataset is supplied. Adding more domains to the dictionary passed to [validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig) will convert more rules from `not_applicable` to executable, giving a more complete conformance picture.


## Controlled Terminology

Codelist checks use the bundled CT package `sdtm-ct-2024-09-27`. Two important behaviors:

- **Case-insensitive matching**: a value of `"beats/min"` matches the codelist term `"BEATS/MIN"`. This avoids false positives for studies that applied mixed-case CT values.
- **SAS/XPT empty strings treated as null**: SAS Transport files encode character missing values as `""` (empty string) rather than `None`. The engine recognizes this and skips codelist and format checks for such cells, preventing the large volumes of false positives that occur with a naive string comparison.

Both behaviors apply automatically with no configuration. If you need to pin a different CT version or supply additional packages, see the next section.


## Supply a Custom CT Package

By default the most recent bundled CT package is used. Pass `ct_packages` to pin a specific version or supply additional packages:


``` python
report = pb.validate_sdtmig(
    {"DM": dm},
    ct_packages=["sdtm-ct-2024-09-27"],
)
```


Pinning the CT version is useful when your study was locked against a specific CDISC CT release and you want the conformance check to reflect that snapshot rather than the latest terms.


## Activating Define-XML Rules

Pass a path to `define.xml` to activate the Define-XML-aware rule types:


``` python
report = pb.validate_sdtmig(
    {"DM": dm, "AE": ae},
    define_xml="path/to/define.xml",
)
```


Without `define_xml`, the 76 `DEFINE_ITEM_METADATA_CHECK` and `DEFINE_CODELIST_CHECK` rules are marked `not_applicable`. Supplying the file allows the engine to verify that every variable present in your datasets is declared in Define-XML with the correct type and, where applicable, a valid codelist.

Define-XML also tells the CT checks about sponsor extensions. Many CDISC codelists are extensible (EPOCH, FREQ, ROUTE, and UNIT among them): a sponsor may add terms, provided Define-XML declares them with `def:ExtendedValue="Yes"` on the codelist item. When `define_xml` is supplied, such declared terms pass the codelist checks for extensible codelists. For example, a crossover study can use the epochs `TREATMENT 1` and `TREATMENT 2`. Without Define-XML, or for a codelist that isn't extensible (such as SEX), terms outside the codelist are still reported.


## Loading XPT Files

For datasets stored as SAS Transport (XPT) files, use `pyreadstat` to load them before passing to [validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig). Install it with `pip install pyreadstat`:


``` python
import pyreadstat
import polars as pl
import pointblank as pb

def load_xpt(path: str) -> pl.DataFrame:
    df, _ = pyreadstat.read_xport(path)
    return pl.from_pandas(df)

report = pb.validate_sdtmig(
    {
        "DM": load_xpt("sdtm/dm.xpt"),
        "AE": load_xpt("sdtm/ae.xpt"),
        "LB": load_xpt("sdtm/lb.xpt"),
        "VS": load_xpt("sdtm/vs.xpt"),
    },
    study_id="STUDY01",
)
```


Pointblank handles the SAS empty-string convention automatically, so XPT data does not require any preprocessing before being passed to [validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig). The same built-in handling applies whether data arrives from XPT files, in-memory DataFrames, or any other narwhals-compatible source.


# Prerequisites

[validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig) and the [ConformanceReport](../../reference/ConformanceReport.md#pointblank.ConformanceReport) methods require no additional dependencies beyond Pointblank itself. The optional extras are:

- **Reading XPT files** with `pyreadstat`: `pip install pyreadstat`
- **Writing XPT files** for CORE (automatic, handled internally): `pip install pyreadstat`
- **Excel export** with [to_excel()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.to_excel): `pip install openpyxl`

``` bash
pip install pointblank[cdisc]    # adds pyreadstat
pip install pointblank[excel]    # adds openpyxl
```

The CDISC CORE engine is not a Python dependency of Pointblank and must be installed separately when needed. See [Installing the CDISC CORE Engine](#installing-the-cdisc-core-engine) below.


# CDISC CORE Engine (Advanced)

The CDISC CORE engine runs the full authoritative conformance rule set and produces reports accepted by FDA and PMDA review tools. Use it as the final pre-submission gate after the built-in engine gives a clean result.


## Installing the CDISC CORE Engine

CORE can be obtained in three ways depending on your environment. The standalone executable is easiest for local use, Docker is preferred in containerized CI pipelines, and the repo checkout is useful when you need a specific unreleased version or want to inspect the rule definitions directly.

**Option 1: Standalone executable.** Download the pre-built binary from the [CDISC CORE releases page](https://github.com/cdisc-org/cdisc-rules-engine/releases). Place it somewhere on your `PATH` under the name `core`:

``` bash
chmod +x core
sudo mv core /usr/local/bin/
core --version
```

Once the binary is on `PATH`, Pointblank will discover it automatically with no additional configuration required.

**Option 2: Docker.** CDISC publishes an official Docker image that includes the engine and its full rules cache:

``` bash
docker pull cdisc/cdisc-rules-engine:latest

docker run --rm \
    -v /path/to/study/data:/data \
    cdisc/cdisc-rules-engine:latest \
    validate -s sdtmig -v 3-4 -d /data -of JSON -o /data/report
```

When running via Docker you will typically invoke CORE directly rather than through Pointblank's subprocess wrapper. The resulting JSON output can be loaded with `parse_core_report()` and wrapped in a [ConformanceReport](../../reference/ConformanceReport.md#pointblank.ConformanceReport) for further analysis.

**Option 3: Repo checkout.**

``` bash
git clone https://github.com/cdisc-org/cdisc-rules-engine.git
cd cdisc-rules-engine
pip install -r requirements.txt
python core.py --version
```

When using a repo checkout, pass the repo root as `core_cwd` so Pointblank sets the subprocess working directory correctly.


## Telling Pointblank Where to Find CORE

Pointblank discovers CORE through three mechanisms, tried in order:

1.  An explicit `core=` argument to [validate_cdisc_submission()](../../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) or `validate_conformance(engine="core")`.
2.  The `POINTBLANK_CDISC_CORE` environment variable (e.g., `export POINTBLANK_CDISC_CORE="python /path/to/core.py"`).
3.  A `core` or `cdisc-rules-engine` executable on `PATH`.

The environment variable approach is convenient for CI systems where the CORE path differs between machines, while the explicit argument is useful in notebooks where you want the path to be self-documenting.


## Running CORE

[validate_cdisc_submission()](../../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) is the simplest way to run a CORE validation. It accepts an in-memory dictionary of DataFrames, a folder path, or an existing [SubmissionPackage](../../reference/SubmissionPackage.md#pointblank.SubmissionPackage):


``` python
report = pb.validate_cdisc_submission(
    {"DM": dm, "AE": ae},
    standard="sdtmig",
    version="3.4",
    agency="FDA",
)
```


For a folder of XPT files pass the path directly. Pointblank skips the materialization step and passes the folder straight to CORE:


``` python
report = pb.validate_cdisc_submission(
    "path/to/sdtm/",
    standard="sdtmig",
    version="3.4",
    agency="FDA",
)
```


For repo-checkout CORE, pass the command prefix and working directory:


``` python
report = pb.validate_cdisc_submission(
    {"DM": dm},
    standard="sdtmig",
    version="3.4",
    core=["python", "/path/to/cdisc-rules-engine/core.py"],
    core_cwd="/path/to/cdisc-rules-engine",
    cache="/path/to/cdisc-rules-engine/resources/cache",
)
```


In all cases the return value is a [ConformanceReport](../../reference/ConformanceReport.md#pointblank.ConformanceReport) with `is_core=True`. The sections below describe the accessor methods available on a CORE report.


## Working with a CORE ConformanceReport

The examples below use a captured real CORE report so that the code runs without requiring CORE to be installed in the docs environment. The structure is identical to what a live run produces.


``` python
import json
from pathlib import Path
from pointblank.metadata import parse_core_report, ConformanceReport

_fixtures = Path(pb.__file__).parent.parent / "tests" / "metadata_fixtures" / "cdisc_core"
raw = json.loads((_fixtures / "core_report_full.json").read_text())

report = ConformanceReport.from_core_report(raw, agency="FDA")
print(report)
```


    ConformanceReport (CORE)
      Agency: FDA
      SDTMIG V3.4 -- CORE 0.16.0
      430 rules (EXECUTION ERROR=2, ISSUE REPORTED=6, SKIPPED=344, SUCCESS=78)
      8 issues -- FAIL


**Overall result.** `all_passed()` returns `True` when no rules reported issues; [is_core](../../reference/ConformanceReport.md#pointblank.ConformanceReport.is_core) confirms the report originated from the CORE engine rather than the built-in engine:


``` python
print("All passed:", report.all_passed())
print("Is CORE report:", report.is_core)
```


    All passed: False
    Is CORE report: True


**Summary dictionary.** `summary()` returns run provenance and rule counts as a plain dictionary, useful for logging, assertions in CI scripts, or building a custom status page:


``` python
s = report.summary()

print(f"Standard:        {s['standard']} {s['version']}")
print(f"Engine:          {s['engine_version']}")
print(f"Rules evaluated: {s['n_rules']}")
print(f"Total issues:    {s['n_issues']}")
print()

print("Rule status breakdown:")
for status, count in sorted(s["status_counts"].items()):
    print(f"  {status:20s}: {count}")
```


    Standard:        SDTMIG V3.4
    Engine:          0.16.0
    Rules evaluated: 430
    Total issues:    8

    Rule status breakdown:
      EXECUTION ERROR     : 2
      ISSUE REPORTED      : 6
      SKIPPED             : 344
      SUCCESS             : 78


The [status_counts](../../reference/Comparison.status_counts.md#pointblank.Comparison.status_counts) dictionary enumerates every status bucket CORE reported. Use it to build pass/fail thresholds or trend charts across multiple submission runs.

**Issues.** [issues()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.issues) returns one record per (dataset, rule) pair that reported at least one issue. Pass a `status` filter to separate conformance violations from execution errors:


``` python
issues = report.issues()
print(f"Issue entries: {len(issues)}")

for issue in issues[:3]:
    print()
    print(f"  Dataset:  {issue['dataset']}")
    print(f"  Rule:     {issue['rule_id']}")
    print(f"  Message:  {issue['message']}")
    print(f"  Issues:   {issue['issues']}")
```


    Issue entries: 8

      Dataset:  STUDY
      Rule:     CORE-000581
      Message:  DM dataset is missing.
      Issues:   1

      Dataset:  TEST_DATASET
      Rule:     CORE-000357
      Message:  Supplemental qualifier dataset associated with a split dataset is greater than 8 characters in length
      Issues:   1

      Dataset:  TEST_DATASET
      Rule:     CORE-000510
      Message:  Split dataset name is not 3 or 4 characters in length
      Issues:   1


``` python
from pointblank.metadata._cdisc_core import STATUS_ISSUE, STATUS_ERROR

reported_issues = report.issues(status=STATUS_ISSUE)
print(f"Rules with issues: {len(reported_issues)}")

exec_errors = report.issues(status=STATUS_ERROR)
print(f"Execution errors:  {len(exec_errors)}")
```


    Rules with issues: 6
    Execution errors:  2


Execution errors (`STATUS_ERROR`) indicate that CORE could not evaluate a rule, typically because a required variable or dataset was absent or had an unexpected format. They are distinct from conformance issues (`STATUS_ISSUE`) where the rule ran successfully but found a violation.

**Row-level findings.** [findings()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.findings) returns the row-level detail from CORE's `Issue_Details` section, giving the exact record that triggered each rule:


``` python
findings = report.findings()
print(f"Row-level findings: {len(findings)}")

f = findings[0]
print()
print(f"Rule:          {f.rule_id}")
print(f"Dataset:       {f.dataset}")
print(f"Row:           {f.row}")
print(f"USUBJID:       {f.usubjid}")
print(f"Variables:     {f.variables}")
print(f"Values:        {f.values}")
```


    Row-level findings: 8

    Rule:          CORE-000357
    Dataset:       TEST_DATASET
    Row:           1
    USUBJID:       None
    Variables:     ['dataset_name']
    Values:        ['TEST_DATASET']


Each finding carries the `USUBJID`, the affected variables and their values, and the 1-based row number in the source dataset, matching the information shown in the built-in engine's findings table.

**Per-rule run results.** [rules()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.rules) returns one `CoreRuleResult` per rule with its run status:


``` python
from pointblank.metadata._cdisc_core import STATUS_SUCCESS, STATUS_SKIPPED

all_rules = report.rules()
print(f"Total rules:  {len(all_rules)}")
print(f"Successful:   {len(report.rules(status=STATUS_SUCCESS))}")
print(f"Skipped:      {len(report.rules(status=STATUS_SKIPPED))}")

failing = [r for r in all_rules if r.is_failing]
print(f"Failing:      {len(failing)}")
for r in failing:
    print(f"  {r.rule_id:15s} [{r.status}]  {(r.message or '')[:60]}")
```


    Total rules:  430
    Successful:   78
    Skipped:      344
    Failing:      8
      CORE-000357     [ISSUE REPORTED]  Supplemental qualifier dataset associated with a split datas
      CORE-000510     [ISSUE REPORTED]  Split dataset name is not 3 or 4 characters in length
      CORE-000539     [ISSUE REPORTED]  Split dataset is present but the two-Letter parent domain is
      CORE-000581     [ISSUE REPORTED]  DM dataset is missing.
      CORE-000598     [ISSUE REPORTED]  Dataset name does not begin with DOMAIN value
      CORE-000778     [ISSUE REPORTED]  Associated Persons non-supplemental qualifier dataset associ
      CORE-000929     [EXECUTION ERROR]  DOMAIN Code is not a published DOMAIN Code in CDISC Controll
      CORE-001081     [EXECUTION ERROR]  Variable role specified in the define-xml does not match the


Rules marked skipped are not failures. CORE skips rules when the dataset or variable they check is absent from the submission. A skipped rule is equivalent to `not_applicable` in the built-in engine and should not be treated as a conformance problem.


# Exporting Reports

Both built-in and CORE reports can be exported for archiving, sharing with a biometrics team, or loading into a review tool. Two formats are supported: JSON for round-trippable machine-readable output, and Excel for spreadsheet-based review workflows.


## JSON

`to_json()` saves the report as a JSON file. For CORE reports the structure mirrors CORE's native output and is parseable by `parse_core_report()`:


``` python
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    dest = report.to_json(Path(tmp) / "conformance_report.json")
    print(f"Written to: {dest.name}")
    print(f"File size:  {dest.stat().st_size:,} bytes")

    reloaded = json.loads(dest.read_text())
    reparsed = parse_core_report(reloaded)
    print(f"Rules after round-trip: {len(reparsed.rules)}")
```


    Written to: conformance_report.json
    File size:  113,079 bytes
    Rules after round-trip: 430


The round-trip fidelity means a JSON file written by Pointblank can be re-loaded later without re-running CORE, which is useful for archiving the state of a submission at a specific point in time.


## Excel

[to_excel()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.to_excel) writes the report as a workbook. Requires `openpyxl`:


``` python
dest = report.to_excel("conformance_report.xlsx")
```


For CORE reports the workbook has four sheets: `Issue_Summary`, `Issue_Details`, `Rules_Report`, and `Conformance_Details`.


# Using Both Engines Together

The built-in engine provides fast feedback during development; CORE is the final gate. The typical workflow is to iterate with the built-in engine until it reports a clean result, then run CORE once as a pre-submission check:


``` python
# Step 1: fast iteration with the built-in engine
report = pb.validate_sdtmig({"DM": dm, "AE": ae}, study_id="STUDY01")

if not report.all_passed():
    print("Fix rule violations before running CORE:")
    df = report.findings_df()
    print(df.group_by("rule_id").agg(pl.len().alias("n")).sort("n", descending=True))
    raise SystemExit(1)

# Step 2: final gate with CDISC CORE
core_report = pb.validate_cdisc_submission(
    {"DM": dm, "AE": ae},
    standard="sdtmig",
    version="3.4",
    agency="FDA",
)

if not core_report.all_passed():
    failing = [r for r in core_report.rules() if r.is_failing]
    print(f"CORE found {len(failing)} failing rules.")
    core_report.to_json("core_report.json")
    raise SystemExit(1)

print("Submission passed all conformance checks.")
```


This pattern catches the majority of conformance problems early (when fixing them is cheap) and reserves the slower CORE invocation for the final verification step. Both reports can be exported and archived alongside the submission package.


# Running the Integration Tests

Pointblank ships integration tests for the full CORE pipeline. They are skipped automatically when CORE is not discoverable:

``` bash
# Run when CORE is on PATH:
pytest -m cdisc_core

# With a repo-checkout CORE:
export POINTBLANK_CDISC_CORE="python /path/to/core.py"
export POINTBLANK_CDISC_CORE_CWD="/path/to/cdisc-rules-engine"
pytest -m cdisc_core
```

The integration tests validate the full subprocess pipeline end-to-end, including XPT materialization, CORE invocation, JSON parsing, and report construction. Running them against your local CORE installation is a good sanity check after upgrading either Pointblank or the CORE engine.
