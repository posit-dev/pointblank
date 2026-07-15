# CDISC Submission Conformance

Regulatory submissions require that every dataset in a study package passes CDISC conformance rules. Pointblank provides a built-in conformance engine that checks your SDTM datasets against the full SDTMIG rule catalog with no external dependencies (no subprocess, no CLI, no network). For those preparing a final submission gate, Pointblank can also delegate to the external [CDISC CORE engine](https://github.com/cdisc-org/cdisc-rules-engine).

Two engines are available:

- **Built-in** (primary): Pointblank's own engine runs 426 SDTMIG 3.4 rules against your datasets in-process. Results are immediately renderable in a notebook. No installation beyond Pointblank is required.
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


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="6" class="gt_heading gt_title gt_font_normal">CDISC Conformance <span style="display:inline-block;padding:2px 9px;border-radius:10px;background:#ffebee;color:#FF3300;font-size:0.78em;font-weight:700;letter-spacing:0.04em;">FAIL</span></th>
</tr>
<tr class="gt_heading">
<th colspan="6" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">SDTMIG 3-4 · 410 passed · 4 failed · 12 n/a</th>
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
<td class="gt_row gt_left" style="background-color: #FF3300; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-033</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; color: #c62828; font-weight: bold; padding-top: 2px; padding-bottom: 2px">1</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM domain must contain RFSTDTC and RFENDTC (reference start/end dates).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-035</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; color: #c62828; font-weight: bold; padding-top: 2px; padding-bottom: 2px">1</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM domain must contain treatment arm variables (ARMCD, ARM, ACTARMCD, ACTARM).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-121</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Domain</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; color: #c62828; font-weight: bold; padding-top: 2px; padding-bottom: 2px">1</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS domain must be present in every SDTM submission.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-122</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Domain</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; color: #c62828; font-weight: bold; padding-top: 2px; padding-bottom: 2px">1</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA domain must be present in every SDTM submission.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-001</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Domain</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM (Demographics) domain is required in every SDTM submission.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-002</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">STUDYID must not be null in any domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-003</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must not be null in any SDTM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-004</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">USUBJID must not be null in any SDTM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-005</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">STUDYID must be consistent (same value) across all records in a dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-006</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must be consistent (same value) across all records in a dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-007</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SEX in DM must use values from the CDISC controlled terminology codelist SEX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-008</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RACE in DM must use values from the CDISC controlled terminology codelist RACE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-009</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">COUNTRY in DM must use ISO 3166 alpha-3 country codes (CDISC COUNTRY codelist).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-010</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DMDTC in DM must be in ISO 8601 extended datetime format.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-011</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RFSTDTC in DM must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-012</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RFENDTC in DM must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-013</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DTHDTC in DM must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-014</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DTHFL in DM must use values from the NY codelist (Y or null).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-015</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUBJID must not be null in DM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-016</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AETERM must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-017</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEDECOD must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-018</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESTDTC in AE must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-019</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEENDTC in AE must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-020</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESER in AE must use values from the NY codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-021</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEOUT in AE must use values from the AEOUT codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-022</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBDTC in LB must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-023</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBTEST must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-024</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBTESTCD must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-025</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSTEST must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-026</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSDTC in VS must be in ISO 8601 extended datetime format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-027</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Metadata</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">USUBJID must be present in every subject-level SDTM domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-028</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Metadata</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">STUDYID must be present in every SDTM domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-029</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Metadata</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must be present in every SDTM domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-030</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ETHNIC in DM must use values from the ETHNIC codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-031</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM domain must contain all required Identifier and Topic variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-032</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM domain must contain required demographic variables (SEX, RACE, ETHNIC, COUNTRY).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-034</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM domain must contain SITEID, AGE, and AGEU.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-036</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE domain must contain required Identifier and sequence variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-037</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE domain must contain the AE term and dictionary-derived variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-038</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE domain must contain severity, seriousness, and outcome variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-039</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS domain must contain required Identifier and test variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-040</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS domain must contain result variables (VSORRES, VSORRESU, VSSTRESC, VSSTRESN, VSSTRESU).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-041</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB domain must contain required Identifier and test variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-042</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB domain must contain result variables (LBORRES, LBORRESU, LBSTRESC, LBSTRESN, LBSTRESU).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-043</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG domain must contain required Identifier and test variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-044</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">In any domain, STUDYID must appear before DOMAIN, which must appear before USUBJID.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-045</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">In findings domains (VS, LB, EG), the sequence variable (xSEQ) must appear after USUBJID.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-046</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">In DM, AGE must be a numeric variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-047</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">In VS, VSSTRESN must be a numeric variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-048</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">In LB, LBSTRESN must be a numeric variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-061</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AGEU in DM must use AGEU codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-062</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">BRTHDTC in DM must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-063</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXTRT must not be null in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-064</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXSTDTC in EX must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-065</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXENDTC in EX must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-066</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSFRQ in EX must use FREQ codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-067</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXROUTE in EX must use ROUTE codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-068</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSFRM in EX must use FRM codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-069</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMTRT must not be null in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-070</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMSTDTC in CM must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-071</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMENDTC in CM must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-072</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMROUTE in CM must use ROUTE codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-073</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSTERM must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-074</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSDECOD must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-075</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSSTDTC in DS must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-076</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSSEQ must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-077</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHTERM must not be null in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-078</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHSTDTC in MH must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-079</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHENDTC in MH must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-080</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must not be null in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-081</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SVSTDTC in SV must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-082</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SVENDTC in SV must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-083</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGTEST must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-084</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGTESTCD must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-085</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGDTC in EG must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-086</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESEV in AE must use AESEV codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-087</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AETOXGR in AE must use AETOXGR codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-088</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEBODSYS must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-089</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEACN in AE must use AEACN codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-090</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESEQ must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-091</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSTESTCD must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-092</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSSTRESU in VS must use VSRESU codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-093</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSBLFL in VS must use NY codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-094</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSTRESU in LB must use LBSTRESU codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-095</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBBLFL in LB must use NY codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-096</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBNRIND in LB must use NRIND codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-097</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PETEST must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-098</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PEDTC in PE must conform to ISO 8601 when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-099</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IETEST must not be null in IE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-100</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IECAT in IE must use IECAT codelist when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-101</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPP--</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RDOMAIN must not be null in SUPP-- datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-102</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPP--</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QNAM must not be null in SUPP-- datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-103</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPP--</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QLABEL must not be null in SUPP-- datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-104</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TSPARMCD must not be null in TS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-105</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TSPARM must not be null in TS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-106</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">STUDYID must be consistent (same value) across all records within each domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-107</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must be consistent (same value) across all records within each dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-108</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX domain must contain required identifier and treatment variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-109</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX domain must contain timing variables EXSTDTC and EXENDTC.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-110</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM domain must contain required identifier and treatment variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-111</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM domain must contain timing variables CMSTDTC and CMENDTC.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-112</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS domain must contain required identifier and disposition variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-113</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH domain must contain required identifier and history variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-114</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV domain must contain required identifier and visit variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-115</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS domain must contain required parameter variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-116</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE domain must contain required inclusion/exclusion criteria variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-117</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE domain must contain required physical examination variables.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-118</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSE must be a numeric variable in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-119</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must be a numeric variable in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-120</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGSTRESN must be a numeric variable in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-123</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ARMCD must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-124</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ARM must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-125</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TAETORD must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-126</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EPOCH must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-127</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ETCD must not be null in TE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-128</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ELEMENT must not be null in TE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-129</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must not be null in TV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-130</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISIT must not be null in TV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-131</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IETESTCD must not be null in TI.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-132</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IETEST must not be null in TI.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-133</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IECAT in TI must use IECAT controlled terminology when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-134</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IEINCLCR in TI must use NY controlled terminology when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-135</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEREL in AE must use AEREL controlled terminology when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-136</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESDTH in AE must use NY controlled terminology when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-137</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESHOSP in AE must use NY controlled terminology when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-138</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESLIFE in AE must use NY controlled terminology when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-139</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXSEQ must not be null in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-140</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMSEQ must not be null in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-141</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHSEQ must not be null in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-142</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGSEQ must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-143</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSSEQ must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-144</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSEQ must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-145</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PESEQ must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-146</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IESEQ must not be null in IE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-147</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESEQ must be unique within USUBJID in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-148</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXSEQ must be unique within USUBJID in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-149</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMSEQ must be unique within USUBJID in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-150</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSEQ must be unique within USUBJID in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-151</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Dataset</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSSEQ must be unique within USUBJID in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-152</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESTDY must be numeric type in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-153</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEENDY must be numeric type in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-154</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBDY must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-155</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSDY must be numeric type in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-156</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXSTDY must be numeric type in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-157</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMSTDY must be numeric type in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-158</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA must contain required variables: STUDYID, DOMAIN, ARMCD, ARM, TAETORD, EPOCH, ELEMENT.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-159</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TE must contain required variables: STUDYID, DOMAIN, ETCD, ELEMENT.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-160</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV must contain required variables: STUDYID, DOMAIN, VISITNUM, VISIT.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-161</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI must contain required variables: STUDYID, DOMAIN, IETESTCD, IETEST, IECAT, IEINCLCR.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-162</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must be present in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-163</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must be present in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-164</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must be present in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-165</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must be present in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-166</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPP--</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QVAL must not be null in SUPP-- supplemental qualifier datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-167</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TSSEQ must not be null in TS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-168</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSTNRLO must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-169</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSTNRHI must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-170</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBORNRLO must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-171</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBORNRHI must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-172</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESEQ must be numeric type in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-173</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXSEQ must be numeric type in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-174</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMSEQ must be numeric type in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-175</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHSEQ must be numeric type in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-176</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSSEQ must be numeric type in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-177</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSEQ must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-178</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGSEQ must be numeric type in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-179</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PESEQ must be numeric type in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-180</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSSEQ must be numeric type in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-181</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'DM' in the DM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-182</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'AE' in the AE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-183</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'LB' in the LB dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-184</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'VS' in the VS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-185</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'EX' in the EX dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-186</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'CM' in the CM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-187</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'DS' in the DS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-188</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'MH' in the MH dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-189</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'EG' in the EG dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-190</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'TA' in the TA dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-191</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RFXSTDTC in DM must conform to ISO 8601 format when present (date of first exposure to trial treatment).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-192</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RFXENDTC in DM must conform to ISO 8601 format when present (date of last exposure to trial treatment).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-193</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RFICDTC in DM must conform to ISO 8601 format when present (date of informed consent).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-194</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RFPENDTC in DM must conform to ISO 8601 format when present (date of end of participation).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-195</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AGE must not be negative in the DM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-196</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DMSEQ must not be null in the DM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-197</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESCONG (congenital anomaly flag) must use the NY codelist when present in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-198</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESDISAB (disability flag) must use the NY codelist when present in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-199</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESMIE (important medical event flag) must use the NY codelist when present in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-200</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AECONTRT (concomitant treatment given flag) must use the NY codelist when present in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-201</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHDECOD (dictionary-derived term) must not be null in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-202</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHPRESP (pre-specified) must use the NY codelist when present in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-203</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHOCCUR (occurrence) must use the NY codelist when present in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-204</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXOCCUR (occurrence) must use the NY codelist when present in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-205</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSTOT (total cumulative dose) must be a numeric variable in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-206</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSNO (dose number in a sequence) must be a numeric variable in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-207</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMOCCUR (occurrence) must use the NY codelist when present in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-208</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMDOSE (dose per administration) must be a numeric variable in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-209</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSCAT (category) must not be null in the DS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-210</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSSTDY (study day of disposition event) must be a numeric variable in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-211</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PETESTCD (test short name) must not be null in the PE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-212</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PESTRESN (numeric result in standard units) must be a numeric variable in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-213</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IEORRES (original result) must not be null in the IE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-214</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IEDTC (date/time of IE assessment) must conform to ISO 8601 format when present in IE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-215</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SVSTDY (study day of visit start) must be a numeric variable in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-216</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SVENDY (study day of visit end) must be a numeric variable in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-217</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TSVAL (parameter value) must not be null in the TS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-218</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TSSEQ (sequence number) must be a numeric variable in TS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-219</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBFAST (fasting status) must use the NY codelist when present in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-220</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBCLSIG (clinically significant) must use the NY codelist when present in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-221</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBLLOQ (lower limit of quantification) must be a numeric variable in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-222</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TESTRL (trial element start rule) must not be null in the TE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-223</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TESTRL (trial element start rule) must be present as a variable in the TE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-224</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TVSTRL (trial visit start rule) must not be null in the TV dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-225</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must be a numeric variable in TV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-226</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TAETORD (order of element within arm) must be a numeric variable in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-227</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ETCD (element code) must not be null in the TA dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-228</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'IE' in the IE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-229</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'SV' in the SV dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-230</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'PE' in the PE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-231</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'TS' in the TS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-232</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'TE' in the TE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-233</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG domain must contain required result variables: EGORRES, EGSTRESC, EGSTRESN, EGSTRESU.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-234</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE domain must contain required result variables: PESTRESC, PESTRESN.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-235</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESTDTC (start date/time of adverse event) must be present as a variable in the AE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-236</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSE (dose per administration) must be present as a variable in the EX dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-237</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHSTDTC (start date/time of medical history event) must be present as a variable in the MH dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-238</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'TV' in the TV dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-239</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN column value must equal 'TI' in the TI dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-240</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IEINCLCR (inclusion/exclusion criterion result flag) must be present as a variable in the TI dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-241</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QNAM must not be null in SUPPDM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-242</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QVAL must not be null in SUPPDM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-243</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QNAM must not be null in SUPPAE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-244</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QVAL must not be null in SUPPAE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-245</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPLB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QNAM must not be null in SUPPLB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-246</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RDOMAIN in SUPPDM must equal 'DM'.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-247</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RDOMAIN in SUPPAE must equal 'AE'.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-248</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPLB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RDOMAIN in SUPPLB must equal 'LB'.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-249</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RDOMAIN in SUPPVS must equal 'VS'.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-250</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QNAM must not be null in SUPPVS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-251</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESTDTC in AE must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-255</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXSTDTC in EX must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-258</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHENDTC in MH must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-259</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSSTDTC in DS must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-261</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSTESTCD (test short name) must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-262</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBTESTCD (test short name) must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-263</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGTESTCD (test short name) must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-264</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSTEST (test name) must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-265</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBTEST (test name) must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-266</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSORRES (original result) must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-267</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBORRES (original result) must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-268</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGORRES (original result) must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-269</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PEORRES (original result) must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-270</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must be a numeric variable in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-271</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESEQ (sequence number) must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-272</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMSEQ (sequence number) must not be null in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-273</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXSEQ (sequence number) must not be null in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-274</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHSEQ (sequence number) must not be null in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-275</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSSEQ (sequence number) must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-276</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSEQ (sequence number) must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-277</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGSEQ (sequence number) must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-278</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PESEQ (sequence number) must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-279</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSSEQ (sequence number) must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-280</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IESEQ (sequence number) must not be null in IE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-281</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSTNRLO (lower limit of reference range) must be a numeric variable in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-282</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSTNRHI (upper limit of reference range) must be a numeric variable in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-283</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSSTRESN (numeric result in standard units) must be a numeric variable in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-285</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSNRIND (reference range indicator) must use values from the NRIND controlled terminology codelist when present in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-286</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGNRIND (reference range indicator) must use values from the NRIND controlled terminology codelist when present in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-287</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PENRIND (reference range indicator) must use values from the NRIND controlled terminology codelist when present in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-288</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must be a numeric variable in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-289</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESER (serious event flag) must be present in the AE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-290</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESEQ must be a numeric variable in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-291</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSSEQ must be a numeric variable in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-292</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSEQ must be a numeric variable in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-293</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGSEQ must be a numeric variable in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-294</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMSEQ must be a numeric variable in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-295</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSSEQ must be a numeric variable in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-296</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXSEQ must be a numeric variable in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-297</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SVSEQ must be a numeric variable in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-298</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHSEQ must be a numeric variable in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-299</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QVAL must not be null in SUPPVS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PETEST (physical exam test name) must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-301</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AETERM (verbatim adverse event term) must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-302</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEDECOD (dictionary-derived preferred term) must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-303</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEBODSYS (body system or organ class) must not be null in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-304</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMTRT (verbatim medication name) must not be null in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-305</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMDECOD (standardized medication name) must not be null in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-306</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSTERM (verbatim disposition term) must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-307</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSDECOD (dictionary-derived disposition term) must not be null in DS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-308</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHTERM (verbatim medical history term) must not be null in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-309</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXTRT (treatment name administered) must not be null in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-310</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSE (dose per administration) must not be null in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-311</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSSTRESC (standardized result in character format) must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-312</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSTRESC (standardized result in character format) must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-313</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGSTRESC (standardized result in character format) must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-314</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PESTRESC (standardized result in character format) must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-315</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSSTRESU column must be present in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-316</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSTRESU column must be present in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-317</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGSTRESU column must be present in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-318</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSTRESN must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-319</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGSTRESN must be numeric type in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-320</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSE must be numeric type in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-321</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ARM (arm name) must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-322</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EPOCH (epoch name) must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-323</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ELEMENT (trial element name) must not be null in TA.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-324</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TEENRL (trial element end rule) must not be null in TE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-325</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISIT (visit description) must not be null in TV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-326</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ARM (arm name) must not be null in TV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-327</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ARMCD (arm code) must not be null in TV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-328</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IETEST (inclusion/exclusion criterion name) must not be null in TI.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-329</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IETESTCD (criterion short name) must not be null in TI.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-330</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TSPARMCD (trial summary parameter short name) must not be null in TS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-331</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must not be null in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-332</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISIT must not be null in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-333</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SVSTDTC must conform to ISO 8601 date/time format when present in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-334</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SVENDTC must conform to ISO 8601 date/time format when present in SV.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-335</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must not be null in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-336</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must not be null in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-337</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must not be null in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-338</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must not be null in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-339</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must be numeric type in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-340</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VISITNUM must be numeric type in PE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-341</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QLABEL must not be null in SUPPDM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-342</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QLABEL must not be null in SUPPAE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-343</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPLB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QLABEL must not be null in SUPPLB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-344</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QNAM must not be null in SUPPEG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-345</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QVAL must not be null in SUPPEG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-346</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RDOMAIN must equal 'EG' in the SUPPEG dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-347</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPCM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QNAM must not be null in SUPPCM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-348</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPCM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RDOMAIN must equal 'CM' in the SUPPCM dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-349</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QNAM must not be null in SUPPDS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-350</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RDOMAIN must equal 'DS' in the SUPPDS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-351</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPMH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QNAM must not be null in SUPPMH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-352</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPMH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RDOMAIN must equal 'MH' in the SUPPMH dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-353</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESTDY (AE start study day) must be numeric type in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-354</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEENDY (AE end study day) must be numeric type in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-355</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMSTDY (CM start study day) must be numeric type in CM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-356</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXSTDY (EX start study day) must be numeric type in EX.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-357</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHSTDY (MH start study day) must be numeric type in MH.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-358</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSSTDY (VS study day) must be numeric type in VS.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-359</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSTDY (LB study day) must be numeric type in LB.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-360</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGSTDY (EG study day) must be numeric type in EG.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-361</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSDTC in VS must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-362</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBDTC in LB must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-363</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGDTC in EG must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-364</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PEDTC in PE must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-365</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEDTC in AE must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-366</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">BRTHDTC in DM must conform to ISO 8601 format when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-368</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXENDY in EX must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-369</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMENDY in CM must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-370</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">MHENDY in MH must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-371</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSPEC (specimen type) column must be present in the LB domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-372</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMROUTE (route of administration) column must be present in the CM domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-373</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXROUTE (route of administration) column must be present in the EX domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-374</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSFRM (dose form) column must be present in the EX domain.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-375</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBSPEC must not be null in LB; specimen type is required for each lab result record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-376</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESER must not be null in AE; the serious adverse event flag is required for each AE record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-377</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">VSDY in VS must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-378</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">LBDY in LB must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-379</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EGDY in EG must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-380</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PEDY in PE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-381</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QLABEL must not be null in SUPPVS; the qualifier label is required in all supplemental datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-382</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QLABEL must not be null in SUPPEG; the qualifier label is required in all supplemental datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-383</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPCM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QLABEL must not be null in SUPPCM; the qualifier label is required in all supplemental datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-384</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QLABEL must not be null in SUPPDS; the qualifier label is required in all supplemental datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-385</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPMH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">QLABEL must not be null in SUPPMH; the qualifier label is required in all supplemental datasets.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-386</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IDVAR must not be null in SUPPDM; the linking variable name is required to connect supplemental records to the parent dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-387</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IDVARVAL must not be null in SUPPDM; the linking variable value is required to connect supplemental records to the parent dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-388</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IDVAR must not be null in SUPPAE; the linking variable name is required to connect supplemental records to the parent AE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-389</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IDVARVAL must not be null in SUPPAE; the linking variable value is required to connect supplemental records to the parent AE dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-390</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPLB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IDVAR must not be null in SUPPLB; the linking variable name is required to connect supplemental records to the parent LB dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-391</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPLB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IDVARVAL must not be null in SUPPLB; the linking variable value is required to connect supplemental records to the parent LB dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-392</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IDVAR must not be null in SUPPVS; the linking variable name is required to connect supplemental records to the parent VS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-393</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IDVARVAL must not be null in SUPPVS; the linking variable value is required to connect supplemental records to the parent VS dataset.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-394</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TSPARM must not be null in TS; the parameter name is required for every trial summary record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-395</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IECAT must not be null in TI; the inclusion/exclusion category is required for each criterion record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-396</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ARMCD must not be null in TA; the arm code is required for every trial arm record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-397</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TAETORD must not be null in TA; the planned element order within the arm is required.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-398</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TVSEQ must not be null in TV; the sequence number is required for every trial visit record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-399</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TVSEQ in TV must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-400</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TISEQ must not be null in TI; the sequence number is required for every trial inclusion/exclusion criteria record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-401</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TI</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TISEQ in TI must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-402</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TASEQ must not be null in TA; the sequence number is required for every trial arm record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-403</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TASEQ in TA must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-404</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TESEQ must not be null in TE; the sequence number is required for every trial element record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-405</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TESEQ in TE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-406</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SVSEQ must not be null in SV; the sequence number is required for every subject visit record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-407</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PESEQ in PE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-408</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IESEQ in IE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-409</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">PEENDY in PE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-410</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IESTDY in IE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-411</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IEENDY in IE must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-412</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSENDY in DS must be a numeric type variable.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-413</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSFRM must not be null in EX; the dose form is required for each exposure record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-414</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXROUTE must not be null in EX; the route of administration is required for each exposure record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-415</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">CMROUTE must not be null in CM; the route of administration is required for each concomitant medication record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-416</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEACN must not be null in AE; the action taken with study treatment is required for each adverse event record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-417</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESTDTC must not be null in AE; the adverse event start date/time is required.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-418</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DSSTDTC must not be null in DS; the disposition date/time is required for each disposition record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-419</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXSTDTC must not be null in EX; the exposure start date/time is required for each exposure record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-420</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SV</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SVSTDTC must not be null in SV; the subject visit start date/time is required for each subject visit record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-421</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must equal 'SUPPDM' in the SUPPDM dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-422</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPAE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must equal 'SUPPAE' in the SUPPAE dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-423</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPLB</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must equal 'SUPPLB' in the SUPPLB dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-424</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPVS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must equal 'SUPPVS' in the SUPPVS dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-425</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must equal 'SUPPEG' in the SUPPEG dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-426</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPCM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must equal 'SUPPCM' in the SUPPCM dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-427</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPDS</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must equal 'SUPPDS' in the SUPPDS dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-428</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPMH</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DOMAIN must equal 'SUPPMH' in the SUPPMH dataset; each record must carry the correct supplemental qualifier domain code.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-429</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IDVAR must not be null in SUPPEG; the identifier variable name is required to link each supplemental qualifier record back to its parent EG record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-430</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SUPPEG</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">IDVARVAL must not be null in SUPPEG; the identifier variable value is required to link each supplemental qualifier record back to its parent EG record.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-432</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXROUTE in EX must be a valid ROUTE codelist term when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-433</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSFRM in EX must be a valid FRM codelist term when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-434</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">TA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EPOCH in TA must be a valid EPOCH codelist term when present.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #4CA64C; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#4CA64C</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-435</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EX</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Record</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">EXDOSFRQ in EX must be a valid FREQ codelist term when present (dosing frequency per interval).</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-049</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable labels in the submission must match the labels specified in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-050</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable data types in the submission must match the types declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-051</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Define</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Each variable in the DM dataset must be declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-052</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Define</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Each variable in the AE dataset must be declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-053</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Define</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variables declared Mandatory in Define-XML must not contain null values in DM.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-054</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Define</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variables declared Mandatory in Define-XML must not contain null values in AE.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-055</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Define</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Variable data types in DM must match the types declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-056</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Codelist</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SEX values in DM must be from the codelist declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-057</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Codelist</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">RACE values in DM must be from the codelist declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-058</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Codelist</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">ETHNIC values in DM must be from the codelist declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-059</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Codelist</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AESEV values in AE must be from the codelist declared in Define-XML.</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #AAAAAA; color: transparent; font-size: 0px; padding-top: 2px; padding-bottom: 2px">#AAAAAA</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-060</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DEFINE</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">Codelist</td>
<td class="gt_row gt_center" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">0</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AEOUT values in AE must be from the codelist declared in Define-XML.</td>
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

Call [get_tabular_report()](../../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report) to get the `GT` object directly if you need to embed it in a report pipeline:


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


    /opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/great_tables/_render_checks.py:37: RenderWarning: Rendering table with .cols_width() in Quarto may result in unexpected behavior. This is because Quarto performs custom table processing. Either use all percentage widths, or set .tab_options(quarto_disable_processing=True) to disable Quarto table processing.
      warnings.warn(


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" class="gt_heading gt_title gt_font_normal">Findings Report for CDISC Conformance</th>
</tr>
<tr class="gt_heading">
<th colspan="8" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">SDTMIG 3-4 · 7 failed (407 passed) · 12 n/a</th>
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
<td class="gt_row gt_left" style="background-color: #FF3300; color: #FF3300; white-space: nowrap; font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-007</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SEX in DM must use values from the CDISC controlled terminology codelist SEX.</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">STUDY01-01-003</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">SEX</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">3</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">UNKNOWN</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: #FF3300; white-space: nowrap; font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-010</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DMDTC in DM must be in ISO 8601 extended datetime format.</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">STUDY01-01-002</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">DMDTC</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">2</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">not-a-date</td>
</tr>
<tr>
<td class="gt_row gt_left" style="background-color: #FF3300; color: #FF3300; white-space: nowrap; font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">#FF3300</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">SDTM-195</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">DM</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px">AGE must not be negative in the DM dataset.</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">STUDY01-01-002</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">AGE</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">2</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px; padding-top: 2px; padding-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">-5</td>
</tr>
</tbody>
</table>


The findings table groups its columns into two spanners:

- **Rule**: `Domain` and `Description` identify which rule fired and where.
- **Finding**: `USUBJID` identifies the subject; `Column` and `Value` show exactly what was wrong; `Row` is the 1-based row number in the source dataset for quick lookup.

At most 100 findings per rule are shown in the table. The true total for each rule is always visible in [get_tabular_report()](../../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report). To export findings to a spreadsheet or join them back to your source data, use [findings_df()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.findings_df) instead.


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
| "SDTM-007" | 1            |
| "SDTM-010" | 1            |
| "SDTM-195" | 1            |


The DataFrame is empty (with the same schema) when all rules pass, so it is safe to call unconditionally in a script. Use it to triage failures before running the more time-intensive CDISC CORE engine, or to generate a custom summary report tailored to your team's workflow.


## What the Rule Catalog Covers

The bundled SDTMIG 3.4 catalog contains 426 rules across seven types:

| Type | Description | Examples |
|----|----|----|
| `RECORD_CHECK` | Per-row value checks | Codelist membership, ISO 8601 dates, numeric ranges |
| `VARIABLE_METADATA_CHECK` | Variable presence and column ordering | USUBJID must precede domain-specific variables |
| `DATASET_METADATA_CHECK` | Dataset-level attributes | Required sort key order |
| `DATASET_CONTENTS_CHECK` | Dataset-level value constraints | All rows in a domain must share the same STUDYID |
| `DOMAIN_PRESENCE_CHECK` | Required or prohibited domain presence | DM must be present |
| `DEFINE_ITEM_METADATA_CHECK` | Variable declarations against Define-XML | Activated when `define_xml` is supplied |
| `DEFINE_CODELIST_CHECK` | Codelist values against Define-XML | Activated when `define_xml` is supplied |

Only `RECORD_CHECK` and `DATASET_CONTENTS_CHECK` rules produce row-level findings accessible via [findings_df()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.findings_df) and [get_findings_table()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.get_findings_table). The other types report a violation count in [get_tabular_report()](../../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report) but do not have individual record detail.

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

**Option 1: Standalone executable.** Download the pre-built binary from the [CDISC CORE releases page](https://github.com/cdisc-org/cdisc-rules-engine/releases). Place it somewhere on your `PATH` under the name [core](../../reference/SDTMVariableSpec.md#pointblank.SDTMVariableSpec.core):

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
3.  A [core](../../reference/SDTMVariableSpec.md#pointblank.SDTMVariableSpec.core) or `cdisc-rules-engine` executable on `PATH`.

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


**Overall result.** [all_passed()](../../reference/Validate.all_passed.md#pointblank.Validate.all_passed) returns `True` when no rules reported issues; [is_core](../../reference/ConformanceReport.md#pointblank.ConformanceReport.is_core) confirms the report originated from the CORE engine rather than the built-in engine:


``` python
print("All passed:", report.all_passed())
print("Is CORE report:", report.is_core)
```


    All passed: False
    Is CORE report: True


**Summary dictionary.** [summary()](../../reference/ContractImport.md#pointblank.ContractImport.summary) returns run provenance and rule counts as a plain dictionary, useful for logging, assertions in CI scripts, or building a custom status page:


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


The `status_counts` dictionary enumerates every status bucket CORE reported. Use it to build pass/fail thresholds or trend charts across multiple submission runs.

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

[to_json()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.to_json) saves the report as a JSON file. For CORE reports the structure mirrors CORE's native output and is parseable by `parse_core_report()`:


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
