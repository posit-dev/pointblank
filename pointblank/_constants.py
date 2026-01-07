GENERAL_COLUMN_TYPES = [
    "numeric",  # Numeric data types (`int`, `float`)
    "str",  # String data type (`string`)
    "bool",  # Boolean data type (`boolean`)
    "datetime",  # Date or Datetime data type (`datetime`)
    "duration",  # Duration data type (`duration`)
]

COMPATIBLE_DTYPES = {
    "gt": ["numeric", "datetime"],
    "lt": ["numeric", "datetime"],
    "eq": ["numeric", "datetime", "str"],
    "ne": ["numeric", "datetime", "str"],
    "ge": ["numeric", "datetime"],
    "le": ["numeric", "datetime"],
    "between": ["numeric", "datetime"],
    "outside": ["numeric", "datetime"],
    "in_set": ["numeric", "str"],
    "not_in_set": ["numeric", "str"],
    "regex": ["str"],
    "within_spec": ["str"],
    "null": ["str", "numeric", "bool", "datetime", "duration"],
    "not_null": ["str", "numeric", "bool", "datetime", "duration"],
}

ASSERTION_TYPE_METHOD_MAP: dict[str, str] = {
    "col_pct_null": "pct_null",
    "col_vals_gt": "gt",
    "col_vals_lt": "lt",
    "col_vals_eq": "eq",
    "col_vals_ne": "ne",
    "col_vals_ge": "ge",
    "col_vals_le": "le",
    "col_vals_between": "between",
    "col_vals_outside": "outside",
    "col_vals_in_set": "in_set",
    "col_vals_not_in_set": "not_in_set",
    "col_vals_regex": "regex",
    "col_vals_within_spec": "within_spec",
    "col_vals_increasing": "increasing",
    "col_vals_decreasing": "decreasing",
    "col_vals_null": "null",
    "col_vals_not_null": "not_null",
    "col_vals_expr": "expr",
    "col_exists": "col_exists",
    "rows_distinct": "rows_distinct",
    "rows_complete": "rows_complete",
    "prompt": "prompt",
    "col_schema_match": "col_schema_match",
    "row_count_match": "row_count_match",
    "col_count_match": "col_count_match",
    "tbl_match": "tbl_match",
    "conjointly": "conjointly",
    "specially": "specially",
    "col_sum_eq": "sum_eq",
}

COMPARISON_OPERATORS = {
    "col_vals_gt": ">",
    "col_vals_ge": ">=",
    "col_vals_lt": "<",
    "col_vals_le": "<=",
    "col_vals_eq": "==",
    "col_vals_ne": "!=",
}

COMPARISON_OPERATORS_AR = {
    "col_vals_gt": "أكبر من",
    "col_vals_ge": "أكبر من أو يساوي",
    "col_vals_lt": "أصغر من",
    "col_vals_le": "أصغر من أو يساوي",
    "col_vals_eq": "يساوي",
    "col_vals_ne": "لا يساوي",
}

ROW_BASED_VALIDATION_TYPES = [
    "col_vals_gt",
    "col_vals_lt",
    "col_vals_eq",
    "col_vals_ne",
    "col_vals_ge",
    "col_vals_le",
    "col_vals_between",
    "col_vals_outside",
    "col_vals_in_set",
    "col_vals_not_in_set",
    "col_vals_regex",
    "col_vals_within_spec",
    "col_vals_increasing",
    "col_vals_decreasing",
    "col_vals_null",
    "col_vals_not_null",
    "col_vals_expr",
    "conjointly",
    "prompt",
]

IBIS_BACKENDS = [
    "bigquery",
    "databricks",
    "duckdb",
    "memtable",
    "mssql",
    "mysql",
    "parquet",
    "postgres",
    "snowflake",
    "sqlite",
]

VALIDATION_REPORT_FIELDS = [
    "i",
    "i_o",
    "assertion_type",
    "column",
    "values",
    "inclusive",
    "na_pass",
    "pre",
    "segments",
    "thresholds",
    "label",
    "brief",
    "active",
    "all_passed",
    "n",
    "n_passed",
    "n_failed",
    "f_passed",
    "f_failed",
    "warning",
    "error",
    "critical",
    "time_processed",
    "proc_duration_s",
]

MODEL_PROVIDERS = [
    "openai",
    "anthropic",
    "ollama",
    "bedrock",
]

TABLE_TYPE_STYLES = {
    "pandas": {"background": "#150458", "text": "#FFFFFF", "label": "Pandas"},
    "polars": {"background": "#0075FF", "text": "#FFFFFF", "label": "Polars"},
    "polars-lazy": {"background": "#0075FF", "text": "#FFFFFF", "label": "Polars (LazyFrame)"},
    "narwhals": {"background": "#78BEAF", "text": "#222222", "label": "Narwhals"},
    "narwhals-lazy": {"background": "#78BEAF", "text": "#222222", "label": "Narwhals (LazyFrame)"},
    "duckdb": {"background": "#000000", "text": "#FFFFFF", "label": "DuckDB"},
    "mysql": {"background": "#EBAD40", "text": "#222222", "label": "MySQL"},
    "postgres": {"background": "#3E638B", "text": "#FFFFFF", "label": "PostgreSQL"},
    "sqlite": {"background": "#BACBEF", "text": "#222222", "label": "SQLite"},
    "parquet": {"background": "#3F9FF9", "text": "#FFFFFF", "label": "Parquet"},
    "memtable": {"background": "#2C3E50", "text": "#FFFFFF", "label": "Ibis memtable"},
    "mssql": {"background": "#E2E2E2", "text": "#222222", "label": "MSSQL"},
    "bigquery": {"background": "#4285F4", "text": "#FFFFFF", "label": "BigQuery"},
    "pyspark": {"background": "#E66F21", "text": "#FFFFFF", "label": "Spark DataFrame"},
    "databricks": {"background": "#FF3621", "text": "#FFFFFF", "label": "Databricks"},
}

REPORTING_LANGUAGES = [
    "en",
    "fr",
    "de",
    "it",
    "es",
    "pt",
    "ro",
    "tr",
    "zh-Hans",
    "zh-Hant",
    "ja",
    "ko",
    "vi",
    "ru",
    "cs",
    "pl",
    "da",
    "sv",
    "nb",
    "nl",
    "fi",
    "is",
    "ar",
    "hi",
    "el",
    "id",
    "uk",
    "bg",
    "hr",
    "et",
    "hu",
    "ga",
    "lv",
    "lt",
    "mt",
    "sk",
    "sl",
    "he",
    "th",
    "fa",
]

RTL_LANGUAGES = [
    "ar",
    "he",
    "fa",
]

SEVERITY_LEVEL_COLORS = {
    "warning": "#AAAAAA",
    "error": "#EBBC14",
    "critical": "#FF3300",
    "green": "#4CA64C",
}

LOG_LEVELS_MAP = {"warning": 30, "error": 40, "critical": 50}

CHECK_MARK_SPAN = "<span style='color: #4CA64C;'>&check;</span>"
CROSS_MARK_SPAN = "<span style='color: #CF142B;'>&cross;</span>"

SVG_ICONS_FOR_ASSERTION_TYPES: dict[str, str] = {
    ## EQ Icons
    "col_vals_eq": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_eq</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_eq" transform="translate(0.000000, 0.275862)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M52.712234,11 L14.712234,11 C13.05989,11 11.712234,12.347656 11.712234,14 L11.712234,52 C11.712234,53.652344 13.05989,55 14.712234,55 L52.712234,55 C54.364578,55 55.712234,53.652344 55.712234,52 L55.712234,14 C55.712234,12.347656 54.364578,11 52.712234,11 Z M46.712234,38 L20.712234,38 L20.712234,36 L46.712234,36 L46.712234,38 Z M46.712234,30 L20.712234,30 L20.712234,28 L46.712234,28 L46.712234,30 Z" id="equals" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sum_eq": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_gt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_gt" transform="translate(0.000000, 0.724138)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.7099609,12 L17.7099609,12 C14.9499609,12 12.7099609,14.24 12.7099609,17 L12.7099609,49 C12.7099609,51.76 14.9499609,54 17.7099609,54 L49.7099609,54 C52.4699609,54 54.7099609,51.76 54.7099609,49 L54.7099609,17 C54.7099609,14.24 52.4699609,12 49.7099609,12 Z M27.2799609,44.82 L26.1399609,43.18 L40.9499609,33 L26.1399609,22.82 L27.2799609,21.18 L44.4799609,33 L27.2799609,44.82 Z" id="greater_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sd_eq": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_gt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_gt" transform="translate(0.000000, 0.724138)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.7099609,12 L17.7099609,12 C14.9499609,12 12.7099609,14.24 12.7099609,17 L12.7099609,49 C12.7099609,51.76 14.9499609,54 17.7099609,54 L49.7099609,54 C52.4699609,54 54.7099609,51.76 54.7099609,49 L54.7099609,17 C54.7099609,14.24 52.4699609,12 49.7099609,12 Z M27.2799609,44.82 L26.1399609,43.18 L40.9499609,33 L26.1399609,22.82 L27.2799609,21.18 L44.4799609,33 L27.2799609,44.82 Z" id="greater_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_avg_eq": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_gt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_gt" transform="translate(0.000000, 0.724138)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.7099609,12 L17.7099609,12 C14.9499609,12 12.7099609,14.24 12.7099609,17 L12.7099609,49 C12.7099609,51.76 14.9499609,54 17.7099609,54 L49.7099609,54 C52.4699609,54 54.7099609,51.76 54.7099609,49 L54.7099609,17 C54.7099609,14.24 52.4699609,12 49.7099609,12 Z M27.2799609,44.82 L26.1399609,43.18 L40.9499609,33 L26.1399609,22.82 L27.2799609,21.18 L44.4799609,33 L27.2799609,44.82 Z" id="greater_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    ## GT Icons
    "col_vals_gt": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_gt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_gt" transform="translate(0.000000, 0.724138)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.7099609,12 L17.7099609,12 C14.9499609,12 12.7099609,14.24 12.7099609,17 L12.7099609,49 C12.7099609,51.76 14.9499609,54 17.7099609,54 L49.7099609,54 C52.4699609,54 54.7099609,51.76 54.7099609,49 L54.7099609,17 C54.7099609,14.24 52.4699609,12 49.7099609,12 Z M27.2799609,44.82 L26.1399609,43.18 L40.9499609,33 L26.1399609,22.82 L27.2799609,21.18 L44.4799609,33 L27.2799609,44.82 Z" id="greater_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sum_gt": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_gt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_gt" transform="translate(0.000000, 0.724138)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.7099609,12 L17.7099609,12 C14.9499609,12 12.7099609,14.24 12.7099609,17 L12.7099609,49 C12.7099609,51.76 14.9499609,54 17.7099609,54 L49.7099609,54 C52.4699609,54 54.7099609,51.76 54.7099609,49 L54.7099609,17 C54.7099609,14.24 52.4699609,12 49.7099609,12 Z M27.2799609,44.82 L26.1399609,43.18 L40.9499609,33 L26.1399609,22.82 L27.2799609,21.18 L44.4799609,33 L27.2799609,44.82 Z" id="greater_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sd_gt": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_gt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_gt" transform="translate(0.000000, 0.724138)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.7099609,12 L17.7099609,12 C14.9499609,12 12.7099609,14.24 12.7099609,17 L12.7099609,49 C12.7099609,51.76 14.9499609,54 17.7099609,54 L49.7099609,54 C52.4699609,54 54.7099609,51.76 54.7099609,49 L54.7099609,17 C54.7099609,14.24 52.4699609,12 49.7099609,12 Z M27.2799609,44.82 L26.1399609,43.18 L40.9499609,33 L26.1399609,22.82 L27.2799609,21.18 L44.4799609,33 L27.2799609,44.82 Z" id="greater_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_avg_gt": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_gt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_gt" transform="translate(0.000000, 0.724138)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.7099609,12 L17.7099609,12 C14.9499609,12 12.7099609,14.24 12.7099609,17 L12.7099609,49 C12.7099609,51.76 14.9499609,54 17.7099609,54 L49.7099609,54 C52.4699609,54 54.7099609,51.76 54.7099609,49 L54.7099609,17 C54.7099609,14.24 52.4699609,12 49.7099609,12 Z M27.2799609,44.82 L26.1399609,43.18 L40.9499609,33 L26.1399609,22.82 L27.2799609,21.18 L44.4799609,33 L27.2799609,44.82 Z" id="greater_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    ## LT Icons
    "col_vals_lt": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_lt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_lt" transform="translate(0.000000, 0.310345)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M52.712234,11 L14.712234,11 C13.05989,11 11.712234,12.347656 11.712234,14 L11.712234,52 C11.712234,53.652344 13.05989,55 14.712234,55 L52.712234,55 C54.364578,55 55.712234,53.652344 55.712234,52 L55.712234,14 C55.712234,12.347656 54.364578,11 52.712234,11 Z M40.419265,46.292969 L39.005203,47.707031 L24.298172,33 L39.005203,18.292969 L40.419265,19.707031 L27.126297,33 L40.419265,46.292969 Z" id="less_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sum_lt": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_lt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_lt" transform="translate(0.000000, 0.310345)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M52.712234,11 L14.712234,11 C13.05989,11 11.712234,12.347656 11.712234,14 L11.712234,52 C11.712234,53.652344 13.05989,55 14.712234,55 L52.712234,55 C54.364578,55 55.712234,53.652344 55.712234,52 L55.712234,14 C55.712234,12.347656 54.364578,11 52.712234,11 Z M40.419265,46.292969 L39.005203,47.707031 L24.298172,33 L39.005203,18.292969 L40.419265,19.707031 L27.126297,33 L40.419265,46.292969 Z" id="less_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sd_lt": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_lt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_lt" transform="translate(0.000000, 0.310345)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M52.712234,11 L14.712234,11 C13.05989,11 11.712234,12.347656 11.712234,14 L11.712234,52 C11.712234,53.652344 13.05989,55 14.712234,55 L52.712234,55 C54.364578,55 55.712234,53.652344 55.712234,52 L55.712234,14 C55.712234,12.347656 54.364578,11 52.712234,11 Z M40.419265,46.292969 L39.005203,47.707031 L24.298172,33 L39.005203,18.292969 L40.419265,19.707031 L27.126297,33 L40.419265,46.292969 Z" id="less_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_avg_lt": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_lt</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_lt" transform="translate(0.000000, 0.310345)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M52.712234,11 L14.712234,11 C13.05989,11 11.712234,12.347656 11.712234,14 L11.712234,52 C11.712234,53.652344 13.05989,55 14.712234,55 L52.712234,55 C54.364578,55 55.712234,53.652344 55.712234,52 L55.712234,14 C55.712234,12.347656 54.364578,11 52.712234,11 Z M40.419265,46.292969 L39.005203,47.707031 L24.298172,33 L39.005203,18.292969 L40.419265,19.707031 L27.126297,33 L40.419265,46.292969 Z" id="less_than" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    ## NE Icons
    "col_vals_ne": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_ne</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_ne" transform="translate(0.000000, 0.758621)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M53.712234,12 L13.712234,12 C13.161453,12 12.712234,12.449219 12.712234,13 L12.712234,53 C12.712234,53.550781 13.161453,54 13.712234,54 L53.712234,54 C54.263015,54 54.712234,53.550781 54.712234,53 L54.712234,13 C54.712234,12.449219 54.263015,12 53.712234,12 Z M46.712234,30 L32.989578,30 L36.805984,36 L46.712234,36 L46.712234,38 L38.079422,38 L43.169265,46 L40.798172,46 L35.708328,38 L20.712234,38 L20.712234,36 L34.43489,36 L30.618484,30 L20.712234,30 L20.712234,28 L29.345047,28 L24.251297,20 L26.62239,20 L31.71614,28 L46.712234,28 L46.712234,30 Z" id="not_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sum_ne": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_ne</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_ne" transform="translate(0.000000, 0.758621)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M53.712234,12 L13.712234,12 C13.161453,12 12.712234,12.449219 12.712234,13 L12.712234,53 C12.712234,53.550781 13.161453,54 13.712234,54 L53.712234,54 C54.263015,54 54.712234,53.550781 54.712234,53 L54.712234,13 C54.712234,12.449219 54.263015,12 53.712234,12 Z M46.712234,30 L32.989578,30 L36.805984,36 L46.712234,36 L46.712234,38 L38.079422,38 L43.169265,46 L40.798172,46 L35.708328,38 L20.712234,38 L20.712234,36 L34.43489,36 L30.618484,30 L20.712234,30 L20.712234,28 L29.345047,28 L24.251297,20 L26.62239,20 L31.71614,28 L46.712234,28 L46.712234,30 Z" id="not_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sd_ne": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_ne</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_ne" transform="translate(0.000000, 0.758621)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M53.712234,12 L13.712234,12 C13.161453,12 12.712234,12.449219 12.712234,13 L12.712234,53 C12.712234,53.550781 13.161453,54 13.712234,54 L53.712234,54 C54.263015,54 54.712234,53.550781 54.712234,53 L54.712234,13 C54.712234,12.449219 54.263015,12 53.712234,12 Z M46.712234,30 L32.989578,30 L36.805984,36 L46.712234,36 L46.712234,38 L38.079422,38 L43.169265,46 L40.798172,46 L35.708328,38 L20.712234,38 L20.712234,36 L34.43489,36 L30.618484,30 L20.712234,30 L20.712234,28 L29.345047,28 L24.251297,20 L26.62239,20 L31.71614,28 L46.712234,28 L46.712234,30 Z" id="not_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_avg_ne": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_ne</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_ne" transform="translate(0.000000, 0.758621)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M53.712234,12 L13.712234,12 C13.161453,12 12.712234,12.449219 12.712234,13 L12.712234,53 C12.712234,53.550781 13.161453,54 13.712234,54 L53.712234,54 C54.263015,54 54.712234,53.550781 54.712234,53 L54.712234,13 C54.712234,12.449219 54.263015,12 53.712234,12 Z M46.712234,30 L32.989578,30 L36.805984,36 L46.712234,36 L46.712234,38 L38.079422,38 L43.169265,46 L40.798172,46 L35.708328,38 L20.712234,38 L20.712234,36 L34.43489,36 L30.618484,30 L20.712234,30 L20.712234,28 L29.345047,28 L24.251297,20 L26.62239,20 L31.71614,28 L46.712234,28 L46.712234,30 Z" id="not_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    ## GE Icons
    "col_vals_ge": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_ge</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_ge" transform="translate(0.000000, 0.241379)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.712234,12 L17.712234,12 C14.952234,12 12.712234,14.24 12.712234,17 L12.712234,49 C12.712234,51.76 14.952234,54 17.712234,54 L49.712234,54 C52.472234,54 54.712234,51.76 54.712234,49 L54.712234,17 C54.712234,14.24 52.472234,12 49.712234,12 Z M44.712234,47 L22.712234,47 L22.712234,45 L44.712234,45 L44.712234,47 Z M24.182234,40.88 L23.242234,39.12 L40.562234,30 L23.242234,20.88 L24.182234,19.12 L44.862234,30 L24.182234,40.88 Z" id="greater_than_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sum_ge": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_ge</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_ge" transform="translate(0.000000, 0.241379)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.712234,12 L17.712234,12 C14.952234,12 12.712234,14.24 12.712234,17 L12.712234,49 C12.712234,51.76 14.952234,54 17.712234,54 L49.712234,54 C52.472234,54 54.712234,51.76 54.712234,49 L54.712234,17 C54.712234,14.24 52.472234,12 49.712234,12 Z M44.712234,47 L22.712234,47 L22.712234,45 L44.712234,45 L44.712234,47 Z M24.182234,40.88 L23.242234,39.12 L40.562234,30 L23.242234,20.88 L24.182234,19.12 L44.862234,30 L24.182234,40.88 Z" id="greater_than_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sd_ge": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_ge</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_ge" transform="translate(0.000000, 0.241379)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.712234,12 L17.712234,12 C14.952234,12 12.712234,14.24 12.712234,17 L12.712234,49 C12.712234,51.76 14.952234,54 17.712234,54 L49.712234,54 C52.472234,54 54.712234,51.76 54.712234,49 L54.712234,17 C54.712234,14.24 52.472234,12 49.712234,12 Z M44.712234,47 L22.712234,47 L22.712234,45 L44.712234,45 L44.712234,47 Z M24.182234,40.88 L23.242234,39.12 L40.562234,30 L23.242234,20.88 L24.182234,19.12 L44.862234,30 L24.182234,40.88 Z" id="greater_than_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_avg_ge": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_ge</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_ge" transform="translate(0.000000, 0.241379)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M49.712234,12 L17.712234,12 C14.952234,12 12.712234,14.24 12.712234,17 L12.712234,49 C12.712234,51.76 14.952234,54 17.712234,54 L49.712234,54 C52.472234,54 54.712234,51.76 54.712234,49 L54.712234,17 C54.712234,14.24 52.472234,12 49.712234,12 Z M44.712234,47 L22.712234,47 L22.712234,45 L44.712234,45 L44.712234,47 Z M24.182234,40.88 L23.242234,39.12 L40.562234,30 L23.242234,20.88 L24.182234,19.12 L44.862234,30 L24.182234,40.88 Z" id="greater_than_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    ## LE Icons
    "col_vals_le": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_le</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_le" transform="translate(0.000000, 0.793103)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M53.712234,12 L13.712234,12 C13.157547,12 12.712234,12.449219 12.712234,13 L12.712234,53 C12.712234,53.550781 13.157547,54 13.712234,54 L53.712234,54 C54.266922,54 54.712234,53.550781 54.712234,53 L54.712234,13 C54.712234,12.449219 54.266922,12 53.712234,12 Z M42.227859,19.125 L43.196609,20.875 L26.770828,30 L43.196609,39.125 L42.227859,40.875 L22.65364,30 L42.227859,19.125 Z M44.712234,47 L22.712234,47 L22.712234,45 L44.712234,45 L44.712234,47 Z" id="less_than_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sum_le": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_le</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_le" transform="translate(0.000000, 0.793103)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M53.712234,12 L13.712234,12 C13.157547,12 12.712234,12.449219 12.712234,13 L12.712234,53 C12.712234,53.550781 13.157547,54 13.712234,54 L53.712234,54 C54.266922,54 54.712234,53.550781 54.712234,53 L54.712234,13 C54.712234,12.449219 54.266922,12 53.712234,12 Z M42.227859,19.125 L43.196609,20.875 L26.770828,30 L43.196609,39.125 L42.227859,40.875 L22.65364,30 L42.227859,19.125 Z M44.712234,47 L22.712234,47 L22.712234,45 L44.712234,45 L44.712234,47 Z" id="less_than_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_sd_le": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_le</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_le" transform="translate(0.000000, 0.793103)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M53.712234,12 L13.712234,12 C13.157547,12 12.712234,12.449219 12.712234,13 L12.712234,53 C12.712234,53.550781 13.157547,54 13.712234,54 L53.712234,54 C54.266922,54 54.712234,53.550781 54.712234,53 L54.712234,13 C54.712234,12.449219 54.266922,12 53.712234,12 Z M42.227859,19.125 L43.196609,20.875 L26.770828,30 L43.196609,39.125 L42.227859,40.875 L22.65364,30 L42.227859,19.125 Z M44.712234,47 L22.712234,47 L22.712234,45 L44.712234,45 L44.712234,47 Z" id="less_than_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_avg_le": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_le</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_le" transform="translate(0.000000, 0.793103)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M53.712234,12 L13.712234,12 C13.157547,12 12.712234,12.449219 12.712234,13 L12.712234,53 C12.712234,53.550781 13.157547,54 13.712234,54 L53.712234,54 C54.266922,54 54.712234,53.550781 54.712234,53 L54.712234,13 C54.712234,12.449219 54.266922,12 53.712234,12 Z M42.227859,19.125 L43.196609,20.875 L26.770828,30 L43.196609,39.125 L42.227859,40.875 L22.65364,30 L42.227859,19.125 Z M44.712234,47 L22.712234,47 L22.712234,45 L44.712234,45 L44.712234,47 Z" id="less_than_equal" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_vals_between": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_between</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_between" transform="translate(0.000000, 0.206897)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M11.993484,21.96875 C10.962234,22.082031 10.188797,22.964844 10.212234,24 L10.212234,42 C10.200515,42.722656 10.579422,43.390625 11.204422,43.753906 C11.825515,44.121094 12.598953,44.121094 13.220047,43.753906 C13.845047,43.390625 14.223953,42.722656 14.212234,42 L14.212234,24 C14.220047,23.457031 14.009109,22.9375 13.626297,22.554688 C13.243484,22.171875 12.723953,21.960938 12.180984,21.96875 C12.118484,21.964844 12.055984,21.964844 11.993484,21.96875 Z M55.993484,21.96875 C54.962234,22.082031 54.188797,22.964844 54.212234,24 L54.212234,42 C54.200515,42.722656 54.579422,43.390625 55.204422,43.753906 C55.825515,44.121094 56.598953,44.121094 57.220047,43.753906 C57.845047,43.390625 58.223953,42.722656 58.212234,42 L58.212234,24 C58.220047,23.457031 58.009109,22.9375 57.626297,22.554688 C57.243484,22.171875 56.723953,21.960938 56.180984,21.96875 C56.118484,21.964844 56.055984,21.964844 55.993484,21.96875 Z M16.212234,22 C15.661453,22 15.212234,22.449219 15.212234,23 C15.212234,23.550781 15.661453,24 16.212234,24 C16.763015,24 17.212234,23.550781 17.212234,23 C17.212234,22.449219 16.763015,22 16.212234,22 Z M20.212234,22 C19.661453,22 19.212234,22.449219 19.212234,23 C19.212234,23.550781 19.661453,24 20.212234,24 C20.763015,24 21.212234,23.550781 21.212234,23 C21.212234,22.449219 20.763015,22 20.212234,22 Z M24.212234,22 C23.661453,22 23.212234,22.449219 23.212234,23 C23.212234,23.550781 23.661453,24 24.212234,24 C24.763015,24 25.212234,23.550781 25.212234,23 C25.212234,22.449219 24.763015,22 24.212234,22 Z M28.212234,22 C27.661453,22 27.212234,22.449219 27.212234,23 C27.212234,23.550781 27.661453,24 28.212234,24 C28.763015,24 29.212234,23.550781 29.212234,23 C29.212234,22.449219 28.763015,22 28.212234,22 Z M32.212234,22 C31.661453,22 31.212234,22.449219 31.212234,23 C31.212234,23.550781 31.661453,24 32.212234,24 C32.763015,24 33.212234,23.550781 33.212234,23 C33.212234,22.449219 32.763015,22 32.212234,22 Z M36.212234,22 C35.661453,22 35.212234,22.449219 35.212234,23 C35.212234,23.550781 35.661453,24 36.212234,24 C36.763015,24 37.212234,23.550781 37.212234,23 C37.212234,22.449219 36.763015,22 36.212234,22 Z M40.212234,22 C39.661453,22 39.212234,22.449219 39.212234,23 C39.212234,23.550781 39.661453,24 40.212234,24 C40.763015,24 41.212234,23.550781 41.212234,23 C41.212234,22.449219 40.763015,22 40.212234,22 Z M44.212234,22 C43.661453,22 43.212234,22.449219 43.212234,23 C43.212234,23.550781 43.661453,24 44.212234,24 C44.763015,24 45.212234,23.550781 45.212234,23 C45.212234,22.449219 44.763015,22 44.212234,22 Z M48.212234,22 C47.661453,22 47.212234,22.449219 47.212234,23 C47.212234,23.550781 47.661453,24 48.212234,24 C48.763015,24 49.212234,23.550781 49.212234,23 C49.212234,22.449219 48.763015,22 48.212234,22 Z M52.212234,22 C51.661453,22 51.212234,22.449219 51.212234,23 C51.212234,23.550781 51.661453,24 52.212234,24 C52.763015,24 53.212234,23.550781 53.212234,23 C53.212234,22.449219 52.763015,22 52.212234,22 Z M21.462234,27.96875 C21.419265,27.976563 21.376297,27.988281 21.337234,28 C21.177078,28.027344 21.02864,28.089844 20.899734,28.1875 L15.618484,32.1875 C15.356765,32.375 15.200515,32.679688 15.200515,33 C15.200515,33.320313 15.356765,33.625 15.618484,33.8125 L20.899734,37.8125 C21.348953,38.148438 21.985672,38.058594 22.321609,37.609375 C22.657547,37.160156 22.567703,36.523438 22.118484,36.1875 L19.212234,34 L49.212234,34 L46.305984,36.1875 C45.856765,36.523438 45.766922,37.160156 46.102859,37.609375 C46.438797,38.058594 47.075515,38.148438 47.524734,37.8125 L52.805984,33.8125 C53.067703,33.625 53.223953,33.320313 53.223953,33 C53.223953,32.679688 53.067703,32.375 52.805984,32.1875 L47.524734,28.1875 C47.30989,28.027344 47.040359,27.960938 46.774734,28 C46.743484,28 46.712234,28 46.680984,28 C46.282547,28.074219 45.96614,28.382813 45.884109,28.78125 C45.802078,29.179688 45.970047,29.585938 46.305984,29.8125 L49.212234,32 L19.212234,32 L22.118484,29.8125 C22.520828,29.566406 22.696609,29.070313 22.536453,28.625 C22.380203,28.179688 21.930984,27.90625 21.462234,27.96875 Z M16.212234,42 C15.661453,42 15.212234,42.449219 15.212234,43 C15.212234,43.550781 15.661453,44 16.212234,44 C16.763015,44 17.212234,43.550781 17.212234,43 C17.212234,42.449219 16.763015,42 16.212234,42 Z M20.212234,42 C19.661453,42 19.212234,42.449219 19.212234,43 C19.212234,43.550781 19.661453,44 20.212234,44 C20.763015,44 21.212234,43.550781 21.212234,43 C21.212234,42.449219 20.763015,42 20.212234,42 Z M24.212234,42 C23.661453,42 23.212234,42.449219 23.212234,43 C23.212234,43.550781 23.661453,44 24.212234,44 C24.763015,44 25.212234,43.550781 25.212234,43 C25.212234,42.449219 24.763015,42 24.212234,42 Z M28.212234,42 C27.661453,42 27.212234,42.449219 27.212234,43 C27.212234,43.550781 27.661453,44 28.212234,44 C28.763015,44 29.212234,43.550781 29.212234,43 C29.212234,42.449219 28.763015,42 28.212234,42 Z M32.212234,42 C31.661453,42 31.212234,42.449219 31.212234,43 C31.212234,43.550781 31.661453,44 32.212234,44 C32.763015,44 33.212234,43.550781 33.212234,43 C33.212234,42.449219 32.763015,42 32.212234,42 Z M36.212234,42 C35.661453,42 35.212234,42.449219 35.212234,43 C35.212234,43.550781 35.661453,44 36.212234,44 C36.763015,44 37.212234,43.550781 37.212234,43 C37.212234,42.449219 36.763015,42 36.212234,42 Z M40.212234,42 C39.661453,42 39.212234,42.449219 39.212234,43 C39.212234,43.550781 39.661453,44 40.212234,44 C40.763015,44 41.212234,43.550781 41.212234,43 C41.212234,42.449219 40.763015,42 40.212234,42 Z M44.212234,42 C43.661453,42 43.212234,42.449219 43.212234,43 C43.212234,43.550781 43.661453,44 44.212234,44 C44.763015,44 45.212234,43.550781 45.212234,43 C45.212234,42.449219 44.763015,42 44.212234,42 Z M48.212234,42 C47.661453,42 47.212234,42.449219 47.212234,43 C47.212234,43.550781 47.661453,44 48.212234,44 C48.763015,44 49.212234,43.550781 49.212234,43 C49.212234,42.449219 48.763015,42 48.212234,42 Z M52.212234,42 C51.661453,42 51.212234,42.449219 51.212234,43 C51.212234,43.550781 51.661453,44 52.212234,44 C52.763015,44 53.212234,43.550781 53.212234,43 C53.212234,42.449219 52.763015,42 52.212234,42 Z" id="inside_range" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_vals_outside": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_outside</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_outside" transform="translate(0.000000, 0.689655)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <g id="outside_range" transform="translate(11.000000, 21.000000)" fill="#000000">
                <path d="M12.993484,0.96875 C11.962234,1.082031 11.188797,1.964844 11.212234,3 L11.212234,21 C11.200515,21.722656 11.579422,22.390625 12.204422,22.753906 C12.825515,23.121094 13.598953,23.121094 14.220047,22.753906 C14.845047,22.390625 15.223953,21.722656 15.212234,21 L15.212234,3 C15.220047,2.457031 15.009109,1.9375 14.626297,1.554688 C14.243484,1.171875 13.723953,0.960938 13.180984,0.96875 C13.118484,0.964844 13.055984,0.964844 12.993484,0.96875 Z M32.993484,0.96875 C31.962234,1.082031 31.188797,1.964844 31.212234,3 L31.212234,21 C31.200515,21.722656 31.579422,22.390625 32.204422,22.753906 C32.825515,23.121094 33.598953,23.121094 34.220047,22.753906 C34.845047,22.390625 35.223953,21.722656 35.212234,21 L35.212234,3 C35.220047,2.457031 35.009109,1.9375 34.626297,1.554688 C34.243484,1.171875 33.723953,0.960938 33.180984,0.96875 C33.118484,0.964844 33.055984,0.964844 32.993484,0.96875 Z M17.212234,1 C16.661453,1 16.212234,1.449219 16.212234,2 C16.212234,2.550781 16.661453,3 17.212234,3 C17.763015,3 18.212234,2.550781 18.212234,2 C18.212234,1.449219 17.763015,1 17.212234,1 Z M21.212234,1 C20.661453,1 20.212234,1.449219 20.212234,2 C20.212234,2.550781 20.661453,3 21.212234,3 C21.763015,3 22.212234,2.550781 22.212234,2 C22.212234,1.449219 21.763015,1 21.212234,1 Z M25.212234,1 C24.661453,1 24.212234,1.449219 24.212234,2 C24.212234,2.550781 24.661453,3 25.212234,3 C25.763015,3 26.212234,2.550781 26.212234,2 C26.212234,1.449219 25.763015,1 25.212234,1 Z M29.212234,1 C28.661453,1 28.212234,1.449219 28.212234,2 C28.212234,2.550781 28.661453,3 29.212234,3 C29.763015,3 30.212234,2.550781 30.212234,2 C30.212234,1.449219 29.763015,1 29.212234,1 Z M17.212234,21 C16.661453,21 16.212234,21.449219 16.212234,22 C16.212234,22.550781 16.661453,23 17.212234,23 C17.763015,23 18.212234,22.550781 18.212234,22 C18.212234,21.449219 17.763015,21 17.212234,21 Z M21.212234,21 C20.661453,21 20.212234,21.449219 20.212234,22 C20.212234,22.550781 20.661453,23 21.212234,23 C21.763015,23 22.212234,22.550781 22.212234,22 C22.212234,21.449219 21.763015,21 21.212234,21 Z M25.212234,21 C24.661453,21 24.212234,21.449219 24.212234,22 C24.212234,22.550781 24.661453,23 25.212234,23 C25.763015,23 26.212234,22.550781 26.212234,22 C26.212234,21.449219 25.763015,21 25.212234,21 Z M29.212234,21 C28.661453,21 28.212234,21.449219 28.212234,22 C28.212234,22.550781 28.661453,23 29.212234,23 C29.763015,23 30.212234,22.550781 30.212234,22 C30.212234,21.449219 29.763015,21 29.212234,21 Z" id="small_range" fill-rule="nonzero"></path>
                <path d="M6.8685059,6.97857933 C7.3372559,6.91607933 7.7864749,7.18951733 7.9427249,7.63482933 C8.1028809,8.08014233 7.9270999,8.57623533 7.5247559,8.82232933 L7.5247559,8.82232933 L4.6185059,11.0098293 L10.2107869,11.0097216 L10.2107869,13.0097216 L4.6185059,13.0098293 L7.5247559,15.1973293 C7.9739749,15.5332673 8.0638189,16.1699853 7.7278809,16.6192043 C7.3919439,17.0684233 6.7552249,17.1582673 6.3060059,16.8223293 L6.3060059,16.8223293 L1.0247559,12.8223293 C0.763036896,12.6348293 0.606786896,12.3301423 0.606786896,12.0098293 C0.606786896,11.6895173 0.763036896,11.3848293 1.0247559,11.1973293 L1.0247559,11.1973293 L6.3060059,7.19732933 C6.4349119,7.09967333 6.5833499,7.03717333 6.7435059,7.00982933 C6.7825689,6.99811033 6.8255369,6.98639233 6.8685059,6.97857933 Z" id="arrow_left"></path>
                <path d="M42.3690246,6.98102058 C42.8377746,6.91852058 43.2869936,7.19195858 43.4432436,7.63727058 C43.6033996,8.08258358 43.4276186,8.57867658 43.0252746,8.82477058 L43.0252746,8.82477058 L40.1190246,11.0122706 L45.7113056,11.0121629 L45.7113056,13.0121629 L40.1190246,13.0122706 L43.0252746,15.1997706 C43.4744936,15.5357086 43.5643376,16.1724266 43.2283996,16.6216456 C42.8924626,17.0708646 42.2557436,17.1607086 41.8065246,16.8247706 L41.8065246,16.8247706 L36.5252746,12.8247706 C36.2635556,12.6372706 36.1073056,12.3325836 36.1073056,12.0122706 C36.1073056,11.6919586 36.2635556,11.3872706 36.5252746,11.1997706 L36.5252746,11.1997706 L41.8065246,7.19977058 C41.9354306,7.10211458 42.0838686,7.03961458 42.2440246,7.01227058 C42.2830876,7.00055158 42.3260556,6.98883358 42.3690246,6.98102058 Z" id="arrow_right" transform="translate(40.909770, 12.000000) rotate(-180.000000) translate(-40.909770, -12.000000) "></path>
            </g>
        </g>
    </g>
</svg>""",
    "col_vals_in_set": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_in_set</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_in_set" transform="translate(0.000000, 0.172414)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M44.127969,41.1538382 L31.0814568,41.1538382 C29.9510748,41.1536429 28.8827052,40.9256134 27.9079888,40.5136953 C26.4467442,39.8960136 25.19849,38.8599685 24.3189894,37.5577099 C23.8792391,36.906727 23.5314818,36.1899233 23.2936866,35.4252675 C23.2130217,35.16589 23.1460289,34.9005554 23.0913409,34.6307286 L44.1278714,34.6307286 C45.028466,34.6306309 45.7586488,33.9004481 45.7586488,32.9998535 C45.7586488,32.0992589 45.028466,31.3690761 44.1278714,31.3690761 L23.0905596,31.3690761 C23.1990567,30.8337194 23.3597028,30.3180894 23.5675173,29.8264831 C24.185199,28.3652386 25.2212442,27.1169844 26.5236004,26.2374838 C27.1745833,25.7977334 27.891387,25.4499762 28.6560428,25.2122786 C29.4208939,24.9744833 30.2334994,24.8459665 31.0813591,24.8459665 L44.1277737,24.8459665 C45.0283683,24.8459665 45.7586488,24.1157837 45.7586488,23.2151891 C45.7586488,22.3145945 45.0283683,21.5844117 44.1277737,21.5844117 L31.0813591,21.5844117 C29.5096643,21.5844117 28.0039858,21.9038483 26.6373711,22.4820765 C24.5866678,23.3498583 22.8469049,24.7950871 21.6163267,26.616296 C20.3856508,28.4362354 19.665136,30.6413347 19.6658191,33.0000488 C19.6658191,34.5717436 19.9852563,36.0774222 20.5635822,37.4440369 C21.4312663,39.4947402 22.8765927,41.2345031 24.697704,42.4650813 C26.5176434,43.6957572 28.7227427,44.4155883 31.0814568,44.4155883 L44.1278714,44.4155883 C45.028466,44.4155883 45.7586488,43.6854055 45.7586488,42.7848109 C45.7586488,41.8842163 45.0285636,41.1538382 44.127969,41.1538382 Z" id="set_of" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_vals_not_in_set": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_not_in_set</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_not_in_set" transform="translate(0.000000, 0.655172)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M44.127969,41.1538382 L31.0814568,41.1538382 C29.9510748,41.1536429 28.8827052,40.9256134 27.9079888,40.5136953 C26.4467442,39.8960136 25.19849,38.8599685 24.3189894,37.5577099 C23.8792391,36.906727 23.5314818,36.1899233 23.2936866,35.4252675 C23.2130217,35.16589 23.1460289,34.9005554 23.0913409,34.6307286 L44.1278714,34.6307286 C45.028466,34.6306309 45.7586488,33.9004481 45.7586488,32.9998535 C45.7586488,32.0992589 45.028466,31.3690761 44.1278714,31.3690761 L23.0905596,31.3690761 C23.1990567,30.8337194 23.3597028,30.3180894 23.5675173,29.8264831 C24.185199,28.3652386 25.2212442,27.1169844 26.5236004,26.2374838 C27.1745833,25.7977334 27.891387,25.4499762 28.6560428,25.2122786 C29.4208939,24.9744833 30.2334994,24.8459665 31.0813591,24.8459665 L44.1277737,24.8459665 C45.0283683,24.8459665 45.7586488,24.1157837 45.7586488,23.2151891 C45.7586488,22.3145945 45.0283683,21.5844117 44.1277737,21.5844117 L31.0813591,21.5844117 C29.5096643,21.5844117 28.0039858,21.9038483 26.6373711,22.4820765 C24.5866678,23.3498583 22.8469049,24.7950871 21.6163267,26.616296 C20.3856508,28.4362354 19.665136,30.6413347 19.6658191,33.0000488 C19.6658191,34.5717436 19.9852563,36.0774222 20.5635822,37.4440369 C21.4312663,39.4947402 22.8765927,41.2345031 24.697704,42.4650813 C26.5176434,43.6957572 28.7227427,44.4155883 31.0814568,44.4155883 L44.1278714,44.4155883 C45.028466,44.4155883 45.7586488,43.6854055 45.7586488,42.7848109 C45.7586488,41.8842163 45.0285636,41.1538382 44.127969,41.1538382 Z" id="set_of" fill="#000000" fill-rule="nonzero"></path>
            <path d="M32.7624859,7.45752636 C33.0386282,7.45752636 33.2624859,7.68138398 33.2624859,7.95752636 L33.2624859,57.0855723 C33.2624859,57.3617146 33.0386282,57.5855723 32.7624859,57.5855723 C32.4863435,57.5855723 32.2624859,57.3617146 32.2624859,57.0855723 L32.2624859,7.95752636 C32.2624859,7.68138398 32.4863435,7.45752636 32.7624859,7.45752636 Z" id="line_black" fill="#000000" transform="translate(32.762486, 32.521549) rotate(-320.000000) translate(-32.762486, -32.521549) "></path>
            <polygon id="line_white" fill="#FFFFFF" transform="translate(34.661982, 31.674853) rotate(-320.000000) translate(-34.661982, -31.674853) " points="34.1619821 8.063154 35.1619821 8.063154 35.1619821 55.2865512 34.1619821 55.2865512"></polygon>
        </g>
    </g>
</svg>""",
    "col_vals_increasing": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_increasing</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_increasing" transform="translate(0.000000, 0.103448)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <g id="increasing" transform="translate(13.712234, 13.000000)" fill="#000000" fill-rule="nonzero">
                <path d="M31.2,0 L31.2,1.6 L37.25,1.6 L27.2,11.65 L24.575,9.025 L24,8.475 L23.425,9.025 L14.4,18.05 L11.775,15.425 L11.2,14.875 L10.625,15.425 L0.225,25.825 L1.375,26.975 L11.2,17.15 L13.825,19.775 L14.4,20.325 L14.975,19.775 L24,10.75 L26.625,13.375 L27.2,13.925 L27.775,13.375 L38.4,2.75 L38.4,8.8 L40,8.8 L40,0 L31.2,0 Z M33.6,11.2 L33.6,40 L35.2,40 L35.2,11.2 L33.6,11.2 Z M38.4,12 L38.4,40 L40,40 L40,12 L38.4,12 Z M24,16 L24,40 L25.6,40 L25.6,16 L24,16 Z M28.8,16 L28.8,40 L30.4,40 L30.4,16 L28.8,16 Z M19.2,19.2 L19.2,40 L20.8,40 L20.8,19.2 L19.2,19.2 Z M9.6,22.4 L9.6,40 L11.2,40 L11.2,22.4 L9.6,22.4 Z M14.4,24 L14.4,40 L16,40 L16,24 L14.4,24 Z M4.8,27.2 L4.8,40 L6.4,40 L6.4,27.2 L4.8,27.2 Z M0,30.4 L0,40 L1.6,40 L1.6,30.4 L0,30.4 Z" id="Shape"></path>
            </g>
        </g>
    </g>
</svg>""",
    "col_vals_decreasing": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_decreasing</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_decreasing" transform="translate(0.000000, 0.586207)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <g id="decreasing" transform="translate(13.712234, 12.500000)" fill="#000000" fill-rule="nonzero">
                <path d="M1.375,0.225 L0.225,1.375 L12.225,13.375 L12.8,13.925 L13.375,13.375 L16,10.75 L25.025,19.775 L25.6,20.325 L26.175,19.775 L28.8,17.15 L37.25,25.6 L32,25.6 L32,27.2 L40,27.2 L40,18.4 L38.4,18.4 L38.4,24.45 L29.375,15.425 L28.8,14.875 L28.225,15.425 L25.6,18.05 L16.575,9.025 L16,8.475 L15.425,9.025 L12.8,11.65 L1.375,0.225 Z M0,6.4 L0,40 L1.6,40 L1.6,6.4 L0,6.4 Z M4.8,11.2 L4.8,40 L6.4,40 L6.4,11.2 L4.8,11.2 Z M9.6,16 L9.6,40 L11.2,40 L11.2,16 L9.6,16 Z M14.4,16 L14.4,40 L16,40 L16,16 L14.4,16 Z M19.2,19.2 L19.2,40 L20.8,40 L20.8,19.2 L19.2,19.2 Z M28.8,22.4 L28.8,40 L30.4,40 L30.4,22.4 L28.8,22.4 Z M24,24 L24,40 L25.6,40 L25.6,24 L24,24 Z M33.6,30.4 L33.6,40 L35.2,40 L35.2,30.4 L33.6,30.4 Z M38.4,30.4 L38.4,40 L40,40 L40,30.4 L38.4,30.4 Z" id="Shape"></path>
            </g>
        </g>
    </g>
</svg>""",
    "col_vals_null": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_null</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_null" transform="translate(0.000000, 0.068966)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M40.6120891,47.0378516 C37.4692395,47.0378516 35.0126155,45.9348776 33.712234,44.0140735 C32.4118524,45.9348776 29.9552284,47.0378516 26.8123796,47.0378516 C22.6574258,47.0378516 16.0646491,43.4437854 16.0646491,33.802163 C16.0646491,29.3401315 17.4715675,18.9621484 30.5035822,18.9621484 C30.9453983,18.9621484 31.3057451,19.3224952 31.3057451,19.7643113 L31.3057451,21.3686372 C31.3057451,21.8104533 30.9453983,22.1708002 30.5035822,22.1708002 C26.6400397,22.1708002 22.4819527,25.8118683 22.4819527,33.802163 C22.4819527,37.5090333 23.7635331,43.0270369 27.2949304,43.0270369 C29.7954231,43.0270369 31.2242758,40.4231405 32.0985075,38.2861287 C30.5067154,35.6101628 29.7014192,33.1034036 29.7014192,30.8347865 C29.7014192,25.6238614 31.8603654,23.7751261 33.712234,23.7751261 C35.5641025,23.7751261 37.7230487,25.6238614 37.7230487,30.8347865 C37.7230487,33.1347385 36.9396869,35.5788287 35.3290937,38.2861287 C36.6294752,41.4321115 38.2432017,43.0270369 40.1295376,43.0270369 C43.6609348,43.0270369 44.9425153,37.5090333 44.9425153,33.802163 C44.9425153,25.8118683 40.7844282,22.1708002 36.9208858,22.1708002 C36.4759364,22.1708002 36.1187228,21.8104533 36.1187228,21.3686372 L36.1187228,19.7643113 C36.1187228,19.3224952 36.4759364,18.9621484 36.9208858,18.9621484 C49.9529005,18.9621484 51.3598189,29.3401315 51.3598189,33.802163 C51.3598189,43.4437854 44.7670421,47.0378516 40.6120891,47.0378516 Z" id="omega" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_pct_null": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF;">
    <title>pct_null</title>
    <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_null" transform="translate(1.000000, 1.581717)" fill-rule="nonzero">
            <path d="M55,0 C57.4852813,0 59.7352813,1.00735931 61.363961,2.63603897 C62.9926407,4.26471863 64,6.51471863 64,9 L64,9 L64,64 L9,64 C6.51471862,64 4.26471862,62.9926407 2.63603897,61.363961 C1.00735931,59.7352814 0,57.4852814 0,55 L0,55 L0,9 C0,6.51471863 1.00735931,4.26471863 2.63603897,2.63603897 C4.26471862,1.00735931 6.51471862,0 9,0 L9,0 L55,0 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M45.4146228,41.5 C43.2877254,41.5 41.6252254,40.7535715 40.745203,39.4536832 C39.8651805,40.7535715 38.2026805,41.5 36.0757835,41.5 C33.2639529,41.5 28.8023458,39.0677454 28.8023458,32.5428572 C28.8023458,29.5232143 29.7544662,22.5 38.5737744,22.5 C38.8727698,22.5 39.1166315,22.7438617 39.1166315,23.0428571 L39.1166315,24.1285714 C39.1166315,24.4275668 38.8727698,24.6714286 38.5737744,24.6714286 C35.9591539,24.6714286 33.1452029,27.1354913 33.1452029,32.5428572 C33.1452029,35.051451 34.0125019,38.7857143 36.4023458,38.7857143 C38.0945336,38.7857143 39.0614978,37.023549 39.6531269,35.5773439 C38.5758948,33.7664061 38.0309172,32.0699775 38.0309172,30.5347097 C38.0309172,27.0082592 39.4919662,25.7571429 40.745203,25.7571429 C41.9984397,25.7571429 43.4594886,27.0082592 43.4594886,30.5347097 C43.4594886,32.0911832 42.929355,33.745201 41.8393994,35.5773439 C42.7194218,37.7063617 43.8114979,38.7857143 45.0880601,38.7857143 C47.477904,38.7857143 48.345203,35.051451 48.345203,32.5428572 C48.345203,27.1354913 45.5312519,24.6714286 42.9166315,24.6714286 C42.6155157,24.6714286 42.3737744,24.4275668 42.3737744,24.1285714 L42.3737744,23.0428571 C42.3737744,22.7438617 42.6155157,22.5 42.9166315,22.5 C51.7359397,22.5 52.6880601,29.5232143 52.6880601,32.5428572 C52.6880601,39.0677454 48.2264529,41.5 45.4146228,41.5 Z" id="omega" fill="#000000"></path>
            <g id="percent" transform="translate(11.268508, 23.854373)" fill="#000000">
                <path d="M1.89920553,17.2037988 C1.78720553,17.1237988 1.664,17.0296274 1.584,16.8936274 C1.504,16.7576274 1.464,16.6096274 1.464,16.4496274 C1.464,16.2416274 1.536,16.0336274 1.68,15.8256274 L12.24,0.489627434 C12.432,0.185627434 12.5942662,0 12.9462662,0 C13.1542662,0 13.472,0.089627434 13.68,0.201627434 C14.048,0.425627434 14.232,0.681627434 14.232,0.969627434 C14.232,1.14562743 14.16,1.34562743 14.016,1.56962743 L3.432,16.9776274 C3.176,17.2816274 2.88,17.4336274 2.544,17.4336274 C2.336,17.4336274 2.13920553,17.3477988 1.89920553,17.2037988 Z M3.864,7.47362743 C3.176,7.47362743 2.536,7.30562743 1.944,6.96962743 C1.352,6.63362743 0.88,6.18162743 0.528,5.61362743 C0.176,5.04562743 0,4.42562743 0,3.75362743 C0,3.08162743 0.172,2.46162743 0.516,1.89362743 C0.86,1.32562743 1.328,0.877627434 1.92,0.549627434 C2.512,0.221627434 3.16,0.057627434 3.864,0.057627434 C4.568,0.057627434 5.216,0.221627434 5.808,0.549627434 C6.4,0.877627434 6.864,1.32562743 7.2,1.89362743 C7.536,2.46162743 7.704,3.08162743 7.704,3.75362743 C7.704,4.42562743 7.532,5.04562743 7.188,5.61362743 C6.844,6.18162743 6.38,6.63362743 5.796,6.96962743 C5.212,7.30562743 4.568,7.47362743 3.864,7.47362743 Z M3.864,5.69762743 C4.408,5.69762743 4.852,5.51362743 5.196,5.14562743 C5.54,4.77762743 5.712,4.31362743 5.712,3.75362743 C5.712,3.17762743 5.54,2.70562743 5.196,2.33762743 C4.852,1.96962743 4.408,1.78562743 3.864,1.78562743 C3.304,1.78562743 2.848,1.96962743 2.496,2.33762743 C2.144,2.70562743 1.968,3.17762743 1.968,3.75362743 C1.968,4.31362743 2.144,4.77762743 2.496,5.14562743 C2.848,5.51362743 3.304,5.69762743 3.864,5.69762743 Z M11.952,17.3856274 C11.248,17.3856274 10.6,17.2176274 10.008,16.8816274 C9.416,16.5456274 8.948,16.0936274 8.604,15.5256274 C8.26,14.9576274 8.088,14.3376274 8.088,13.6656274 C8.088,12.9936274 8.26,12.3736274 8.604,11.8056274 C8.948,11.2376274 9.416,10.7896274 10.008,10.4616274 C10.6,10.1336274 11.248,9.96962743 11.952,9.96962743 C12.656,9.96962743 13.3,10.1336274 13.884,10.4616274 C14.468,10.7896274 14.928,11.2376274 15.264,11.8056274 C15.6,12.3736274 15.768,12.9936274 15.768,13.6656274 C15.768,14.3376274 15.596,14.9576274 15.252,15.5256274 C14.908,16.0936274 14.444,16.5456274 13.86,16.8816274 C13.276,17.2176274 12.64,17.3856274 11.952,17.3856274 Z M11.952,15.6096274 C12.48,15.6096274 12.92,15.4296274 13.272,15.0696274 C13.624,14.7096274 13.8,14.2416274 13.8,13.6656274 C13.8,13.1056274 13.624,12.6416274 13.272,12.2736274 C12.92,11.9056274 12.48,11.7216274 11.952,11.7216274 C11.392,11.7216274 10.932,11.9056274 10.572,12.2736274 C10.212,12.6416274 10.032,13.1056274 10.032,13.6656274 C10.032,14.2416274 10.212,14.7096274 10.572,15.0696274 C10.932,15.4296274 11.392,15.6096274 11.952,15.6096274 Z"></path>
            </g>
        </g>
    </g>
</svg>""",
    "col_vals_not_null": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_not_null</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_not_null" transform="translate(0.000000, 0.551724)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M40.6120805,47.037834 C37.4692348,47.037834 35.0126139,45.9348613 33.712234,44.0140597 C32.4118541,45.9348613 29.9552331,47.037834 26.8123883,47.037834 C22.6574397,47.037834 16.0646712,43.4437723 16.0646712,33.8021619 C16.0646712,29.3401361 17.4715879,18.962166 30.5035862,18.962166 C30.9454018,18.962166 31.3057481,19.3225124 31.3057481,19.7643279 L31.3057481,21.3686518 C31.3057481,21.8104674 30.9454018,22.1708138 30.5035862,22.1708138 C26.6400486,22.1708138 22.4819668,25.8118774 22.4819668,33.8021619 C22.4819668,37.5090277 23.7635456,43.0270243 27.2949384,43.0270243 C29.795428,43.0270243 31.224279,40.4231312 32.0985095,38.2861221 C30.5067194,35.6101596 29.7014243,33.1034035 29.7014243,30.8347892 C29.7014243,25.6238707 31.8603677,23.7751377 33.712234,23.7751377 C35.5641002,23.7751377 37.7230437,25.6238707 37.7230437,30.8347892 C37.7230437,33.1347383 36.9396828,35.5788255 35.3290916,38.2861221 C36.6294715,41.4321009 38.243196,43.0270243 40.1295295,43.0270243 C43.6609223,43.0270243 44.9425012,37.5090277 44.9425012,33.8021619 C44.9425012,25.8118774 40.7844193,22.1708138 36.9208817,22.1708138 C36.4759329,22.1708138 36.1187198,21.8104674 36.1187198,21.3686518 L36.1187198,19.7643279 C36.1187198,19.3225124 36.4759329,18.962166 36.9208817,18.962166 C49.9528801,18.962166 51.3597967,29.3401361 51.3597967,33.8021619 C51.3597967,43.4437723 44.7670282,47.037834 40.6120805,47.037834 Z" id="omega" fill="#000000" fill-rule="nonzero"></path>
            <path d="M33,7.93597705 C33.2761424,7.93597705 33.5,8.15983467 33.5,8.43597705 L33.5,57.564023 C33.5,57.8401653 33.2761424,58.064023 33,58.064023 C32.7238576,58.064023 32.5,57.8401653 32.5,57.564023 L32.5,8.43597705 C32.5,8.15983467 32.7238576,7.93597705 33,7.93597705 Z" id="line_black" fill="#000000" transform="translate(33.000000, 33.000000) rotate(-320.000000) translate(-33.000000, -33.000000) "></path>
            <polygon id="line_white" fill="#FFFFFF" transform="translate(34.899496, 32.153303) rotate(-320.000000) translate(-34.899496, -32.153303) " points="34.3994962 8.54160469 35.3994962 8.54160469 35.3994962 55.7650019 34.3994962 55.7650019"></polygon>
        </g>
    </g>
</svg>""",
    "col_vals_regex": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_regex</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_regex" transform="translate(0.000000, 0.034483)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <g id="regex_symbols" transform="translate(18.000000, 12.000000)" fill="#000000" fill-rule="nonzero">
                <path d="M4.17434508,33.013582 C1.94895328,33.013582 0.138006923,34.8245284 0.138006923,37.0499202 C0.138006923,39.275312 1.94895328,41.0862583 4.17434508,41.0862583 C6.39973688,41.0862583 8.21068324,39.275312 8.21068324,37.0499202 C8.21068324,34.8245284 6.39973688,33.013582 4.17434508,33.013582 Z" id="full_stop"></path>
                <path d="M23.9479718,23.3175402 L21.5628264,23.3175402 C21.2344032,23.3175402 20.9665401,23.0520067 20.9665401,22.7212538 L20.9665401,15.1022979 L14.3445004,18.8873192 C14.0626621,19.050366 13.7016292,18.952538 13.5362533,18.6706991 L12.3436806,16.6442575 C12.262157,16.506832 12.2388642,16.3437852 12.2807909,16.1900549 C12.3203879,16.0363251 12.4205455,15.9058874 12.557971,15.8266929 L19.1800101,11.9880994 L12.557971,8.15183511 C12.4205455,8.07264112 12.3203879,7.93987439 12.2807909,7.78614401 C12.2388642,7.63241423 12.262157,7.46936689 12.3413509,7.33194137 L13.5339237,5.30549975 C13.6993001,5.02366143 14.0626621,4.92816199 14.3445004,5.09120934 L20.9665401,8.87390091 L20.9665401,1.25494501 C20.9665401,0.926521818 21.2344032,0.658658658 21.5628264,0.658658658 L23.9479718,0.658658658 C24.2787247,0.658658658 24.5442582,0.926521818 24.5442582,1.25494501 L24.5442582,8.87390091 L31.1662979,5.09120934 C31.4481362,4.92816199 31.8091691,5.02366143 31.9745455,5.30549975 L33.1671182,7.33194137 C33.2486413,7.46936689 33.2719341,7.63241423 33.2300074,7.78614401 C33.1904104,7.93987439 33.0902528,8.07264112 32.9528278,8.15183511 L26.3307882,11.9880994 L32.9528278,15.8243638 C33.0879237,15.9058874 33.1880813,16.0363251 33.2300074,16.1900549 C33.269605,16.3437852 33.2486413,16.506832 33.1671182,16.6442575 L31.9745455,18.6706991 C31.8091691,18.952538 31.4481362,19.050366 31.1662979,18.8849895 L24.5442582,15.1022979 L24.5442582,22.7212538 C24.5442582,23.0520067 24.2787247,23.3175402 23.9479718,23.3175402 Z" id="asterisk"></path>
            </g>
        </g>
    </g>
</svg>""",
    "col_vals_within_spec": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_within_spec</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_within_spec" transform="translate(0.000000, 0.517241)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <g id="globe" transform="translate(9.712234, 9.000000)" fill="#000000" fill-rule="nonzero">
                <path d="M24,0.9199219 C11.265113,0.9199219 0.9199219,11.265113 0.9199219,24 C0.9199219,36.734887 11.265113,47.080078 24,47.080078 C36.734887,47.080078 47.080078,36.734887 47.080078,24 C47.080078,11.265113 36.734887,0.9199219 24,0.9199219 Z M23,3.0371094 L23,12.970703 C20.43251,12.91011 17.991445,12.623022 15.740234,12.152344 C16.136627,10.9515 16.587103,9.835559 17.087891,8.8339844 C18.740825,5.5281156 20.837286,3.5160498 23,3.0371094 Z M25,3.0371094 C27.162714,3.5160498 29.259175,5.5281156 30.912109,8.8339844 C31.414496,9.838757 31.866379,10.958806 32.263672,12.164062 C30.015269,12.630037 27.5709,12.911377 25,12.970703 L25,3.0371094 Z M18.171875,3.7402344 C17.086481,4.8772845 16.118637,6.2998344 15.298828,7.9394531 C14.727502,9.082106 14.226366,10.34271 13.789062,11.6875 C11.825936,11.158279 10.050587,10.492361 8.5175781,9.71875 C11.097501,6.9227151 14.411073,4.8180109 18.171875,3.7402344 Z M29.828125,3.7402344 C33.585289,4.8169681 36.895737,6.9189636 39.474609,9.710938 C37.96425,10.498668 36.190868,11.172098 34.216797,11.703125 C33.778353,10.352409 33.274712,9.086534 32.701172,7.9394531 C31.881363,6.2998344 30.913519,4.8772845 29.828125,3.7402344 Z M40.783203,11.273438 C43.280319,14.563254 44.849591,18.596544 45.054688,23 L36.013672,23 C35.940686,19.642697 35.511581,16.472843 34.777344,13.632812 C37.021162,13.025788 39.043564,12.230357 40.783203,11.273438 Z M7.2050781,11.289062 C8.9635362,12.222752 10.989301,13.00879 13.226562,13.615234 C12.489633,16.459671 12.059462,19.635904 11.986328,23 L2.9453125,23 C3.1500856,18.603485 4.7148727,14.576078 7.2050781,11.289062 Z M15.175781,14.085938 C17.608124,14.603479 20.237145,14.911692 23,14.972656 L23,23 L13.986328,23 C14.060725,19.787369 14.480743,16.762271 15.175781,14.085938 Z M32.828125,14.099609 C33.521088,16.7726 33.939401,19.792796 34.013672,23 L25,23 L25,14.972656 C27.764457,14.913393 30.396477,14.612271 32.828125,14.099609 Z M2.9453125,25 L11.986328,25 C12.059314,28.357303 12.488419,31.527156 13.222656,34.367188 C10.978838,34.974212 8.9564363,35.769643 7.2167969,36.726562 C4.7196806,33.436746 3.1504088,29.403456 2.9453125,25 Z M13.986328,25 L23,25 L23,33.027344 C20.235543,33.086607 17.603523,33.387729 15.171875,33.900391 C14.478912,31.2274 14.060599,28.207204 13.986328,25 Z M25,25 L34.013672,25 C33.939275,28.212631 33.519257,31.237729 32.824219,33.914062 C30.391876,33.396521 27.762855,33.088308 25,33.027344 L25,25 Z M36.013672,25 L45.054688,25 C44.849914,29.396515 43.285127,33.423922 40.794922,36.710938 C39.036464,35.777248 37.010699,34.99121 34.773438,34.384766 C35.510367,31.540329 35.940538,28.364096 36.013672,25 Z M23,35.029297 L23,44.962891 C20.837286,44.48395 18.740825,42.471884 17.087891,39.166016 C16.585504,38.161243 16.133621,37.041194 15.736328,35.835938 C17.984731,35.369963 20.4291,35.088623 23,35.029297 Z M25,35.029297 C27.56749,35.08989 30.008555,35.376978 32.259766,35.847656 C31.863373,37.0485 31.412897,38.16444 30.912109,39.166016 C29.259175,42.471884 27.162714,44.48395 25,44.962891 L25,35.029297 Z M13.783203,36.296875 C14.221647,37.647591 14.725288,38.913466 15.298828,40.060547 C16.118637,41.700166 17.086481,43.122716 18.171875,44.259766 C14.414711,43.183032 11.104263,41.081036 8.5253906,38.289062 C10.03575,37.501332 11.809132,36.827902 13.783203,36.296875 Z M34.210938,36.3125 C36.174064,36.841721 37.949413,37.507639 39.482422,38.28125 C36.902499,41.077285 33.588927,43.181989 29.828125,44.259766 C30.913519,43.122716 31.881363,41.700166 32.701172,40.060547 C33.272498,38.917894 33.773634,37.65729 34.210938,36.3125 Z" id="Shape"></path>
            </g>
        </g>
    </g>
</svg>""",
    "col_exists": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_exists</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_exists" transform="translate(0.000000, 0.827586)">
            <path d="M56.712234,1.01466935 C59.1975153,1.01466935 61.4475153,2.02202867 63.076195,3.65070832 C64.7048747,5.27938798 65.712234,7.52938798 65.712234,10.0146694 L65.712234,10.0146694 L65.712234,65.0146694 L10.712234,65.0146694 C8.22695259,65.0146694 5.97695259,64.00731 4.34827294,62.3786304 C2.71959328,60.7499507 1.71223397,58.4999507 1.71223397,56.0146694 L1.71223397,56.0146694 L1.71223397,10.0146694 C1.71223397,7.52938798 2.71959328,5.27938798 4.34827294,3.65070832 C5.97695259,2.02202867 8.22695259,1.01466935 10.712234,1.01466935 L10.712234,1.01466935 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <rect id="column" fill="#000000" x="12.2117153" y="12.0146694" width="20" height="42" rx="1"></rect>
            <path d="M44.3177114,43.0146694 L44.3177114,40.5136928 L46.818688,40.5136928 L46.818688,43.0146694 L44.3177114,43.0146694 Z M44.3177114,38.0000209 L44.3177114,37.314474 C44.3177114,35.6979295 44.9397755,34.178739 46.1839224,32.7568569 L46.9837271,31.830099 C48.3125097,30.3066539 48.9768911,29.058294 48.9768911,28.0849819 C48.9768911,27.3317229 48.6849019,26.7350492 48.1009146,26.2949428 C47.5169273,25.8548364 46.7298258,25.6347865 45.7395864,25.6347865 C44.4446581,25.6347865 43.0693463,25.9479345 41.6136099,26.5742397 L41.6136099,24.4541225 C43.1793729,23.9801618 44.643551,23.743185 46.006188,23.743185 C47.7327591,23.743185 49.1038392,24.1303881 50.1194692,24.9048061 C51.1350993,25.679224 51.6429067,26.7265768 51.6429067,28.0468959 C51.6429067,28.7916913 51.4969121,29.4327982 51.2049185,29.9702358 C50.9129248,30.5076733 50.3522208,31.1699389 49.5227896,31.9570522 L48.7356802,32.6933803 C47.9485669,33.4381757 47.432296,34.0623556 47.1868521,34.5659389 C46.9414081,35.0695221 46.818688,35.7487146 46.818688,36.6035365 L46.818688,38.0000209 L44.3177114,38.0000209 Z" id="?" fill="#000000"></path>
        </g>
    </g>
</svg>""",
    "rows_distinct": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>rows_distinct</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="rows_distinct" transform="translate(0.000000, 0.482759)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <g id="no_gemini" transform="translate(17.000000, 13.000000)">
                <path d="M3.66705619,6.62107508 C3.12510104,6.64066386 2.67455974,7.04767444 2.60273432,7.58527682 C2.52873228,8.1228792 2.85303526,8.6343627 3.37104848,8.79760239 C4.40489909,9.15455286 6.70113553,9.87063021 9.86580595,10.364702 L9.86580595,30.7369976 C6.65978137,31.2332458 4.37225104,31.964559 3.37104848,32.3215095 C2.78991555,32.5282797 2.48520173,33.1681784 2.69197196,33.7493114 C2.8987422,34.3304443 3.53864094,34.6351581 4.11977387,34.4283879 C5.54975217,33.9190808 10.2031678,32.4608072 16.5172734,32.4608072 C22.7943781,32.4608072 27.5500901,33.8907855 29.0540706,34.4109757 C29.6352036,34.6133926 30.2707495,34.304326 30.4731664,33.7231931 C30.6755833,33.1420601 30.3665167,32.5065142 29.7853838,32.3040973 C28.7493568,31.9449704 26.4313554,31.2441283 23.2383897,30.7544098 L23.2383897,10.3821143 C26.4444143,9.88804243 28.745004,9.16108259 29.7679716,8.79760239 C30.3491045,8.59083215 30.6538184,7.95093341 30.4470481,7.36980048 C30.2402779,6.78866755 29.6003791,6.48395373 29.0192462,6.69072396 C27.5587963,7.20873774 22.9162637,8.65830464 16.6391589,8.65830464 C10.3707603,8.65830464 5.61287188,7.21309051 4.10236166,6.69072396 C3.96306391,6.6384873 3.81506005,6.61454548 3.66705619,6.62107508 Z M12.0945699,10.6432975 C13.4940771,10.789125 15.0198226,10.8870686 16.6391589,10.8870686 C18.1997289,10.8870686 19.6623552,10.7978311 21.0096257,10.6607098 L21.0096257,30.4584021 C19.623178,30.316928 18.1191975,30.2320433 16.5172734,30.2320433 C14.9327615,30.2320433 13.4570763,30.3191044 12.0945699,30.4584021 L12.0945699,10.6432975 Z" id="gemini" stroke="#000000" stroke-width="2" fill="#000000" fill-rule="nonzero"></path>
                <path d="M16.6605354,-5.05929499 C16.9366778,-5.05929499 17.1605354,-4.83543737 17.1605354,-4.55929499 L17.1605354,44.5687509 C17.1605354,44.8448933 16.9366778,45.0687509 16.6605354,45.0687509 C16.384393,45.0687509 16.1605354,44.8448933 16.1605354,44.5687509 L16.1605354,-4.55929499 C16.1605354,-4.83543737 16.384393,-5.05929499 16.6605354,-5.05929499 Z" id="line_black" fill="#000000" transform="translate(16.660535, 20.004728) rotate(-320.000000) translate(-16.660535, -20.004728) "></path>
                <polygon id="line_white" fill="#FFFFFF" transform="translate(18.560032, 19.158031) rotate(-320.000000) translate(-18.560032, -19.158031) " points="18.0600316 -4.45366735 19.0600316 -4.45366735 19.0600316 42.7697299 18.0600316 42.7697299"></polygon>
            </g>
        </g>
    </g>
</svg>""",
    "rows_complete": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>rows_complete</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="rows_complete" transform="translate(0.000000, 0.965517)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <g id="complete_me" transform="translate(12.500000, 9.500000)" fill="#000000">
                <path d="M8,0 L8,10 L16,10 L16,18 L26,18 L26,10 L34,10 L34,0 L8,0 Z M10,2 L16,2 L16,8 L10,8 L10,2 Z M18,2 L24,2 L24,8 L18,8 L18,2 Z M26,2 L32,2 L32,8 L26,8 L26,2 Z M18,10 L24,10 L24,16 L18,16 L18,10 Z M0,21 L0,47 L42,47 L42,21 L32,21 L32,29 L24,29 L24,37 L18,37 L18,29 L10,29 L10,21 L0,21 Z M2,23 L8,23 L8,29 L2,29 L2,23 Z M34,23 L40,23 L40,29 L34,29 L34,23 Z M2,31 L8,31 L8,37 L2,37 L2,31 Z M10,31 L16,31 L16,37 L10,37 L10,31 Z M26,31 L32,31 L32,37 L26,37 L26,31 Z M34,31 L40,31 L40,37 L34,37 L34,31 Z M2,39 L8,39 L8,45 L2,45 L2,39 Z M10,39 L16,39 L16,45 L10,45 L10,39 Z M18,39 L24,39 L24,45 L18,45 L18,39 Z M26,39 L32,39 L32,45 L26,45 L26,39 Z M34,39 L40,39 L40,45 L34,45 L34,39 Z" id="Shape" fill-rule="nonzero"></path>
                <path d="M22.4566476,18.35817 C22.9253976,18.29567 23.3746166,18.569108 23.5308666,19.01442 C23.6910226,19.459733 23.5152416,19.955826 23.1128976,20.20192 L23.1128976,20.20192 L20.2066476,22.38942 L25.7989286,22.3893123 L25.7989286,24.3893123 L20.2066476,24.38942 L23.1128976,26.57692 C23.5621166,26.912858 23.6519606,27.549576 23.3160226,27.998795 C22.9800856,28.448014 22.3433666,28.537858 21.8941476,28.20192 L21.8941476,28.20192 L16.6128976,24.20192 C16.3511786,24.01442 16.1949286,23.709733 16.1949286,23.38942 C16.1949286,23.069108 16.3511786,22.76442 16.6128976,22.57692 L16.6128976,22.57692 L21.8941476,18.57692 C22.0230536,18.479264 22.1714916,18.416764 22.3316476,18.38942 C22.3707106,18.377701 22.4136786,18.365983 22.4566476,18.35817 Z" id="arrow_right" transform="translate(20.997393, 23.377149) rotate(-90.000000) translate(-20.997393, -23.377149) "></path>
            </g>
        </g>
    </g>
</svg>""",
    "prompt": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>prompt</title>
    <defs>
        <radialGradient id="EyeCore" cx="50%" cy="50%" r="50%">
            <stop offset="0%" style="stop-color:#ff0000;stop-opacity:1" />
            <stop offset="70%" style="stop-color:#cc0000;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#990000;stop-opacity:1" />
        </radialGradient>
        <radialGradient id="EyeOuter" cx="50%" cy="50%" r="50%">
            <stop offset="0%" style="stop-color:#ff3333;stop-opacity:0.8" />
            <stop offset="50%" style="stop-color:#ff0000;stop-opacity:0.9" />
            <stop offset="100%" style="stop-color:#660000;stop-opacity:1" />
        </radialGradient>
        <filter id="redGlow" color-interpolation-filters="sRGB">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
    </defs>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="prompt" transform="translate(0.000000, 0.965517)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#333333" stroke-width="1" fill="#000000"></path>
            <g id="eye" transform="translate(33.5, 33.5)">
                <circle cx="0" cy="0" r="18" fill="#ff0000" opacity="0.3" filter="url(#redGlow)"/>
                <circle cx="0" cy="0" r="15" fill="url(#EyeOuter)" stroke="#ff0000" stroke-width="1"/>
                <circle cx="0" cy="0" r="12" fill="none" stroke="#ff6666" stroke-width="1.2" opacity="0.8"/>
                <circle cx="0" cy="0" r="9" fill="url(#EyeCore)"/>
                <circle cx="0" cy="0" r="6" fill="none" stroke="#ffcccc" stroke-width="0.8" opacity="0.6"/>
                <circle cx="0" cy="0" r="3" fill="#000000" stroke="#ff0000" stroke-width="0.5"/>
                <circle cx="0" cy="0" r="1" fill="#ff0000"/>
                <rect x="-1" y="-8" width="2" height="6" fill="#ffffff" opacity="0.7" rx="1"/>
                <rect x="-6" y="-2" width="4" height="1" fill="#ffffff" opacity="0.5" rx="0.5"/>
            </g>
            <g id="corner_brackets" stroke="#333333" stroke-width="1" fill="none" opacity="0.6">
                <path d="M8,12 L8,8 L12,8"/>
                <path d="M55,8 L59,8 L59,12"/>
                <path d="M59,54 L59,58 L55,58"/>
                <path d="M12,58 L8,58 L8,54"/>
            </g>
        </g>
    </g>
</svg>""",
    "col_schema_match": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_schema_match</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_schema_match" transform="translate(0.000000, 0.310345)">
            <path d="M56.712234,1.01466935 C59.1975153,1.01466935 61.4475153,2.02202867 63.076195,3.65070832 C64.7048747,5.27938798 65.712234,7.52938798 65.712234,10.0146694 L65.712234,10.0146694 L65.712234,65.0146694 L10.712234,65.0146694 C8.22695259,65.0146694 5.97695259,64.00731 4.34827294,62.3786304 C2.71959328,60.7499507 1.71223397,58.4999507 1.71223397,56.0146694 L1.71223397,56.0146694 L1.71223397,10.0146694 C1.71223397,7.52938798 2.71959328,5.27938798 4.34827294,3.65070832 C5.97695259,2.02202867 8.22695259,1.01466935 10.712234,1.01466935 L10.712234,1.01466935 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M53.712234,39.7885268 L54.212234,56.2885268 L42.212234,56.7885268 L42.212234,39.7885268 L53.712234,39.7885268 Z M39.712234,39.7885268 L39.712234,56.7885268 L27.712234,56.7885268 L27.712234,39.7885268 L39.712234,39.7885268 Z M25.212234,39.7885268 L25.212234,56.7885268 L13.712234,56.7885268 L13.212234,40.2885268 L25.212234,39.7885268 Z" id="columns_schema" stroke="#000000" fill-rule="nonzero"></path>
            <g id="vertical_equal" transform="translate(30.000000, 29.000000)" stroke="#000000" stroke-linecap="square">
                <line x1="2.21223397" y1="0.514669353" x2="2.21223397" y2="7.58573716" id="Line"></line>
                <line x1="5.21223397" y1="0.514669353" x2="5.21223397" y2="7.58573716" id="Line-Copy"></line>
            </g>
            <path d="M41.712234,9.01466935 L41.712234,27.0146694 L53.712234,27.0146694 C54.262234,27.0146694 54.712234,26.5646694 54.712234,26.0146694 L54.712234,10.0146694 C54.712234,9.46466935 54.262234,9.01466935 53.712234,9.01466935 L41.712234,9.01466935 Z M27.212234,9.01466935 C27.212234,9.01466935 27.212234,15.0146694 27.212234,27.0146694 L40.212234,27.0146694 L40.212234,9.01466935 C31.5455673,9.01466935 27.212234,9.01466935 27.212234,9.01466935 Z M13.712234,9.01466935 C13.162234,9.01466935 12.712234,9.46466935 12.712234,10.0146694 L12.712234,26.0146694 C12.712234,26.5646694 13.162234,27.0146694 13.712234,27.0146694 L25.712234,27.0146694 L25.712234,9.01466935 L13.712234,9.01466935 Z" id="columns_real" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "row_count_match": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>row_count_match</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="row_count_match" transform="translate(0.000000, 0.793103)">
            <path d="M56.712234,1.01466935 C59.1975153,1.01466935 61.4475153,2.02202867 63.076195,3.65070832 C64.7048747,5.27938798 65.712234,7.52938798 65.712234,10.0146694 L65.712234,10.0146694 L65.712234,65.0146694 L10.712234,65.0146694 C8.22695259,65.0146694 5.97695259,64.00731 4.34827294,62.3786304 C2.71959328,60.7499507 1.71223397,58.4999507 1.71223397,56.0146694 L1.71223397,56.0146694 L1.71223397,10.0146694 C1.71223397,7.52938798 2.71959328,5.27938798 4.34827294,3.65070832 C5.97695259,2.02202867 8.22695259,1.01466935 10.712234,1.01466935 L10.712234,1.01466935 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M29.1646149,28.4853306 L29.1646149,69.4853306 L27.0549084,69.4850493 C26.9296179,69.4826944 26.6295085,69.4606645 26.4163708,69.2548765 C26.2882481,69.1311718 26.212234,68.9348269 26.212234,68.6519973 L26.212234,68.6519973 L26.212234,29.318664 C26.212234,29.0358344 26.2882481,28.8394895 26.4163708,28.7157848 C26.6561507,28.4842733 27.0060041,28.4853306 27.0931863,28.4853306 L29.1646149,28.4853306 Z M35.1599393,28.4853306 L35.1599393,69.4853306 L32.2645286,69.4853306 L32.2645286,28.4853306 L35.1599393,28.4853306 Z M40.3593475,28.4857507 C40.4733953,28.488728 40.76682,28.5048146 40.9853552,28.6949451 C41.1008215,28.7954034 41.1819999,28.9513684 41.2051776,29.174832 L41.2051776,29.174832 L41.212234,68.6519973 C41.212234,68.9309833 41.1385277,69.1258369 41.0132803,69.2498094 C40.7943074,69.4665535 40.478047,69.4818787 40.3594139,69.4849106 L40.3594139,69.4849106 L38.259853,69.4852498 L38.259853,28.4854114 Z" id="rows_two" stroke="#000000" fill-rule="nonzero" transform="translate(33.712234, 48.985331) rotate(-90.000000) translate(-33.712234, -48.985331) "></path>
            <g id="vertical_equal" transform="translate(30.000000, 29.380570)" stroke="#000000" stroke-linecap="square">
                <line x1="2.21223397" y1="0.514669353" x2="2.21223397" y2="7.58573716" id="Line"></line>
                <line x1="5.21223397" y1="0.514669353" x2="5.21223397" y2="7.58573716" id="Line"></line>
            </g>
            <path d="M27.0931863,-2.98533065 C26.8836625,-2.98533065 25.712234,-2.93533065 25.712234,-1.65199731 L25.712234,37.681336 C25.712234,38.9646694 26.8836625,39.0146694 27.0931863,39.0146694 L29.6646149,39.0146694 L29.6646149,-2.98533065 L27.0931863,-2.98533065 Z M35.6721771,-2.98533065 L35.6721771,39.0146694 L31.7522908,39.0146694 L31.7522908,39.0146694 L31.7522908,-2.98533065 L31.7522908,-2.98533065 L35.6721771,-2.98533065 Z M40.3656149,-2.9849096 C40.6448786,-2.97823698 41.712234,-2.87699731 41.712234,-1.65199731 L41.712234,-1.65199731 L41.712234,37.681336 C41.712234,38.9646694 40.5408054,39.0146694 40.3312816,39.0146694 L40.3312816,39.0146694 L37.759853,39.0146694 L37.759853,-2.98533065 Z" id="rows_one" fill="#000000" fill-rule="nonzero" transform="translate(33.712234, 18.014669) rotate(-90.000000) translate(-33.712234, -18.014669) "></path>
        </g>
    </g>
</svg>""",
    "col_count_match": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_count_match</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_count_match" transform="translate(0.000000, 0.275862)">
            <path d="M56.712234,1.01466935 C59.1975153,1.01466935 61.4475153,2.02202867 63.076195,3.65070832 C64.7048747,5.27938798 65.712234,7.52938798 65.712234,10.0146694 L65.712234,10.0146694 L65.712234,65.0146694 L10.712234,65.0146694 C8.22695259,65.0146694 5.97695259,64.00731 4.34827294,62.3786304 C2.71959328,60.7499507 1.71223397,58.4999507 1.71223397,56.0146694 L1.71223397,56.0146694 L1.71223397,10.0146694 C1.71223397,7.52938798 2.71959328,5.27938798 4.34827294,3.65070832 C5.97695259,2.02202867 8.22695259,1.01466935 10.712234,1.01466935 L10.712234,1.01466935 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M44.6352762,13.0146694 L44.6352762,54.0145886 L42.5357154,54.0142493 C42.4170822,54.0112174 42.1008219,53.9958922 41.8818489,53.7791481 C41.7566015,53.6551756 41.6828953,53.460322 41.6828953,53.181336 L41.6828953,53.181336 L41.6828953,13.8480027 C41.6828953,13.5651731 41.7589094,13.3688282 41.8870321,13.2451235 C42.126812,13.013612 42.4766654,13.0146694 42.5638476,13.0146694 L44.6352762,13.0146694 Z M50.6306006,13.0146694 L50.6306006,54.0146694 L47.7351899,54.0146694 L47.7351899,13.0146694 L50.6306006,13.0146694 Z M53.7305143,13.0147502 L55.8300088,13.0150894 C55.9440566,13.0180667 56.2374813,13.0341534 56.4560165,13.2242838 C56.5714828,13.3247421 56.6526612,13.4807072 56.6758389,13.7041707 L56.6758389,13.7041707 L56.6828953,53.181336 C56.6828953,53.460322 56.609189,53.6551756 56.4839416,53.7791481 C56.2649687,53.9958922 55.9487083,54.0112174 55.8300752,54.0142493 L55.8300752,54.0142493 L53.7305143,54.0145886 L53.7305143,13.0147502 Z" id="rows_two" stroke="#000000" fill-rule="nonzero" transform="translate(49.182895, 33.514669) rotate(-180.000000) translate(-49.182895, -33.514669) "></path>
            <g id="vertical_equal" transform="translate(34.135195, 33.726903) rotate(-90.000000) translate(-34.135195, -33.726903) translate(30.635195, 29.226903)" stroke="#000000" stroke-linecap="square">
                <line x1="2.21223397" y1="0.514669353" x2="2.21223397" y2="7.58573716" id="Line"></line>
                <line x1="5.21223397" y1="0.514669353" x2="5.21223397" y2="7.58573716" id="Line"></line>
            </g>
            <path d="M11.5931863,12.5146694 C11.3836625,12.5146694 10.212234,12.5646694 10.212234,13.8480027 L10.212234,53.181336 C10.212234,54.4646694 11.3836625,54.5146694 11.5931863,54.5146694 L14.1646149,54.5146694 L14.1646149,12.5146694 L11.5931863,12.5146694 Z M20.1721771,12.5146694 L20.1721771,54.5146694 L16.2522908,54.5146694 L16.2522908,54.5146694 L16.2522908,12.5146694 L16.2522908,12.5146694 L20.1721771,12.5146694 Z M24.8656149,12.5150904 C25.1448786,12.521763 26.212234,12.6230027 26.212234,13.8480027 L26.212234,13.8480027 L26.212234,53.181336 C26.212234,54.4646694 25.0408054,54.5146694 24.8312816,54.5146694 L24.8312816,54.5146694 L22.259853,54.5146694 L22.259853,12.5146694 Z" id="rows_one" fill="#000000" fill-rule="nonzero" transform="translate(18.212234, 33.514669) rotate(-180.000000) translate(-18.212234, -33.514669) "></path>
        </g>
    </g>
</svg>""",
    "tbl_match": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>tbl_match</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="tbl_match" transform="translate(0.000000, 0.758621)">
            <path d="M56.712234,1.01466935 C59.1975153,1.01466935 61.4475153,2.02202867 63.076195,3.65070832 C64.7048747,5.27938798 65.712234,7.52938798 65.712234,10.0146694 L65.712234,10.0146694 L65.712234,65.0146694 L10.712234,65.0146694 C8.22695259,65.0146694 5.97695259,64.00731 4.34827294,62.3786304 C2.71959328,60.7499507 1.71223397,58.4999507 1.71223397,56.0146694 L1.71223397,56.0146694 L1.71223397,10.0146694 C1.71223397,7.52938798 2.71959328,5.27938798 4.34827294,3.65070832 C5.97695259,2.02202867 8.22695259,1.01466935 10.712234,1.01466935 L10.712234,1.01466935 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <g id="equal" transform="translate(46.026611, 20.710122) rotate(-90.000000) translate(-46.026611, -20.710122) translate(42.526611, 16.210122)" stroke="#000000" stroke-linecap="square">
                <line x1="2.21223397" y1="0.514669353" x2="2.21223397" y2="7.58573716" id="Line"></line>
                <line x1="5.21223397" y1="0.514669353" x2="5.21223397" y2="7.58573716" id="Line"></line>
            </g>
            <g id="equal" transform="translate(21.397857, 45.319217) rotate(-90.000000) translate(-21.397857, -45.319217) translate(17.897857, 40.819217)" stroke="#000000" stroke-linecap="square">
                <line x1="2.21223397" y1="0.514669353" x2="2.21223397" y2="7.58573716" id="Line"></line>
                <line x1="5.21223397" y1="0.514669353" x2="5.21223397" y2="7.58573716" id="Line"></line>
            </g>
            <path d="M21.3882419,7.77869783 C21.3584298,7.77935177 21.328704,7.78216367 21.2992996,7.78711914 L9.09014824,7.78711914 C8.75029431,7.78715312 8.47479677,8.06265067 8.47476279,8.4025046 L8.47476279,16.2991498 C8.46377874,16.3656061 8.46377874,16.4334149 8.47476279,16.4998713 L8.47476279,24.9145461 C8.46377874,24.9810025 8.46377874,25.0488112 8.47476279,25.1152676 L8.47476279,33.0179226 C8.47479677,33.3577766 8.75029431,33.6332741 9.09014824,33.6333081 L21.2944916,33.6333081 C21.3609479,33.6442921 21.4287567,33.6442921 21.4952131,33.6333081 L33.7055663,33.6333081 C34.0454202,33.6332741 34.3209178,33.3577766 34.3209517,33.0179226 L34.3209517,25.1212775 C34.3319358,25.0548211 34.3319358,24.9870123 34.3209517,24.920556 L34.3209517,16.5058811 C34.3319358,16.4394248 34.3319358,16.371616 34.3209517,16.3051596 L34.3209517,8.4025046 C34.3209178,8.06265067 34.0454202,7.78715312 33.7055663,7.78711914 L21.4928094,7.78711914 C21.4582565,7.78134369 21.4232736,7.77869783 21.3882419,7.77869783 Z M9.70553369,9.01789005 L20.7824718,9.01789005 L20.7824718,15.78713 L9.70553369,15.78713 L9.70553369,9.01789005 Z M22.0132427,9.01789005 L33.0901808,9.01789005 L33.0901808,15.78713 L22.0132427,15.78713 L22.0132427,9.01789005 Z M9.70553369,17.0179009 L20.7824718,17.0179009 L20.7824718,24.4025263 L9.70553369,24.4025263 L9.70553369,17.0179009 Z M22.0132427,17.0179009 L33.0901808,17.0179009 L33.0901808,24.4025263 L22.0132427,24.4025263 L22.0132427,17.0179009 Z M9.70553369,25.6332972 L20.7824718,25.6332972 L20.7824718,32.4025372 L9.70553369,32.4025372 L9.70553369,25.6332972 Z M22.0132427,25.6332972 L33.0901808,25.6332972 L33.0901808,32.4025372 L22.0132427,32.4025372 L22.0132427,25.6332972 Z" id="table" fill="#000000" fill-rule="nonzero"></path>
            <path d="M46.0169953,32.3877926 C45.9871832,32.3884465 45.9574575,32.3912584 45.928053,32.3962139 L33.7189016,32.3962139 C33.3790477,32.3962479 33.1035502,32.6717454 33.1035162,33.0115993 L33.1035162,40.9082445 C33.0925322,40.9747009 33.0925322,41.0425097 33.1035162,41.108966 L33.1035162,49.5236408 C33.0925322,49.5900972 33.0925322,49.657906 33.1035162,49.7243624 L33.1035162,57.6270174 C33.1035502,57.9668713 33.3790477,58.2423689 33.7189016,58.2424028 L45.923245,58.2424028 C45.9897014,58.2533869 46.0575101,58.2533869 46.1239665,58.2424028 L58.3343197,58.2424028 C58.6741736,58.2423689 58.9496712,57.9668713 58.9497051,57.6270174 L58.9497051,49.7303722 C58.9606892,49.6639158 58.9606892,49.5961071 58.9497051,49.5296507 L58.9497051,41.1149759 C58.9606892,41.0485195 58.9606892,40.9807107 58.9497051,40.9142544 L58.9497051,33.0115993 C58.9496712,32.6717454 58.6741736,32.3962479 58.3343197,32.3962139 L46.1215628,32.3962139 C46.0870099,32.3904384 46.052027,32.3877926 46.0169953,32.3877926 Z M34.3342871,33.6269848 L45.4112252,33.6269848 L45.4112252,40.3962248 L34.3342871,40.3962248 L34.3342871,33.6269848 Z M46.6419961,33.6269848 L57.7189342,33.6269848 L57.7189342,40.3962248 L46.6419961,40.3962248 L46.6419961,33.6269848 Z M34.3342871,41.6269957 L45.4112252,41.6269957 L45.4112252,49.0116211 L34.3342871,49.0116211 L34.3342871,41.6269957 Z M46.6419961,41.6269957 L57.7189342,41.6269957 L57.7189342,49.0116211 L46.6419961,49.0116211 L46.6419961,41.6269957 Z M34.3342871,50.242392 L45.4112252,50.242392 L45.4112252,57.0116319 L34.3342871,57.0116319 L34.3342871,50.242392 Z M46.6419961,50.242392 L57.7189342,50.242392 L57.7189342,57.0116319 L46.6419961,57.0116319 L46.6419961,50.242392 Z" id="table" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "col_vals_expr": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="66px" viewBox="0 0 67 66" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>col_vals_expr</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="col_vals_expr" transform="translate(0.000000, 0.000000)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M22.1682701,14.6021863 L22.1682701,17.9472433 L25.5133271,17.9472433 C26.8395904,17.9472433 28.0482531,18.7508414 28.544785,19.9856375 L31.2103774,26.623485 L20.4957415,51.3978137 L24.1543977,51.3978137 L32.9351724,31.0138724 L39.4684869,47.2687589 C40.4680837,49.7644859 42.8984767,51.3978137 45.5836693,51.3978137 L48.9287264,51.3978137 L48.9287264,48.0527567 L45.5836693,48.0527567 C44.2508732,48.0527567 43.0487433,47.2491586 42.5522114,46.0143625 L31.6285095,18.7312411 C30.6289128,16.2355157 28.1985198,14.6021863 25.5133271,14.6021863 L22.1682701,14.6021863 Z" id="lambda" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "conjointly": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>conjointly</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="conjointly" transform="translate(0.000000, 0.241379)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <path d="M51.8485976,12 L15.5758703,12 C13.9986329,12 12.712234,13.2863989 12.712234,14.8636364 L12.712234,51.1363636 C12.712234,52.7136011 13.9986329,54 15.5758703,54 L51.8485976,54 C53.4258351,54 54.712234,52.7136011 54.712234,51.1363636 L54.712234,14.8636364 C54.712234,13.2863989 53.4258351,12 51.8485976,12 Z M37.072234,44 L20.272234,44 L20.272234,42 L37.072234,42 L37.072234,44 Z M37.072234,34 L20.272234,34 L20.272234,32 L37.072234,32 L37.072234,34 Z M37.072234,24 L20.272234,24 L20.272234,22 L37.072234,22 L37.072234,24 Z M47.9233279,41.773438 L45.5706719,45.773438 C45.4427029,45.996094 45.239265,46.148438 45.0095779,46.1875 C44.9702029,46.195313 44.9275469,46.199219 44.88489,46.199219 C44.70114,46.199219 44.5206719,46.128906 44.373015,45.992188 L42.1877029,43.992188 C41.8202029,43.65625 41.7512969,43.027344 42.033484,42.589844 C42.3156719,42.152344 42.8439529,42.070313 43.2114529,42.40625 L44.697859,43.769531 L46.548484,40.625 C46.814265,40.171875 47.335984,40.0625 47.716609,40.378906 C48.097234,40.695313 48.189109,41.320313 47.9233279,41.773438 Z M47.9233279,31.773438 L45.5706719,35.773438 C45.4427029,35.996094 45.239265,36.148438 45.0095779,36.1875 C44.9702029,36.195313 44.9275469,36.199219 44.88489,36.199219 C44.70114,36.199219 44.5206719,36.128906 44.373015,35.992188 L42.1877029,33.992188 C41.8202029,33.65625 41.7512969,33.027344 42.033484,32.589844 C42.3156719,32.152344 42.8439529,32.070313 43.2114529,32.40625 L44.697859,33.769531 L46.548484,30.628906 C46.814265,30.175781 47.335984,30.0625 47.716609,30.382813 C48.097234,30.699219 48.189109,31.320313 47.9233279,31.773438 Z M47.9233279,21.773438 L45.5706719,25.773438 C45.4427029,25.996094 45.239265,26.148438 45.0095779,26.1875 C44.9702029,26.195313 44.9275469,26.199219 44.88489,26.199219 C44.70114,26.199219 44.5206719,26.128906 44.373015,25.992188 L42.1877029,23.992188 C41.8202029,23.65625 41.7512969,23.027344 42.033484,22.589844 C42.3156719,22.152344 42.8439529,22.070313 43.2114529,22.40625 L44.697859,23.769531 L46.548484,20.625 C46.814265,20.171875 47.335984,20.0625 47.716609,20.378906 C48.097234,20.699219 48.189109,21.320313 47.9233279,21.773438 Z" id="conjoint" fill="#000000" fill-rule="nonzero"></path>
        </g>
    </g>
</svg>""",
    "specially": """<?xml version="1.0" encoding="UTF-8"?>
<svg width="67px" height="67px" viewBox="0 0 67 67" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>specially</title>
    <g id="All-Icons" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="specially" transform="translate(0.000000, 0.206897)">
            <path d="M56.712234,1 C59.1975153,1 61.4475153,2.00735931 63.076195,3.63603897 C64.7048747,5.26471863 65.712234,7.51471863 65.712234,10 L65.712234,10 L65.712234,65 L10.712234,65 C8.22695259,65 5.97695259,63.9926407 4.34827294,62.363961 C2.71959328,60.7352814 1.71223397,58.4852814 1.71223397,56 L1.71223397,56 L1.71223397,10 C1.71223397,7.51471863 2.71959328,5.26471863 4.34827294,3.63603897 C5.97695259,2.00735931 8.22695259,1 10.712234,1 L10.712234,1 Z" id="rectangle" stroke="#000000" stroke-width="2" fill="#FFFFFF"></path>
            <g id="star" transform="translate(8.500000, 8.500000)" fill="#000000" fill-rule="nonzero">
                <path d="M25,0 C24.5874484,0 24.2174517,0.254002847 24.068359,0.6386719 L17.902344,16.535156 L0.94921875,17.400391 C0.536124409,17.4213013 0.17852458,17.6943523 0.0495446395,18.0873515 C-0.0794353012,18.4803507 0.046820452,18.9122002 0.3671875,19.173828 L13.568359,29.966797 L9.2324219,46.34375 C9.12646963,46.7428009 9.27663058,47.1659433 9.61042698,47.4089402 C9.94422338,47.651937 10.3930345,47.664834 10.740234,47.441406 L25,38.289062 L39.259766,47.441406 C39.6069655,47.6648339 40.0557766,47.6519369 40.3895729,47.4089401 C40.7233693,47.1659432 40.8735302,46.7428009 40.767578,46.34375 L36.431641,29.966797 L49.632812,19.173828 C49.953179,18.9122002 50.0794348,18.4803507 49.9504549,18.0873516 C49.821475,17.6943524 49.4638753,17.4213014 49.050781,17.400391 L32.097656,16.535156 L25.931641,0.6386719 C25.7825483,0.254002847 25.4125516,0 25,0 Z M25,3.7636719 L30.466797,17.861328 C30.609689,18.2291416 30.9554962,18.4785515 31.349609,18.498047 L46.359375,19.265625 L34.667969,28.826172 C34.3646054,29.0742114 34.2340493,29.4765679 34.333984,29.855469 L38.175781,44.369141 L25.541016,36.257812 C25.2114789,36.0458536 24.7885211,36.0458536 24.458984,36.257812 L11.824219,44.369141 L15.666016,29.855469 C15.7659507,29.4765679 15.6353946,29.0742114 15.332031,28.826172 L3.640625,19.265625 L18.650391,18.498047 C19.0445038,18.4785515 19.390311,18.2291416 19.533203,17.861328 L25,3.7636719 Z" id="Shape"></path>
            </g>
        </g>
    </g>
</svg>""",
}


SVG_ICONS_FOR_TBL_STATUS = {
    "unchanged": """<svg width="25px" height="25px" viewBox="0 0 25 25" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="vertical-align: middle;">
    <g id="unchanged" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="unchanged" transform="translate(0.500000, 0.570147)">
            <rect id="Rectangle" x="0.125132506" y="0" width="23.749735" height="23.7894737"></rect>
            <path d="M5.80375046,8.18194736 C3.77191832,8.18194736 2.11875046,9.83495328 2.11875046,11.8669474 C2.11875046,13.8989414 3.77191832,15.5519474 5.80375046,15.5519474 C7.8355826,15.5519474 9.48875046,13.8989414 9.48875046,11.8669474 C9.48875046,9.83495328 7.83552863,8.18194736 5.80375046,8.18194736 Z M5.80375046,14.814915 C4.17821997,14.814915 2.85578285,13.4924778 2.85578285,11.8669474 C2.85578285,10.2414169 4.17821997,8.91897975 5.80375046,8.91897975 C7.42928095,8.91897975 8.75171807,10.2414169 8.75171807,11.8669474 C8.75171807,13.4924778 7.42928095,14.814915 5.80375046,14.814915 Z" id="Shape" fill="#000000" fill-rule="nonzero"></path>
            <path d="M13.9638189,8.699335 C13.9364621,8.70430925 13.9091059,8.71176968 13.8842359,8.71923074 C13.7822704,8.73663967 13.6877654,8.77643115 13.6056956,8.83860518 L10.2433156,11.3852598 C10.0766886,11.5046343 9.97720993,11.6986181 9.97720993,11.9025491 C9.97720993,12.1064807 10.0766886,12.3004639 10.2433156,12.4198383 L13.6056956,14.966493 C13.891697,15.1803725 14.2970729,15.1231721 14.5109517,14.8371707 C14.7248313,14.5511692 14.6676309,14.145794 14.3816294,13.9319145 L12.5313257,12.5392127 L21.8812495,12.5392127 L21.8812495,11.2658854 L12.5313257,11.2658854 L14.3816294,9.87318364 C14.6377872,9.71650453 14.7497006,9.40066014 14.6477351,9.11714553 C14.5482564,8.83363156 14.262255,8.65954352 13.9638189,8.699335 Z" id="arrow" fill="#000000" transform="translate(15.929230, 11.894737) rotate(-180.000000) translate(-15.929230, -11.894737) "></path>
        </g>
    </g>
</svg>""",
    "modified": """<svg width="24px" height="25px" viewBox="0 0 24 25" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="vertical-align: middle;">
    <g id="modified" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="modified" transform="translate(0.000000, 0.570147)">
            <rect id="Rectangle" x="0.125132506" y="0" width="23.749735" height="23.7894737"></rect>
            <rect id="Rectangle" fill="#000000" x="15.2215626" y="14.8025885" width="6.34089821" height="6.34089821"></rect>
            <rect id="Rectangle" fill="#9A7CB4" x="16.1264309" y="15.7074568" width="4.53116158" height="4.53116158"></rect>
            <path d="M4.40094816,14.3498527 C2.36911602,14.3498527 0.715948163,16.0028586 0.715948163,18.0348527 C0.715948163,20.0668467 2.36911602,21.7198527 4.40094816,21.7198527 C6.43278031,21.7198527 8.08594816,20.0668467 8.08594816,18.0348527 C8.08594816,16.0028586 6.43272633,14.3498527 4.40094816,14.3498527 Z M4.40094816,20.9828203 C2.77541767,20.9828203 1.45298055,19.6603831 1.45298055,18.0348527 C1.45298055,16.4093222 2.77541767,15.086885 4.40094816,15.086885 C6.02647865,15.086885 7.34891578,16.4093222 7.34891578,18.0348527 C7.34891578,19.6603831 6.02647865,20.9828203 4.40094816,20.9828203 Z" id="Shape" fill="#000000" fill-rule="nonzero"></path>
            <path d="M18.4333601,-0.640287588 L18.4333601,14.5882952 L9.52131131,14.5882952 L11.3716151,15.980997 C11.6576165,16.1948765 11.7148169,16.6002518 11.5009374,16.8862532 C11.2870585,17.1722546 10.8816827,17.229455 10.5956813,17.0155755 L10.5956813,17.0155755 L7.23330126,14.4689208 C7.06667429,14.3495464 6.96719559,14.1555632 6.96719559,13.9516316 C6.96719559,13.7477006 7.06667429,13.5537168 7.23330126,13.4343424 L7.23330126,13.4343424 L10.5956813,10.8876877 C10.677751,10.8255137 10.7722561,10.7857222 10.8742216,10.7683132 C10.8990916,10.7608522 10.9264478,10.7533918 10.9538046,10.7484175 C11.2522407,10.708626 11.5382421,10.8827141 11.6377208,11.166228 C11.7396863,11.4497426 11.6277729,11.765587 11.3716151,11.9222661 L11.3716151,11.9222661 L9.52131131,13.3149679 L17.1593533,13.3147124 L17.1594577,0.633614828 L6.9283533,0.633614828 L6.9283533,-0.640287588 L18.4333601,-0.640287588 Z" id="arrow" fill="#000000" transform="translate(12.680857, 8.252286) rotate(-90.000000) translate(-12.680857, -8.252286) "></path>
        </g>
    </g>
</svg>""",
    "segmented": """<svg width="24px" height="25px" viewBox="0 0 24 25" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="vertical-align: middle;">
    <g id="segmented" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="segmented" transform="translate(0.000000, 0.570147)">
            <rect id="Rectangle" x="0.125132506" y="0" width="23.749735" height="23.7894737"></rect>
            <path d="M18.4051095,14.3518889 C20.4405142,14.3518889 22.0905358,16.0019105 22.0905358,18.0373152 C22.0905358,20.07272 20.4405142,21.7227416 18.4051095,21.7227416 C16.3697047,21.7227416 14.7196831,20.07272 14.7196831,18.0373152 C14.7196831,16.0019105 16.3697047,14.3518889 18.4051095,14.3518889 Z M18.4051095,14.9787621 C16.7159172,14.9787621 15.3465563,16.348123 15.3465563,18.0373152 C15.3465563,18.7697385 15.604001,19.4420322 16.0333357,19.9686419 L20.3364361,15.6655415 C19.8098264,15.2362068 19.1375327,14.9787621 18.4051095,14.9787621 Z" id="Shape" fill="#000000" fill-rule="nonzero"></path>
            <path d="M4.40094816,14.3498527 C2.36911602,14.3498527 0.715948163,16.0028586 0.715948163,18.0348527 C0.715948163,20.0668467 2.36911602,21.7198527 4.40094816,21.7198527 C6.43278031,21.7198527 8.08594816,20.0668467 8.08594816,18.0348527 C8.08594816,16.0028586 6.43272633,14.3498527 4.40094816,14.3498527 Z M4.40094816,20.9828203 C2.77541767,20.9828203 1.45298055,19.6603831 1.45298055,18.0348527 C1.45298055,16.4093222 2.77541767,15.086885 4.40094816,15.086885 C6.02647865,15.086885 7.34891578,16.4093222 7.34891578,18.0348527 C7.34891578,19.6603831 6.02647865,20.9828203 4.40094816,20.9828203 Z" id="Shape" fill="#000000" fill-rule="nonzero"></path>
            <path d="M18.4333601,-0.640287588 L18.4333601,14.5882952 L9.52131131,14.5882952 L11.3716151,15.980997 C11.6576165,16.1948765 11.7148169,16.6002518 11.5009374,16.8862532 C11.2870585,17.1722546 10.8816827,17.229455 10.5956813,17.0155755 L10.5956813,17.0155755 L7.23330126,14.4689208 C7.06667429,14.3495464 6.96719559,14.1555632 6.96719559,13.9516316 C6.96719559,13.7477006 7.06667429,13.5537168 7.23330126,13.4343424 L7.23330126,13.4343424 L10.5956813,10.8876877 C10.677751,10.8255137 10.7722561,10.7857222 10.8742216,10.7683132 C10.8990916,10.7608522 10.9264478,10.7533918 10.9538046,10.7484175 C11.2522407,10.708626 11.5382421,10.8827141 11.6377208,11.166228 C11.7396863,11.4497426 11.6277729,11.765587 11.3716151,11.9222661 L11.3716151,11.9222661 L9.52131131,13.3149679 L17.1593533,13.3147124 L17.1594577,0.633614828 L6.9283533,0.633614828 L6.9283533,-0.640287588 L18.4333601,-0.640287588 Z" id="arrow" fill="#000000" transform="translate(12.680857, 8.252286) rotate(-90.000000) translate(-12.680857, -8.252286) "></path>
        </g>
    </g>
</svg>""",
}


SVG_ICONS_FOR_DATA_TYPES = {
    "numeric": """<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;">
    <title>numeric</title>
    <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="numeric" transform="translate(4.000000, 4.339342)">
            <rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"></rect>
            <g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero">
                <polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"></polygon>
            </g>
        </g>
    </g>
</svg>""",
    "string": """<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;">
    <title>string</title>
    <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="string" transform="translate(4.000000, 4.500000)">
            <rect id="square" stroke="#9A8700" stroke-width="4" fill="#CFB600" x="2" y="2" width="168" height="168" rx="8"></rect>
            <g id="S" transform="translate(49.516000, 38.280000)" fill="#FFFFFF" fill-rule="nonzero">
                <path d="M34.98,95.304 C26.884,95.304 20.02,93.918 14.388,91.146 C8.756,88.374 3.96,84.788 0,80.388 L13.2,67.056 C19.448,74.096 27.148,77.616 36.3,77.616 C41.228,77.616 44.88,76.604 47.256,74.58 C49.632,72.556 50.82,69.872 50.82,66.528 C50.82,63.976 50.116,61.842 48.708,60.126 C47.3,58.41 44.44,57.244 40.128,56.628 L31.02,55.44 C21.252,54.208 14.102,51.26 9.57,46.596 C5.038,41.932 2.772,35.728 2.772,27.984 C2.772,23.848 3.564,20.064 5.148,16.632 C6.732,13.2 8.998,10.252 11.946,7.788 C14.894,5.324 18.502,3.41 22.77,2.046 C27.038,0.682 31.9,0 37.356,0 C44.308,0 50.424,1.1 55.704,3.3 C60.984,5.5 65.516,8.712 69.3,12.936 L55.968,26.4 C53.768,23.848 51.106,21.758 47.982,20.13 C44.858,18.502 40.92,17.688 36.168,17.688 C31.68,17.688 28.336,18.48 26.136,20.064 C23.936,21.648 22.836,23.848 22.836,26.664 C22.836,29.832 23.694,32.12 25.41,33.528 C27.126,34.936 29.92,35.948 33.792,36.564 L42.9,38.016 C52.404,39.512 59.444,42.438 64.02,46.794 C68.596,51.15 70.884,57.332 70.884,65.34 C70.884,69.74 70.092,73.788 68.508,77.484 C66.924,81.18 64.614,84.348 61.578,86.988 C58.542,89.628 54.78,91.674 50.292,93.126 C45.804,94.578 40.7,95.304 34.98,95.304 Z"></path>
            </g>
        </g>
    </g>
</svg>""",
    "date": """<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;">
    <title>date</title>
    <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="date" transform="translate(4.000000, 4.339342)">
            <rect id="square" stroke="#007D00" stroke-width="4" fill="#2CCF00" x="2" y="2" width="168" height="168" rx="8"></rect>
            <g id="D" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero">
                <path d="M0,0 L34.716,0 C40.788,0 46.332,0.968 51.348,2.904 C56.364,4.84 60.632,7.722 64.152,11.55 C67.672,15.378 70.4,20.174 72.336,25.938 C74.272,31.702 75.24,38.412 75.24,46.068 C75.24,53.724 74.272,60.434 72.336,66.198 C70.4,71.962 67.672,76.758 64.152,80.586 C60.632,84.414 56.364,87.296 51.348,89.232 C46.332,91.168 40.788,92.136 34.716,92.136 L0,92.136 L0,0 Z M34.716,74.316 C40.7,74.316 45.408,72.644 48.84,69.3 C52.272,65.956 53.988,60.588 53.988,53.196 L53.988,38.94 C53.988,31.548 52.272,26.18 48.84,22.836 C45.408,19.492 40.7,17.82 34.716,17.82 L20.064,17.82 L20.064,74.316 L34.716,74.316 Z"></path>
            </g>
        </g>
    </g>
</svg>""",
    "boolean": """<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;">
    <title>boolean</title>
    <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="boolean" transform="translate(4.000000, 4.339342)">
            <rect id="square" stroke="#9C3E00" stroke-width="4" fill="#CF6100" x="2" y="2" width="168" height="168" rx="8"></rect>
            <g id="T" transform="translate(32.322000, 29.972000)" fill="#FFFFFF" fill-rule="nonzero">
                <polygon points="29.326 11.61 29.326 60.028 16.254 60.028 16.254 11.61 0 11.61 0 0 45.58 0 45.58 11.61"></polygon>
            </g>
            <g id="F" transform="translate(99.122000, 81.972000)" fill="#FFFFFF" fill-rule="nonzero">
                <polygon points="0 60.028 0 0 39.818 0 39.818 11.61 13.072 11.61 13.072 23.908 35.862 23.908 35.862 35.432 13.072 35.432 13.072 60.028"></polygon>
            </g>
            <rect id="line" stroke="#FFFFFF" fill="#FFFFFF" transform="translate(82.022437, 84.377940) rotate(-315.000000) translate(-82.022437, -84.377940) " x="78.0224369" y="25.37794" width="8" height="118"></rect>
        </g>
    </g>
</svg>""",
    "object": """<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;">
    <title>object</title>
    <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="object" transform="translate(4.000000, 4.339342)">
            <rect id="square" stroke="#131313" stroke-width="4" fill="#646464" x="2" y="2" width="168" height="168" rx="8"></rect>
            <g id="O" transform="translate(46.100000, 38.280000)" fill="#FFFFFF" fill-rule="nonzero">
                <path d="M40.524,95.304 C34.452,95.304 28.908,94.292 23.892,92.268 C18.876,90.244 14.608,87.23 11.088,83.226 C7.568,79.222 4.84,74.25 2.904,68.31 C0.968,62.37 0,55.484 0,47.652 C0,39.82 0.968,32.934 2.904,26.994 C4.84,21.054 7.568,16.082 11.088,12.078 C14.608,8.074 18.876,5.06 23.892,3.036 C28.908,1.012 34.452,0 40.524,0 C46.596,0 52.14,1.012 57.156,3.036 C62.172,5.06 66.44,8.074 69.96,12.078 C73.48,16.082 76.208,21.054 78.144,26.994 C80.08,32.934 81.048,39.82 81.048,47.652 C81.048,55.484 80.08,62.37 78.144,68.31 C76.208,74.25 73.48,79.222 69.96,83.226 C66.44,87.23 62.172,90.244 57.156,92.268 C52.14,94.292 46.596,95.304 40.524,95.304 Z M40.524,77.484 C46.596,77.484 51.326,75.504 54.714,71.544 C58.102,67.584 59.796,61.908 59.796,54.516 L59.796,40.788 C59.796,33.396 58.102,27.72 54.714,23.76 C51.326,19.8 46.596,17.82 40.524,17.82 C34.452,17.82 29.722,19.8 26.334,23.76 C22.946,27.72 21.252,33.396 21.252,40.788 L21.252,54.516 C21.252,61.908 22.946,67.584 26.334,71.544 C29.722,75.504 34.452,77.484 40.524,77.484 Z"></path>
            </g>
        </g>
    </g>
</svg>""",
}
