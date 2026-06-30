try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:  # pragma: no cover
    __version__ = version("pointblank")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Import objects from the module
from pointblank.actions import send_slack_notification
from pointblank.adapters import (
    ContractAdapter,
    ContractImport,
    export_contract,
    import_contract,
    list_adapters,
    register_adapter,
)
from pointblank.assistant import assistant
from pointblank.column import (
    col,
    contains,
    ends_with,
    everything,
    expr_col,
    first_n,
    last_n,
    matches,
    ref,
    starts_with,
)
from pointblank.contract import Contract, Step
from pointblank.datascan import DataScan, col_summary_tbl
from pointblank.draft import DraftValidation
from pointblank.field import (
    BoolField,
    DateField,
    DatetimeField,
    DurationField,
    Field,
    FloatField,
    IntField,
    StringField,
    TimeField,
    bool_field,
    date_field,
    datetime_field,
    duration_field,
    float_field,
    int_field,
    profile_fields,
    string_field,
    time_field,
)
from pointblank.generate.base import GeneratorConfig
from pointblank.inspect import has_columns, has_rows
from pointblank.integrations.otel import emit_otel
from pointblank.missing import MissingSpec
from pointblank.metadata import (
    ADaMDatasetTemplate,
    ADaMVariableSpec,
    Codelist,
    CodelistEntry,
    MetadataImport,
    MetadataPackage,
    MissingValueCode,
    SDTMDomainTemplate,
    SDTMVariableSpec,
    VariableMetadata,
    adam_to_metadata,
    export_metadata,
    get_adam_dataset,
    get_sdtm_domain,
    import_metadata,
    list_adam_datasets,
    list_sdtm_domains,
    load_metadata_example,
    sdtm_to_metadata,
    validate_adam,
    validate_adam_structure,
    validate_sdtm,
    validate_sdtm_structure,
)
from pointblank.pipeline import Pipeline, PipelineResult
from pointblank.schema import Schema, generate_dataset, schema_from_tbl
from pointblank.segments import seg_group
from pointblank.thresholds import Actions, FinalActions, Thresholds
from pointblank.validate import (
    Validate,
    config,
    connect_to_table,
    get_action_metadata,
    get_column_count,
    get_data_path,
    get_row_count,
    get_validation_summary,
    load_dataset,
    missing_vals_tbl,
    preview,
    print_database_tables,
    read_file,
    write_file,
)
from pointblank.yaml import (
    validate_yaml,
    yaml_interrogate,
    yaml_to_python,
)

__all__ = [
    "assistant",
    "Validate",
    "Thresholds",
    "Actions",
    "FinalActions",
    "Schema",
    "Contract",
    "Step",
    "Pipeline",
    "PipelineResult",
    "DataScan",
    "DraftValidation",
    "MissingSpec",
    "col",
    "ref",
    "expr_col",
    "col_summary_tbl",
    "starts_with",
    "ends_with",
    "contains",
    "matches",
    "everything",
    "first_n",
    "last_n",
    "load_dataset",
    "write_file",
    "read_file",
    "get_data_path",
    "config",
    "connect_to_table",
    "print_database_tables",
    "preview",
    "missing_vals_tbl",
    "get_action_metadata",
    "get_validation_summary",
    "get_column_count",
    "get_row_count",
    "seg_group",
    "send_slack_notification",
    "emit_otel",
    # Data generation - Field classes
    "Field",
    "IntField",
    "FloatField",
    "StringField",
    "BoolField",
    "DateField",
    "DatetimeField",
    "TimeField",
    "DurationField",
    # Data generation - helper functions
    "int_field",
    "float_field",
    "string_field",
    "bool_field",
    "date_field",
    "datetime_field",
    "time_field",
    "duration_field",
    "profile_fields",
    # Data generation - configuration
    "GeneratorConfig",
    # Data generation - convenience function
    "generate_dataset",
    "schema_from_tbl",
    # Table inspection functions (for use with `active=`)
    "has_columns",
    "has_rows",
    # YAML functionality
    "yaml_interrogate",
    "validate_yaml",
    "yaml_to_python",
    # Contract import/export
    "ContractAdapter",
    "ContractImport",
    "import_contract",
    "export_contract",
    "list_adapters",
    "register_adapter",
    # Metadata standards import/export
    "import_metadata",
    "load_metadata_example",
    "export_metadata",
    "MetadataImport",
    "MetadataPackage",
    "VariableMetadata",
    "Codelist",
    "CodelistEntry",
    "MissingValueCode",
    # SDTM domain validation
    "SDTMDomainTemplate",
    "SDTMVariableSpec",
    "get_sdtm_domain",
    "list_sdtm_domains",
    "validate_sdtm_structure",
    "sdtm_to_metadata",
    "validate_sdtm",
    # ADaM dataset validation
    "ADaMDatasetTemplate",
    "ADaMVariableSpec",
    "get_adam_dataset",
    "list_adam_datasets",
    "validate_adam_structure",
    "adam_to_metadata",
    "validate_adam",
]
