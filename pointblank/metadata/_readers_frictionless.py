from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pointblank.metadata._types import (
    Codelist,
    CodelistEntry,
    MetadataImport,
    MetadataPackage,
    MissingValueCode,
    VariableMetadata,
)

__all__ = [
    "_read_frictionless_metadata",
    "_read_csvw_metadata",
]

# Mapping from Frictionless field types to Pointblank dtype strings
_FRICTIONLESS_TYPE_MAP: dict[str, str] = {
    "integer": "Int64",
    "number": "Float64",
    "string": "String",
    "boolean": "Boolean",
    "date": "Date",
    "datetime": "Datetime",
    "time": "Time",
    "duration": "Duration",
    "year": "Int64",
    "yearmonth": "String",
    "object": "String",
    "array": "String",
    "geopoint": "String",
    "geojson": "String",
    "any": "String",
}

# Mapping from CSVW datatype base values to Pointblank dtype strings
_CSVW_DATATYPE_MAP: dict[str, str] = {
    "integer": "Int64",
    "int": "Int64",
    "long": "Int64",
    "short": "Int64",
    "byte": "Int64",
    "nonNegativeInteger": "Int64",
    "positiveInteger": "Int64",
    "unsignedInt": "Int64",
    "unsignedLong": "Int64",
    "unsignedShort": "Int64",
    "float": "Float64",
    "double": "Float64",
    "decimal": "Float64",
    "number": "Float64",
    "string": "String",
    "normalizedString": "String",
    "token": "String",
    "anyURI": "String",
    "boolean": "Boolean",
    "date": "Date",
    "dateTime": "Datetime",
    "datetime": "Datetime",
    "time": "Time",
    "duration": "Duration",
    "gDay": "String",
    "gMonth": "String",
    "gYear": "Int64",
    "gYearMonth": "String",
    "gMonthDay": "String",
    "hexBinary": "String",
    "base64Binary": "String",
    "anyAtomicType": "String",
    "json": "String",
    "xml": "String",
    "html": "String",
}


