from __future__ import annotations

from typing import Any

from pointblank.metadata._adam_templates import (
    get_adam_dataset,
)
from pointblank.metadata._types import (
    MetadataImport,
    VariableMetadata,
)

__all__ = [
    "adam_to_metadata",
    "validate_adam",
]


def adam_to_metadata(
    dataset: str,
    study_id: str | None = None,
) -> MetadataImport:
    """Convert an ADaM dataset template to a MetadataImport object.

    Parameters
    ----------
    dataset
        ADaM dataset name (e.g., `"ADSL"`, `"BDS"`, `"ADAE"`, `"ADTTE"`). This is case-insensitive.
    study_id
        Optional study identifier.

    Returns
    -------
    MetadataImport
        A MetadataImport representing the ADaM dataset template.
    """
    template = get_adam_dataset(dataset)

    variables: list[VariableMetadata] = []
    for spec in template.variables:
        dtype = "Float64" if spec.dtype == "Num" else "String"
        var = VariableMetadata(
            name=spec.name,
            label=spec.label,
            dtype=dtype,
            required=spec.required,
            max_length=spec.max_length,
            controlled_term=spec.controlled_term,
            cdisc_domain=template.name,
            cdisc_role=spec.core,
            adam_derivation=spec.source,
        )
        variables.append(var)

    return MetadataImport(
        source_format="cdisc_adam",
        source_version="IG 1.1",
        dataset_name=template.name,
        dataset_label=template.label,
        dataset_description=template.description,
        study_id=study_id,
        domain=template.name,
        variables=variables,
    )


