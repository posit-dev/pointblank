"""
SDTM and ADaM templates must fit SAS transport (XPT v5) limits and use the IG wording.

XPT v5, the format required for FDA submissions, caps variable names at 8 characters and
variable and dataset labels at 40 characters.
"""

from __future__ import annotations

import pytest

from pointblank.metadata._adam_templates import get_adam_dataset, list_adam_datasets
from pointblank.metadata._sdtm_templates import get_sdtm_domain, list_sdtm_domains

TEMPLATES = [("SDTM", d, get_sdtm_domain(d)) for d in list_sdtm_domains()] + [
    ("ADaM", d, get_adam_dataset(d)) for d in list_adam_datasets()
]


@pytest.mark.parametrize(
    "template", [t for _, _, t in TEMPLATES], ids=[f"{s}-{d}" for s, d, _ in TEMPLATES]
)
def test_template_fits_sas_transport_limits(template):
    name = getattr(template, "domain", None) or template.name
    assert len(name) <= 8
    assert len(template.label) <= 40, template.label

    too_long = [
        f"{v.name}: {v.label!r} ({len(v.label)})" for v in template.variables if len(v.label) > 40
    ]
    assert too_long == []
    assert [v.name for v in template.variables if len(v.name) > 8] == []


@pytest.mark.parametrize(
    "get, dataset, variable, label",
    [
        # SDTMIG 3.4
        (get_sdtm_domain, "DS", "DSSTDY", "Study Day of Start of Disposition Event"),
        (get_sdtm_domain, "MH", "MHSTDTC", "Start Date/Time of Medical History Event"),
        (get_sdtm_domain, "MH", "MHENDTC", "End Date/Time of Medical History Event"),
        (get_sdtm_domain, "LB", "LBORNRLO", "Reference Range Lower Limit in Orig Unit"),
        (get_sdtm_domain, "LB", "LBORNRHI", "Reference Range Upper Limit in Orig Unit"),
        # ADaMIG 1.3
        (get_adam_dataset, "ADSL", "DCTREAS", "Reason for Discontinuation of Treatment"),
        (get_adam_dataset, "BDS", "ANL01FL", "Analysis Flag 01"),
        (get_adam_dataset, "BDS", "SRCDOM", "Source Data"),
        (get_adam_dataset, "BDS", "SRCVAR", "Source Variable"),
        (get_adam_dataset, "BDS", "SRCSEQ", "Source Sequence Number"),
        # ADaM OCCDS
        (get_adam_dataset, "ADAE", "ADURN", "Analysis Duration (N)"),
        (get_adam_dataset, "ADAE", "ADURU", "Analysis Duration Units"),
        (get_adam_dataset, "ADAE", "TRTEMFL", "Treatment Emergent Analysis Flag"),
        # ADaM BDS for time-to-event
        (get_adam_dataset, "ADTTE", "STARTDT", "Time-to-Event Origin Date for Subject"),
        (get_adam_dataset, "ADTTE", "EVNTDESC", "Event or Censoring Description"),
        (get_adam_dataset, "ADTTE", "SRCDOM", "Source Data"),
    ],
)
def test_template_labels_use_ig_wording(get, dataset, variable, label):
    spec = next(v for v in get(dataset).variables if v.name == variable)
    assert spec.label == label
