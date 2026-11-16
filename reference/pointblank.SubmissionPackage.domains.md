# domains

The names of the datasets in the package, in alphabetical order.

`domains: list[str]`

The names are the domain codes (such as `"AE"` and `"DM"`) under which the datasets were added, in uppercase. This is a property, so it's used without parentheses.

## Examples

Let's make a package with three datasets and get their names:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"USUBJID": ["S-001", "S-002"]})
ae = pl.DataFrame({"USUBJID": ["S-001"], "AESEQ": [1]})
suppae = pl.DataFrame({"USUBJID": ["S-001"], "RDOMAIN": ["AE"]})

study = pb.SubmissionPackage(datasets={"dm": dm, "SUPPAE": suppae, "ae": ae})

study.domains
```

    ['AE', 'DM', 'SUPPAE']

The names are in uppercase and in alphabetical order, whatever order the datasets were given in. A name can also be tested with `in`, which ignores case (`"ae" in study` is `True`), and `len(study)` gives the number of datasets.
