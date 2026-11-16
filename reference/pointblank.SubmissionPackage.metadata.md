# metadata

The package's Define-XML, as a [MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage).

`metadata: MetadataPackage | None`

If the package's `define` is a path to a `define.xml` file, the file is imported with [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) the first time this property is used, and the result is kept for later use. If `define` is already a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage), that is returned. The metadata has a [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport) for each dataset that the Define-XML describes, with its variables and codelists. The property is `None` if the package has no Define-XML. This is a property, so it's used without parentheses.

## Examples

Let's make a package with the DM dataset and the Define-XML file that come with Pointblank:

``` python
import pointblank as pb
import pyreadstat

dm, _ = pyreadstat.read_xport(pb.load_metadata_example("dm.xpt"))

study = pb.SubmissionPackage(
    datasets={"DM": dm}, define=pb.load_metadata_example("define.xml")
)

print(study.metadata)
```

    Metadata Package
      Name: STUDY.XYZ789
      Description: CDISC Define-XML 2.0 study metadata
      Datasets: 2

      [DM] 13 variables
      [AE] 10 variables

The Define-XML describes two datasets, DM and AE, though the package only has DM. The metadata of a dataset can be looked at in more detail, such as the values allowed by the codelist of a variable:

``` python
race = study.metadata["DM"].get_variable("RACE")

race.allowed_values
```

    ['WHITE',
     'BLACK OR AFRICAN AMERICAN',
     'ASIAN',
     'AMERICAN INDIAN OR ALASKA NATIVE',
     'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']

`BLACK` isn't one of them, though one of the subjects in DM has it as their `RACE`. Without a `define`, the property is `None`.
