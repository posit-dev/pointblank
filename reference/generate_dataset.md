# generate_dataset()

Generate synthetic test data from a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/schema.py#L1296-L1565)

``` python
generate_dataset(
    schema,
    n=100,
    seed=None,
    output="polars",
    country="US",
    shuffle=True,
    weighted=True,
)
```

This function generates random data that conforms to a schema's column definitions. When the schema is defined using `Field` objects with constraints (e.g., `min_val=`, `max_val=`, `pattern=`, `preset=`), the generated data will respect those constraints.

## Parameters

`schema: Schema`  
The schema object defining the structure and constraints of the data to generate. Each column can be specified using a field helper function (e.g., [int_field()](../reference/int_field.md#pointblank.int_field), [string_field()](../reference/string_field.md#pointblank.string_field)) for fine-grained control, or as a simple dtype string (e.g., `"Int64"`, `"String"`) for unconstrained generation.

`n: int = 100`  
Number of rows to generate. The default is `100`.

`seed: int | None = None`  
Random seed for reproducibility. If provided, the same seed will produce the same data. Default is `None` (non-deterministic).

`output: Literal["polars", "pandas", "dict"] = "polars"`  
Output format for the generated data. Options are: (1) `"polars"` (the default) returns a Polars DataFrame, (2) `"pandas"` returns a Pandas DataFrame, and (3) `"dict"` returns a dictionary of lists.

`country: str | list[str] | dict[str, float] = "US"`  
Country code(s) for locale-aware generation when using presets. Accepts a single ISO 3166-1 alpha-2 or alpha-3 code (e.g., `"US"`, `"DEU"`), a list of codes for uniform mixing (e.g., `["US", "DE", "JP"]`), or a dict mapping codes to positive weights (e.g., `{"US": 60, "DE": 25, "JP": 15}`). See the *Locale Mixing* section below for details. The default is `"US"`.

`shuffle: bool = True`  
When `country=` is a list or dict (multi-country mixing), controls whether rows from different countries are interleaved randomly (`True`, the default) or grouped by country in the order the countries are specified (`False`). Ignored when `country=` is a single string.

`weighted: bool = True`  
When `True`, names and locations are sampled according to real-world frequency tiers. Common names like "James" and "Smith" appear far more often than rare names. Large cities like New York and Los Angeles dominate over small towns. Only affects data files that have been migrated to the tiered format; flat-list data always uses uniform sampling. Default is `True`.

## Returns

`DataFrame or dict`  
Generated data in the requested format.

## Raises

`ValueError`  
If the schema has no columns or if constraints cannot be satisfied.

`ImportError`  
If required optional dependencies are not installed.

## Presets And The `country=` Parameter

Several [string_field()](../reference/string_field.md#pointblank.string_field) presets produce locale-aware data that varies depending on the `country=` parameter. The following presets are particularly affected:

- **Address-related presets** (`"address"`, `"city"`, `"state"`, `"postcode"`, `"phone_number"`, `"latitude"`, `"longitude"`, `"license_plate"`): produce addresses, cities, postal codes, phone numbers, and license plates formatted for the specified country. For example, `country="DE"` yields German street names and PLZ postal codes, while `country="JP"` yields Japanese addresses. License plates for CA, US, DE, AU, and GB use province/state-specific formats when location fields are present.
- **Person-related presets** (`"name"`, `"name_full"`, `"first_name"`, `"last_name"`, `"email"`, `"user_name"`) produce culturally appropriate names for the specified country. For example, `country="FR"` produces French names, while `country="KR"` produces Korean names.
- **Business-related presets** (`"job"`, `"company"`): when both are present, the job and company are drawn from the same industry for realism. The `"name_full"` preset will also add profession-matched titles (e.g., "Dr." for doctors, "Prof." for professors), and integer columns named `age` are automatically constrained to working-age range (22-65).
- **Financial presets** (`"iban"`, `"ssn"`, `"license_plate"`): produce identifiers in the format used by the specified country.
- **Locale preset** (`"locale_code"`): returns a locale identifier (e.g., `"en_US"`, `"de_DE"`) derived from the country. Multilingual countries randomly select among their official locale codes (e.g., `"CH"` yields `"de_CH"`, `"fr_CH"`, or `"it_CH"`).

When multiple columns in the same schema use related presets, the generated data is automatically coherent across those columns within each row. Person-related presets will share the same identity (e.g., the email is derived from the name), address-related presets will share the same location (e.g., the city matches the address), and business-related presets will share the same industry context.

## Locale Mixing

The `country=` parameter accepts three input forms for flexible locale control:

1)  a **single string** (the default), such as `"US"` or `"DEU"`, which generates all rows from one locale; (2) a **list of strings**, such as `["US", "DE", "JP"]`, which splits rows equally across the listed countries; and (3) a **dict of weights**, such as `{"US": 0.6, "DE": 0.3, "FR": 0.1}`, which allocates rows proportionally (weights are auto-normalized, so `{"US": 6, "DE": 3, "FR": 1}` is equivalent).

Row counts are distributed using largest-remainder apportionment so they always sum to exactly `n=`. Each country's rows are generated as an independent batch (preserving all cross-column coherence within each batch), then either interleaved randomly (`shuffle=True`, the default) or left in contiguous country blocks (`shuffle=False`).

## Supported Countries

The `country=` parameter currently supports 100 countries with full locale data:

**Europe (38 countries):** Armenia (`"AM"`), Austria (`"AT"`), Azerbaijan (`"AZ"`), Belgium (`"BE"`), Bulgaria (`"BG"`), Croatia (`"HR"`), Cyprus (`"CY"`), Czech Republic (`"CZ"`), Denmark (`"DK"`), Estonia (`"EE"`), Finland (`"FI"`), France (`"FR"`), Georgia (`"GE"`), Germany (`"DE"`), Greece (`"GR"`), Hungary (`"HU"`), Iceland (`"IS"`), Ireland (`"IE"`), Italy (`"IT"`), Latvia (`"LV"`), Lithuania (`"LT"`), Luxembourg (`"LU"`), Malta (`"MT"`), Moldova (`"MD"`), Netherlands (`"NL"`), Norway (`"NO"`), Poland (`"PL"`), Portugal (`"PT"`), Romania (`"RO"`), Russia (`"RU"`), Serbia (`"RS"`), Slovakia (`"SK"`), Slovenia (`"SI"`), Spain (`"ES"`), Sweden (`"SE"`), Switzerland (`"CH"`), Ukraine (`"UA"`), United Kingdom (`"GB"`)

**Americas (19 countries):** Argentina (`"AR"`), Bolivia (`"BO"`), Brazil (`"BR"`), Canada (`"CA"`), Chile (`"CL"`), Colombia (`"CO"`), Costa Rica (`"CR"`), Dominican Republic (`"DO"`), Ecuador (`"EC"`), El Salvador (`"SV"`), Guatemala (`"GT"`), Honduras (`"HN"`), Jamaica (`"JM"`), Mexico (`"MX"`), Panama (`"PA"`), Paraguay (`"PY"`), Peru (`"PE"`), United States (`"US"`), Uruguay (`"UY"`)

**Asia-Pacific (22 countries):** Australia (`"AU"`), Bangladesh (`"BD"`), Cambodia (`"KH"`), China (`"CN"`), Hong Kong (`"HK"`), India (`"IN"`), Indonesia (`"ID"`), Japan (`"JP"`), Kazakhstan (`"KZ"`), Malaysia (`"MY"`), Myanmar (`"MM"`), Nepal (`"NP"`), New Zealand (`"NZ"`), Pakistan (`"PK"`), Philippines (`"PH"`), Singapore (`"SG"`), South Korea (`"KR"`), Sri Lanka (`"LK"`), Taiwan (`"TW"`), Thailand (`"TH"`), Uzbekistan (`"UZ"`), Vietnam (`"VN"`)

**Middle East & Africa (21 countries):** Algeria (`"DZ"`), Cameroon (`"CM"`), Egypt (`"EG"`), Ethiopia (`"ET"`), Ghana (`"GH"`), Israel (`"IL"`), Jordan (`"JO"`), Kenya (`"KE"`), Lebanon (`"LB"`), Morocco (`"MA"`), Mozambique (`"MZ"`), Nigeria (`"NG"`), Rwanda (`"RW"`), Saudi Arabia (`"SA"`), Senegal (`"SN"`), South Africa (`"ZA"`), Tanzania (`"TZ"`), Tunisia (`"TN"`), Turkey (`"TR"`), Uganda (`"UG"`), United Arab Emirates (`"AE"`)

## Pytest Fixture

When Pointblank is installed, a [generate_dataset](../reference/generate_dataset.md#pointblank.generate_dataset) pytest fixture is automatically available in all test files: no imports or `conftest.py` setup required. The fixture behaves identically to this function, but derives a deterministic seed from the test's fully-qualified name when `seed=` is not provided.

This means:

- the **same test** always produces the **same data**, with no manual seed management.
- **different tests** get different seeds, so they exercise different data.
- **you** can still pass an explicit `seed=` to override the automatic seed.
- **calling** the fixture **multiple times** within one test produces different (but still deterministic) data on each call.
- the fixture exposes `.default_seed` and `.last_seed` attributes for debugging.

``` python
def test_my_pipeline(generate_dataset):
    import pointblank as pb

    schema = pb.Schema(
        user_id=pb.int_field(unique=True),
        email=pb.string_field(preset="email"),
        age=pb.int_field(min_val=18, max_val=100),
    )
    df = generate_dataset(schema, n=500, country="DE")
    # seed is derived from "test_my_pipeline" -- same data every run
    result = my_pipeline(df)
    assert result.shape[0] == 500
```

Multiple datasets can be generated within the same test, each with its own deterministic seed:

``` python
def test_merge(generate_dataset):
    customers = generate_dataset(customer_schema, n=1000, country="US")
    orders = generate_dataset(order_schema, n=5000)
    # Both DataFrames are deterministic; each call gets a unique seed
```

When a test fails, include the seed in the assertion message so the failure is easy to reproduce:

``` python
def test_age_range(generate_dataset):
    df = generate_dataset(schema, n=100)
    assert df["age"].min() >= 18, f"Failed with seed {generate_dataset.last_seed}"
```

## Seed Stability

A given seed (whether explicit or auto-derived) is guaranteed to produce identical output **within the same Pointblank version**. Across versions, changes to country data files or generator logic may alter the output for a given seed.

For CI pipelines that require bit-exact data across library upgrades, save generated DataFrames as Parquet or CSV snapshot files rather than relying on cross-version seed stability. This is the same approach used by snapshot-testing tools like `pytest-snapshot` and `syrupy`.

## Examples

Here we define a schema with field constraints and generate test data from it:

``` python
import pointblank as pb

schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns4 |  |  |  |  |
|----|----|----|----|----|
|  | user_id *Int64* | email *String* | age *Int64* | status *String* |
| 1 | 7188536481533917197 | d_martin@aol.com | 55 | pending |
| 2 | 2674009078779859984 | nancygonzalez@icloud.com | 28 | active |
| 3 | 7652102777077138151 | jturner@aol.com | 20 | active |
| 4 | 157503859921753049 | georgeevans@zoho.com | 93 | inactive |
| 5 | 2829213282471975080 | pwilliams@outlook.com | 57 | pending |
| 96 | 7027508096731143831 | isaiah.murphy@zoho.com | 68 | active |
| 97 | 6055996548456656575 | brodriguez@yandex.com | 20 | inactive |
| 98 | 3822709996092631588 | mstevens26@aol.com | 38 | inactive |
| 99 | 1522653102058131295 | pjenkins29@yandex.com | 46 | active |
| 100 | 5690877051669225499 | stephanie.santos40@gmail.com | 19 | pending |

It's also possible to generate data from a simple, dtype-only schema. Setting `output="pandas"` returns a Pandas DataFrame:

``` python
schema = pb.Schema(name="String", age="Int64", active="Boolean")

pb.preview(pb.generate_dataset(schema, n=50, seed=23, output="pandas"))
```

| PandasRows50Columns3 |                     |                      |               |
|----------------------|---------------------|----------------------|---------------|
|                      | name *object*       | age *int64*          | active *bool* |
| 1                    | 51fbLtByHw          | -1406612057389349638 | False         |
| 2                    | UmrCa               | -2617964757147985650 | False         |
| 3                    | ND5bgfTF            | -5681649629593590626 | False         |
| 4                    | bGOUBwXdnYcLxQ      | -8963716282372353309 | True          |
| 5                    | NnVxKW              | -7269866261640175410 | False         |
| 46                   | 8VQTQ3rUkjMe        | 6777163490966252062  | True          |
| 47                   | ZGDIWh7eBERjPZthNbW | 4534912642422597042  | False         |
| 48                   | MnIPm2wYtrTsBF6I8   | -7714433421897454051 | False         |
| 49                   | sv9VboYQKY5JjeSX8i  | -4108772566563722234 | True          |
| 50                   | S6tq                | -7629746523602015996 | True          |

When using presets, the `country=` parameter controls the locale. Here, `country="DE"` produces German names and addresses:

``` python
schema = pb.Schema(
    name=pb.string_field(preset="name"),
    address=pb.string_field(preset="address"),
    city=pb.string_field(preset="city"),
)

pb.preview(pb.generate_dataset(schema, n=20, seed=23, country="DE"))
```

| PolarsRows20Columns3 |  |  |  |
|----|----|----|----|
|  | name *String* | address *String* | city *String* |
| 1 | Alexandra Koch | Königstraße 3852, 14877 Potsdam | Potsdam |
| 2 | Christiane Becker | Oleariusstraße 65, Whg. 768, 06602 Halle (Saale) | Halle (Saale) |
| 3 | Thomas Mertens | Pfingstweidstraße 3336, Whg. 764, 60027 Frankfurt am Main | Frankfurt am Main |
| 4 | Jule Schwarz | Ferdinand-Rhode-Straße 1341, Whg. 706, 04479 Leipzig | Leipzig |
| 5 | Gerda Haas | Hohenzollernring 8621, 50441 Köln | Cologne |
| 16 | Frauke Kaiser | Seckenheimer Straße 4826, 68490 Mannheim | Mannheim |
| 17 | Lukas Herrmann | Gartenstraße 9878, 15915 Frankfurt (Oder) | Frankfurt (Oder) |
| 18 | Bernhard Schulz | Herrenstraße 5744, 76233 Karlsruhe | Karlsruhe |
| 19 | Irma Stock | Waldstraße 5190, Whg. 602, 41938 Mönchengladbach | Monchengladbach |
| 20 | Berthold Scholz | Moserstraße 5930, Whg. 468, 70384 Stuttgart | Stuttgart |

We can combine several field types with nullable columns in a mixed-type dataset:

``` python
from datetime import date, timedelta

schema = pb.Schema(
    id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    score=pb.float_field(min_val=0.0, max_val=100.0),
    is_active=pb.bool_field(p_true=0.75),
    joined=pb.date_field(min_date=date(2020, 1, 1), max_date=date(2024, 12, 31)),
    session_time=pb.duration_field(
        min_duration=timedelta(minutes=1),
        max_duration=timedelta(hours=3),
        nullable=True, null_probability=0.2,
    ),
)

pb.preview(pb.generate_dataset(schema, n=50, seed=23))
```

| PolarsRows50Columns6 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | id *Int64* | name *String* | score *Float64* | is_active *Boolean* | joined *Date* | session_time *Duration* |
| 1 | 7188536481533917197 | Doris Martin | 92.48652516259452 | False | 2024-05-15 | 1:20:09 |
| 2 | 2674009078779859984 | Nancy Gonzalez | 94.86057779931771 | False | 2021-08-16 | 0:23:48 |
| 3 | 7652102777077138151 | Jessica Turner | 89.24333440485793 | False | 2024-08-26 | None |
| 4 | 157503859921753049 | George Evans | 8.355067683068363 | True | 2020-06-20 | 2:42:39 |
| 5 | 2829213282471975080 | Patricia Williams | 59.20272268857353 | True | 2020-02-04 | None |
| 46 | 8670836018805171304 | Michael Hoffman | 27.556446150015233 | True | 2023-03-04 | 2:12:54 |
| 47 | 2587902378814764220 | Brian Campbell | 57.282189488843784 | True | 2024-04-05 | None |
| 48 | 5441450987457280882 | Teresa Roberts | 82.06631808725244 | False | 2024-10-27 | None |
| 49 | 1005771189117755519 | Vincent Rodriguez | 33.08048479932988 | True | 2022-01-25 | 2:56:24 |
| 50 | 8302188861545620440 | Susan Ramirez | 36.96539320060992 | True | 2023-03-17 | 0:45:40 |
