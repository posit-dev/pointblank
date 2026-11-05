# string_field()

Create a string column specification for use in a schema.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/field.py#L831-L1028)

``` python
string_field(
    min_length=None,
    max_length=None,
    pattern=None,
    preset=None,
    allowed=None,
    nullable=False,
    null_probability=0.0,
    unique=False,
    generator=None,
)
```

The [string_field()](../reference/string_field.md#pointblank.string_field) function defines the constraints and behavior for a string column when generating synthetic data with [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset). It provides three main modes of string generation: (1) controlled random strings with `min_length=`/`max_length=`, (2) strings matching a regular expression via `pattern=`, or (3) realistic data using `preset=` (e.g., `"email"`, `"name"`, `"address"`). You can also restrict values to a fixed set with `allowed=`. Only one of `preset=`, `pattern=`, or `allowed=` can be specified at a time.

When no special mode is selected, random alphanumeric strings are generated with lengths between `min_length=` and `max_length=` (defaulting to 1-20 characters).

## Parameters

`min_length: int | None = None`  
Minimum string length (for random string generation). Default is `None` (defaults to `1`). Only applies when `preset=`, `pattern=`, and `allowed=` are all `None`.

`max_length: int | None = None`  
Maximum string length (for random string generation). Default is `None` (defaults to `20`). Only applies when `preset=`, `pattern=`, and `allowed=` are all `None`.

`pattern: str | None = None`  
Regular expression pattern that generated strings must match. Supports character classes (e.g., `[A-Z]`, `[0-9]`), quantifiers (e.g., `{3}`, `{2,5}`), alternation, and groups. Cannot be combined with `preset=` or `allowed=`.

`preset: str | None = None`  
Preset name for generating realistic data. When specified, values are produced using locale-aware data generation, and the `country=` parameter of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset) controls the locale. Cannot be combined with `pattern=` or `allowed=`. See the **Available Presets** section below for the full list.

`allowed: list[str] | None = None`  
List of allowed string values (categorical constraint). Values are sampled uniformly from this list. Cannot be combined with `preset=` or `pattern=`.

`nullable: bool = False`  
Whether the column can contain null values. Default is `False`.

`null_probability: float = 0.0`  
Probability of generating a null value for each row when `nullable=True`. Must be between `0.0` and `1.0`. Default is `0.0`.

`unique: bool = False`  
Whether all values must be unique. Default is `False`. When `True`, the generator will retry until it produces `n` distinct values.

`generator: Callable[[], Any] | None = None`  
Custom callable that generates values. When provided, it's used instead of the other constraints, while `unique=` and `nullable=` still apply. The callable should take no arguments and return a single string value. It isn't controlled by the `seed=` of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset), so it needs its own seeded random number generator for reproducible data.

## Returns

`StringField`  
A string field specification that can be passed to [Schema()](../reference/Schema.md#pointblank.Schema).

## Raises

`ValueError`  
If more than one of `preset=`, `pattern=`, or `allowed=` is specified; if `allowed=` is an empty list; if `min_length` or `max_length` is negative; if `min_length` exceeds `max_length`; or if `preset` is not a recognized preset name.

## Available Presets

The `preset=` parameter accepts one of the following preset names, organized by category. When a preset is used, the `country=` parameter of [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset) controls the locale for region-specific formatting (e.g., address formats, phone number patterns).

**Personal:** `"name"` (first + last name), `"name_full"` (full name with possible prefix or suffix), `"first_name"`, `"last_name"`, `"gender"` (person's gender, coherent with name), `"email"` (realistic email address), `"phone_number"`, `"address"` (full street address), `"city"`, `"state"`, `"country"`, `"country_code_2"` (ISO 3166-1 alpha-2 code, e.g., `"US"`), `"country_code_3"` (ISO 3166-1 alpha-3 code, e.g., `"USA"`), `"postcode"`, `"latitude"`, `"longitude"`

**Business:** `"company"` (company name), `"job"` (job title), `"catch_phrase"`

**Internet:** `"url"`, `"domain_name"`, `"ipv4"`, `"ipv6"`, `"user_name"`, `"password"`

**Text:** `"text"` (paragraph of text), `"sentence"`, `"paragraph"`, `"word"`

**Financial:** `"credit_card_number"`, `"credit_card_provider"` (Visa, Mastercard, American Express, or Discover), `"iban"`, `"currency_code"`

**Identifiers:** `"uuid4"`, `"md5"` (MD5 hash, 32 hex chars), `"sha1"` (SHA-1 hash, 40 hex chars), `"sha256"` (SHA-256 hash, 64 hex chars), `"ssn"` (social security number), `"license_plate"`

**Barcodes:** `"ean8"` (EAN-8 barcode with valid check digit), `"ean13"` (EAN-13 barcode with valid check digit)

**Date/Time (as strings):** `"date_this_year"`, `"date_this_decade"`, `"date_between"` (random date between 2000-2025), `"date_range"` (two dates joined with an en-dash, e.g., `"2012-05-12 - 2015-11-22"`), `"future_date"` (up to 1 year ahead), `"past_date"` (up to 10 years back), `"time"`

**Miscellaneous:** `"color_name"`, `"file_name"`, `"file_extension"`, `"mime_type"`, `"user_agent"` (browser user agent string with country-specific browser weighting), `"locale_code"` (locale identifier like `"en_US"`, `"de_DE"`; multilingual countries return a random official locale)

## Coherent Data Generation

When multiple columns in the same schema use related presets, the generated data will be coherent across those columns within each row. Specifically:

- **Person-related presets** (`"name"`, `"name_full"`, `"first_name"`, `"last_name"`, `"gender"`, `"email"`, `"user_name"`): the email and username will be derived from the person's name, and `"gender"` will match the person's first name.
- **Address-related presets** (`"address"`, `"city"`, `"state"`, `"postcode"`, `"phone_number"`, `"latitude"`, `"longitude"`): the city, state, and postcode will correspond to the same location within the address.
- **Credit card presets** (`"credit_card_number"`, `"credit_card_provider"`): the card number prefix and provider name will be consistent (e.g., "Visa" with a "4"-prefixed number).

This coherence is automatic and requires no additional configuration.

## Examples

The `preset=` parameter generates realistic personal data, while `allowed=` restricts values to a categorical set:

``` python
import pointblank as pb

schema = pb.Schema(
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email", unique=True),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns3 |  |  |  |
|----|----|----|----|
|  | name *String* | email *String* | status *String* |
| 1 | Doris Martin | d_martin@aol.com | pending |
| 2 | Nancy Gonzalez | nancygonzalez@icloud.com | pending |
| 3 | Jessica Turner | jturner@aol.com | inactive |
| 4 | George Evans | georgeevans@zoho.com | pending |
| 5 | Patricia Williams | pwilliams@outlook.com | active |
| 96 | Isaiah Murphy | isaiah.murphy@zoho.com | active |
| 97 | Brittany Rodriguez | brodriguez@yandex.com | active |
| 98 | Megan Stevens | mstevens26@aol.com | active |
| 99 | Pamela Jenkins | pjenkins29@yandex.com | inactive |
| 100 | Stephanie Santos | stephanie.santos40@gmail.com | active |

We can also generate strings that match a regular expression with `pattern=` (e.g., product codes, identifiers):

``` python
schema = pb.Schema(
    product_code=pb.string_field(pattern=r"[A-Z]{3}-[0-9]{4}"),
    batch_id=pb.string_field(pattern=r"BATCH-[A-Z][0-9]{3}"),
    sku=pb.string_field(pattern=r"[A-Z]{2}[0-9]{6}"),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=23))
```

| PolarsRows30Columns3 |                       |                   |              |
|----------------------|-----------------------|-------------------|--------------|
|                      | product_code *String* | batch_id *String* | sku *String* |
| 1                    | TUA-7878              | BATCH-C572        | ON747986     |
| 2                    | CIR-3746              | BATCH-E915        | RG842341     |
| 3                    | JKS-6678              | BATCH-Y948        | MT768044     |
| 4                    | DVC-7585              | BATCH-E014        | KN345811     |
| 5                    | BLC-1897              | BATCH-X610        | IO470502     |
| 26                   | GFC-4249              | BATCH-K937        | BZ789672     |
| 27                   | BKQ-1618              | BATCH-E186        | VL008570     |
| 28                   | ZAI-6997              | BATCH-M421        | DQ431897     |
| 29                   | AXI-9531              | BATCH-R168        | IA831952     |
| 30                   | LOT-0185              | BATCH-U105        | BZ850635     |

For random alphanumeric strings, `min_length=` and `max_length=` control the length. Adding `nullable=True` introduces missing values:

``` python
schema = pb.Schema(
    short_code=pb.string_field(min_length=3, max_length=5),
    notes=pb.string_field(
        min_length=10, max_length=50,
        nullable=True, null_probability=0.4,
    ),
)

pb.preview(pb.generate_dataset(schema, n=30, seed=7))
```

| PolarsRows30Columns2 |  |  |
|----|----|----|
|  | short_code *String* | notes *String* |
| 1 | tU42 | None |
| 2 | MEcNn | 4zs23WtLuh9sGp7VDs12SAQlXK8Ey41VEhvtMJqGrHjB8 |
| 3 | jnO5 | None |
| 4 | 4iq | tzSuQ08PkX1mAzXNMgFeMm |
| 5 | 4Ev | CHCDN7OCn1VA0UGobtiFtBMHvFsvgh4tPLf2HYZy76OE0Fx4VG |
| 26 | tRt | None |
| 27 | 8gGNT | None |
| 28 | Nsz | None |
| 29 | AfwD3 | oPnSPvRPVOvRbABS9rwsnTbmQJ4xY4UhXKSkUO6Fk |
| 30 | zbInW | kKWu7q4gmGMzIc |

It's possible to combine business and internet presets to build a company directory:

``` python
schema = pb.Schema(
    company=pb.string_field(preset="company"),
    domain=pb.string_field(preset="domain_name"),
    industry_tag=pb.string_field(allowed=["tech", "finance", "health", "retail"]),
)

pb.preview(pb.generate_dataset(schema, n=20, seed=55))
```

| PolarsRows20Columns3 |  |  |  |
|----|----|----|----|
|  | company *String* | domain *String* | industry_tag *String* |
| 1 | King Corp | made.me | retail |
| 2 | Robertson LLC | company.gov | tech |
| 3 | Premier Analytics | hand.com | health |
| 4 | Superior Research | who.tech | finance |
| 5 | Slack | had.com | tech |
| 16 | Microsoft | been.biz | health |
| 17 | Coleman Realty | number.app | tech |
| 18 | Gonzalez and Riley | the.app | finance |
| 19 | Prestige Technologies Partners | day.dev | tech |
| 20 | Rodriguez Ltd | back.site | retail |

Unlike the person and address presets, `"domain_name"` isn't coherent with `"company"`, so each domain is unrelated to the company in its row. The `industry_tag` values come from the `allowed=` list, also independently of the company.
