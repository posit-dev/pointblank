# Generating Test Data

Pointblank provides a built-in test data generation system that creates realistic, locale-aware synthetic data based on schema definitions. This is useful for testing validation rules, creating sample datasets, and generating fixture data for development.

This page covers the basics: defining a schema with typed fields, the available presets, and the output formats. The rest of this section covers [profile fields](profile-fields.md) for bundles of person data, [country-specific data](country-specific-data.md) for locale-aware generation, and the [pytest fixture](pytest-fixture.md) for using generated data in tests.

> **Note**
>
> Throughout this guide, we use `pb.preview()` to display generated datasets with nice HTML formatting. This is optional: `pb.generate_dataset()` returns a standard DataFrame that you can display or manipulate however you prefer.

## Quick Start

Generate test data using a schema with field constraints:

``` python
import pointblank as pb

# Define a schema with typed field specifications
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=80),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# Generate 100 rows of test data (seed ensures reproducibility)
pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns5 |  |  |  |  |  |
|----|----|----|----|----|----|
|  | user_id *Int64* | name *String* | email *String* | age *Int64* | status *String* |
| 1 | 5702568738186988586 | Doris Martin | d_martin@aol.com | 30 | pending |
| 2 | 2878223714068359465 | Nancy Gonzalez | nancygonzalez@icloud.com | 74 | pending |
| 3 | 2827316211584448787 | Jessica Turner | jturner@aol.com | 66 | inactive |
| 4 | 4882856902780995319 | George Evans | georgeevans@zoho.com | 60 | pending |
| 5 | 5672268219098327547 | Patricia Williams | pwilliams@outlook.com | 42 | active |
| 96 | 8515820166226230863 | Isaiah Murphy | isaiah.murphy@zoho.com | 74 | active |
| 97 | 1273514605515757248 | Brittany Rodriguez | brodriguez@yandex.com | 33 | active |
| 98 | 302880528361637242 | Megan Stevens | mstevens26@aol.com | 36 | active |
| 99 | 7613481963472661909 | Pamela Jenkins | pjenkins29@yandex.com | 28 | inactive |
| 100 | 8634311640077946044 | Stephanie Santos | stephanie.santos40@gmail.com | 68 | active |

## Field Types

Pointblank provides helper functions for defining typed columns with constraints:

| Function | Description | Key Parameters |
|----|----|----|
| [int_field()](../../reference/int_field.md#pointblank.int_field) | Integer columns | `min_val`, `max_val`, `allowed`, `unique` |
| [float_field()](../../reference/float_field.md#pointblank.float_field) | Float columns | `min_val`, `max_val`, `allowed` |
| [string_field()](../../reference/string_field.md#pointblank.string_field) | String columns | `preset`, `pattern`, `allowed`, `unique` |
| [bool_field()](../../reference/bool_field.md#pointblank.bool_field) | Boolean columns | `p_true` (probability of True) |
| [date_field()](../../reference/date_field.md#pointblank.date_field) | Date columns | `min_date`, `max_date` |
| [datetime_field()](../../reference/datetime_field.md#pointblank.datetime_field) | Datetime columns | `min_date`, `max_date` |
| [time_field()](../../reference/time_field.md#pointblank.time_field) | Time columns | `min_time`, `max_time` |
| [duration_field()](../../reference/duration_field.md#pointblank.duration_field) | Duration columns | `min_duration`, `max_duration` |
| [profile_fields()](../../reference/profile_fields.md#pointblank.profile_fields) | Bundled person-profile fields | `set`, `split_name`, `include`, `exclude`, `prefix` |

### Integer Fields

Integer fields support range constraints with `min_val` and `max_val`, discrete allowed values with `allowed`, and uniqueness enforcement with `unique=True`:

``` python
schema = pb.Schema(
    id=pb.int_field(min_val=1000, max_val=9999, unique=True),
    quantity=pb.int_field(min_val=1, max_val=100),
    rating=pb.int_field(allowed=[1, 2, 3, 4, 5]),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns3 |            |                  |                |
|-----------------------|------------|------------------|----------------|
|                       | id *Int64* | quantity *Int64* | rating *Int64* |
| 1                     | 1729       | 76               | 5              |
| 2                     | 4280       | 64               | 5              |
| 3                     | 7912       | 55               | 1              |
| 4                     | 7034       | 93               | 2              |
| 5                     | 7349       | 6                | 5              |
| 96                    | 4609       | 23               | 1              |
| 97                    | 6309       | 20               | 5              |
| 98                    | 6278       | 85               | 4              |
| 99                    | 2442       | 73               | 4              |
| 100                   | 7384       | 51               | 1              |

The `unique=True` constraint ensures no duplicate values appear in that column, which is useful for generating primary keys or identifiers.

### Float Fields

Float fields work similarly to integers, with `min_val` and `max_val` defining the range of generated values:

``` python
schema = pb.Schema(
    price=pb.float_field(min_val=0.0, max_val=1000.0),
    discount=pb.float_field(min_val=0.0, max_val=0.5),
    temperature=pb.float_field(min_val=-40.0, max_val=50.0),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns3 |  |  |  |
|----|----|----|----|
|  | price *Float64* | discount *Float64* | temperature *Float64* |
| 1 | 593.8006425754224 | 0.2535084284143328 | -5.157722860993047 |
| 2 | 760.528834268454 | 0.34000920640464805 | 31.296623312850457 |
| 3 | 172.85941641181378 | 0.4165680275292023 | 39.77800136715432 |
| 4 | 894.1264909136073 | 0.09245033007853415 | -12.968123980080339 |
| 5 | 605.7497536309945 | 0.1974449434309311 | -2.867472818974349 |
| 96 | 805.6472016008495 | 0.4481010934681121 | -23.090891067448077 |
| 97 | 36.35284018944773 | 0.2980495735979266 | 25.794353646053793 |
| 98 | 853.6603628813587 | 0.27386014084204674 | 5.683426744810603 |
| 99 | 37.50970265716247 | 0.40789062461815445 | 42.06519501931237 |
| 100 | 382.81356300515137 | 0.281960777497605 | 16.95315561462961 |

Values are uniformly distributed across the specified range, making this useful for simulating measurements, prices, or any continuous numeric data.

### String Fields with Presets

Presets generate realistic data like names, emails, and addresses. When you include related fields like `name` and `email` in the same schema, Pointblank ensures **coherence** (e.g., the email address will be derived from the person's name), making the generated data more realistic:

``` python
schema = pb.Schema(
    full_name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    company=pb.string_field(preset="company"),
    city=pb.string_field(preset="city"),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns4 |  |  |  |  |
|----|----|----|----|----|
|  | full_name *String* | email *String* | company *String* | city *String* |
| 1 | Weston Parker | weston.parker23@gmail.com | Miller Group | Lubbock |
| 2 | Hazel Torres | hazel723@hotmail.com | Pacific Research | Anaheim |
| 3 | Lawrence Mitchell | lawrence_mitchell@zoho.com | Diaz Marketing | Phoenix |
| 4 | Maria Garcia | m_garcia@hotmail.com | Allen Industries | Denver |
| 5 | Michael Hoffman | michael.hoffman@gmail.com | Silver Systems Group | San Antonio |
| 96 | Daniel Torres | daniel_torres@icloud.com | Heritage Beverages Corp | El Paso |
| 97 | Helen Simpson | hsimpson20@yandex.com | Zoom | El Paso |
| 98 | Mark Graham | mark.graham65@mail.com | EY | Charlotte |
| 99 | Brian Moore | bmoore95@zoho.com | National Properties Associates | Los Angeles |
| 100 | Michael Ward | michael_ward@yahoo.com | Heritage Brands | San Diego |

This coherence extends to other related fields like `user_name`, which will also reflect the person's name when included alongside name and email fields.

### String Fields with Patterns

Use regex patterns to generate strings matching specific formats:

``` python
schema = pb.Schema(
    product_code=pb.string_field(pattern=r"[A-Z]{3}-[0-9]{4}"),
    phone=pb.string_field(pattern=r"\([0-9]{3}\) [0-9]{3}-[0-9]{4}"),
    hex_color=pb.string_field(pattern=r"#[0-9A-F]{6}"),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns3 |  |  |  |
|----|----|----|----|
|  | product_code *String* | phone *String* | hex_color *String* |
| 1 | TUA-7878 | \(110\) 092-9021 | \#C71D0B |
| 2 | CIR-3746 | \(463\) 602-7755 | \#1518C7 |
| 3 | JKS-6678 | \(452\) 524-0234 | \#D91C27 |
| 4 | DVC-7585 | \(519\) 205-6689 | \#CF65B0 |
| 5 | BLC-1897 | \(038\) 660-3940 | \#D12F40 |
| 96 | CBP-7267 | \(810\) 022-6051 | \#2BC09A |
| 97 | WBA-1668 | \(032\) 808-8038 | \#E298DA |
| 98 | XCW-1485 | \(219\) 255-4111 | \#831BBD |
| 99 | CWX-3605 | \(836\) 917-9664 | \#A04152 |
| 100 | NWK-7475 | \(739\) 136-9826 | \#7EBD4B |

Patterns support standard regex character classes and quantifiers, giving you flexibility to generate data matching virtually any format specification.

### Boolean Fields

Control the probability of `True` values:

``` python
schema = pb.Schema(
    is_active=pb.bool_field(p_true=0.8),      # 80% True
    is_premium=pb.bool_field(p_true=0.2),     # 20% True
    is_verified=pb.bool_field(),              # 50% True (default)
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns3 |  |  |  |
|----|----|----|----|
|  | is_active *Boolean* | is_premium *Boolean* | is_verified *Boolean* |
| 1 | False | True | True |
| 2 | False | False | False |
| 3 | True | False | False |
| 4 | True | False | True |
| 5 | True | False | True |
| 96 | True | False | True |
| 97 | False | False | True |
| 98 | True | False | True |
| 99 | True | False | True |
| 100 | True | True | True |

This probabilistic control is helpful when you need to simulate real-world distributions where certain states are more common than others.

### Date and Datetime Fields

Temporal fields accept Python `date` and `datetime` objects for their range boundaries, generating values uniformly distributed within the specified period:

``` python
from datetime import date, datetime

schema = pb.Schema(
    birth_date=pb.date_field(
        min_date=date(1960, 1, 1),
        max_date=date(2005, 12, 31)
    ),
    created_at=pb.datetime_field(
        min_date=datetime(2024, 1, 1),
        max_date=datetime(2024, 12, 31)
    ),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns2 |                   |                       |
|-----------------------|-------------------|-----------------------|
|                       | birth_date *Date* | created_at *Datetime* |
| 1                     | 1983-05-26        | 2024-02-17 16:33:53   |
| 2                     | 1987-05-07        | 2024-11-14 21:21:34   |
| 3                     | 2002-04-20        | 2024-12-29 14:40:05   |
| 4                     | 1978-09-08        | 2024-12-05 21:08:23   |
| 5                     | 1973-03-31        | 2024-02-14 09:38:39   |
| 96                    | 1999-10-20        | 2024-12-10 00:21:07   |
| 97                    | 1982-10-12        | 2024-11-05 10:23:02   |
| 98                    | 1964-05-24        | 2024-04-07 14:52:22   |
| 99                    | 1986-08-27        | 2024-02-07 05:34:29   |
| 100                   | 1963-08-21        | 2024-04-10 21:16:58   |

The same pattern applies to [time_field()](../../reference/time_field.md#pointblank.time_field) and [duration_field()](../../reference/duration_field.md#pointblank.duration_field), allowing you to generate realistic temporal data for any use case.

## Available Presets

The `preset=` parameter in [string_field()](../../reference/string_field.md#pointblank.string_field) supports many data types:

**Personal Data:**

- `name`: full name (first + last)
- `name_full`: full name with optional prefix/suffix (e.g., "Dr. Ana Sousa", "Prof. Tanaka Yuki")
- `first_name`: first name only
- `last_name`: last name only
- `gender`: person's gender (`"male"` or `"female"`), coherent with name fields
- `email`: email address
- `phone_number`: phone number in country-specific format

**Location Data:**

- `address`: full street address
- `city`: city name
- `state`: state/province name
- `country`: country name
- `country_code_2`: ISO 3166-1 alpha-2 country code (e.g., `"US"`)
- `country_code_3`: ISO 3166-1 alpha-3 country code (e.g., `"USA"`)
- `postcode`: postal/ZIP code
- `latitude`: latitude coordinate
- `longitude`: longitude coordinate

**Business Data:**

- `company`: company name
- `job`: job title
- `catch_phrase`: business catch phrase

**Internet Data:**

- `url`: website URL
- `domain_name`: domain name
- `ipv4`: IPv4 address
- `ipv6`: IPv6 address
- `user_name`: username
- `password`: password

**Financial Data:**

- `credit_card_number`: credit card number
- `credit_card_provider`: card network name (Visa, Mastercard, American Express, or Discover); coherent with `credit_card_number`
- `iban`: International Bank Account Number
- `currency_code`: currency code (USD, EUR, etc.)

**Identifiers:**

- `uuid4`: UUID version 4
- `md5`: MD5 hash (32 hex characters)
- `sha1`: SHA-1 hash (40 hex characters)
- `sha256`: SHA-256 hash (64 hex characters)
- `ssn`: Social Security Number (country-specific format)
- `license_plate`: vehicle license plate (location-aware for CA, US, DE, AU, GB)

**Barcodes:**

- `ean8`: EAN-8 barcode with valid check digit
- `ean13`: EAN-13 barcode with valid check digit

**Date/Time:**

- `date_this_year`: a date within the current year
- `date_this_decade`: a date within the current decade
- `date_between`: a random date between 2000 and 2025
- `date_range`: two dates joined with an en-dash (e.g., `"2012-05-12 - 2015-11-22"`)
- `future_date`: a date up to 1 year in the future
- `past_date`: a date up to 10 years in the past
- `time`: a time value

**Text:**

- `word`: single word
- `sentence`: full sentence
- `paragraph`: paragraph of text
- `text`: multiple paragraphs

**Miscellaneous:**

- `color_name`: color name
- `file_name`: file name
- `file_extension`: file extension
- `mime_type`: MIME type
- `user_agent`: browser user agent string (country-weighted)
- `locale_code`: locale identifier (e.g., `"en_US"`, `"de_DE"`; multilingual countries return a random official locale)

## Output Formats

The [generate_dataset()](../../reference/generate_dataset.md#pointblank.generate_dataset) function supports multiple output formats via the `output=` parameter, making it easy to integrate with your preferred data processing library.

``` python
schema = pb.Schema(
    id=pb.int_field(min_val=1),
    name=pb.string_field(preset="name"),
)
```

The default output is a Polars DataFrame, which offers excellent performance and a modern API for data manipulation:

``` python
polars_df = pb.generate_dataset(schema, n=100, seed=23, output="polars")

pb.preview(polars_df)
```

| PolarsRows100Columns2 |                     |                    |
|-----------------------|---------------------|--------------------|
|                       | id *Int64*          | name *String*      |
| 1                     | 5887552627537497772 | Doris Martin       |
| 2                     | 410844076336018751  | Nancy Gonzalez     |
| 3                     | 3891601934267714614 | Jessica Turner     |
| 4                     | 3397059975466237203 | George Evans       |
| 5                     | 7125830477412884969 | Patricia Williams  |
| 96                    | 3639168401353181645 | Isaiah Murphy      |
| 97                    | 5551865056303128007 | Brittany Rodriguez |
| 98                    | 5799605629891303422 | Megan Stevens      |
| 99                    | 2032001359082679845 | Pamela Jenkins     |
| 100                   | 7040735844921168140 | Stephanie Santos   |

If your workflow uses Pandas, simply specify `output="pandas"` to get a **Pandas DataFrame**:

``` python
pandas_df = pb.generate_dataset(schema, n=100, seed=23, output="pandas")

pb.preview(pandas_df)
```

| PandasRows100Columns2 |                     |                    |
|-----------------------|---------------------|--------------------|
|                       | id *int64*          | name *str*         |
| 1                     | 5887552627537497772 | Doris Martin       |
| 2                     | 410844076336018751  | Nancy Gonzalez     |
| 3                     | 3891601934267714614 | Jessica Turner     |
| 4                     | 3397059975466237203 | George Evans       |
| 5                     | 7125830477412884969 | Patricia Williams  |
| 96                    | 3639168401353181645 | Isaiah Murphy      |
| 97                    | 5551865056303128007 | Brittany Rodriguez |
| 98                    | 5799605629891303422 | Megan Stevens      |
| 99                    | 2032001359082679845 | Pamela Jenkins     |
| 100                   | 7040735844921168140 | Stephanie Santos   |

Both formats work seamlessly with Pointblank's validation functions, so you can choose whichever fits best with your existing data pipeline.

## Using Generated Data for Validation Testing

A common use case is generating test data to validate your validation rules:

``` python
# Define a schema with constraints
schema = pb.Schema(
    user_id=pb.int_field(min_val=1, unique=True),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=100),
    status=pb.string_field(allowed=["active", "pending", "inactive"]),
)

# Generate test data
test_data = pb.generate_dataset(schema, n=100, seed=23)

# Validate the generated data (it should pass all checks)
validation = (
    pb.Validate(test_data)
    .col_vals_gt("user_id", 0)
    .col_vals_regex("email", r".+@.+\..+")
    .col_vals_between("age", 18, 100)
    .col_vals_in_set("status", ["active", "pending", "inactive"])
    .interrogate()
)

validation
```

[TABLE]

Since the generated data respects the constraints defined in the schema, it should pass all validation checks. This workflow is particularly useful for testing validation logic before applying it to production data, or for creating reproducible test fixtures in your CI/CD pipeline.

## Conclusion

Test data generation provides a convenient way to create realistic synthetic datasets directly from schema definitions. While the concept is straightforward (defining field types and constraints, then generating matching data), the feature can be invaluable in many development and testing workflows. By incorporating test data generation into your process, you can:

- quickly prototype validation rules before working with production data
- create reproducible test fixtures for automated testing and CI/CD pipelines
- generate locale-specific data for internationalization testing across 100 countries
- ensure coherent relationships between related fields like names, emails, addresses, jobs, and license plates
- produce datasets of any size with consistent, realistic values

Whether you're building validation logic, testing data pipelines, or simply need sample data for development, the schema-based generation approach gives you precise control over data characteristics while maintaining the realism needed to uncover edge cases and validate your assumptions about data quality.
