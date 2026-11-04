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
| 1 | 7188536481533917197 | Doris Martin | d_martin@aol.com | 77 | pending |
| 2 | 2674009078779859984 | Nancy Gonzalez | nancygonzalez@icloud.com | 67 | active |
| 3 | 7652102777077138151 | Jessica Turner | jturner@aol.com | 78 | active |
| 4 | 157503859921753049 | George Evans | georgeevans@zoho.com | 36 | inactive |
| 5 | 2829213282471975080 | Patricia Williams | pwilliams@outlook.com | 75 | pending |
| 96 | 7027508096731143831 | Isaiah Murphy | isaiah.murphy@zoho.com | 55 | active |
| 97 | 6055996548456656575 | Brittany Rodriguez | brodriguez@yandex.com | 39 | inactive |
| 98 | 3822709996092631588 | Megan Stevens | mstevens26@aol.com | 24 | inactive |
| 99 | 1522653102058131295 | Pamela Jenkins | pjenkins29@yandex.com | 41 | active |
| 100 | 5690877051669225499 | Stephanie Santos | stephanie.santos40@gmail.com | 75 | pending |

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
| 1                     | 5749       | 100              | 3              |
| 2                     | 2368       | 38               | 1              |
| 3                     | 1279       | 11               | 1              |
| 4                     | 6025       | 3                | 5              |
| 5                     | 7942       | 76               | 3              |
| 96                    | 5330       | 64               | 2              |
| 97                    | 8634       | 31               | 1              |
| 98                    | 9982       | 43               | 2              |
| 99                    | 4221       | 70               | 1              |
| 100                   | 8520       | 19               | 5              |

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
| 1 | 924.8652516259452 | 0.4624326258129726 | 43.23787264633508 |
| 2 | 948.6057779931772 | 0.47430288899658857 | 45.37452001938594 |
| 3 | 892.4333440485793 | 0.44621667202428966 | 40.31900096437214 |
| 4 | 83.55067683068363 | 0.04177533841534181 | -32.48043908523847 |
| 5 | 592.0272268857353 | 0.29601361344286764 | 13.282450419716177 |
| 96 | 444.6925279641446 | 0.2223462639820723 | 0.022327516773010814 |
| 97 | 342.7762214585577 | 0.17138811072927884 | -9.150140068729808 |
| 98 | 892.3288689140903 | 0.4461644344570452 | 40.309598202268134 |
| 99 | 813.7559456012128 | 0.4068779728006064 | 33.238035104109144 |
| 100 | 895.1816604808429 | 0.44759083024042146 | 40.56634944327587 |

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
| 1 | Weston Parker | weston.parker23@gmail.com | Innovative Systems Solutions | Lubbock |
| 2 | Hazel Torres | hazel723@hotmail.com | Sterling Engineering | Anaheim |
| 3 | Lawrence Mitchell | lawrence_mitchell@zoho.com | Goldman Sachs | Phoenix |
| 4 | Maria Garcia | m_garcia@hotmail.com | Evans Group | Denver |
| 5 | Michael Hoffman | michael.hoffman@gmail.com | Goodwin and Garrett | San Antonio |
| 96 | Daniel Torres | daniel_torres@icloud.com | Henry Construction | El Paso |
| 97 | Helen Simpson | hsimpson20@yandex.com | Thompson Technologies | El Paso |
| 98 | Mark Graham | mark.graham65@mail.com | Universal Consulting | Charlotte |
| 99 | Brian Moore | bmoore95@zoho.com | Long Industries | Los Angeles |
| 100 | Michael Ward | michael_ward@yahoo.com | Pioneer Solutions | San Diego |

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
| 1 | CAS-6685 | \(109\) 668-2347 | \#209DCB |
| 2 | XGI-0397 | \(397\) 117-0865 | \#68E07E |
| 3 | DCW-6086 | \(309\) 293-9594 | \#32FD0D |
| 4 | YBG-9529 | \(917\) 797-2285 | \#161B56 |
| 5 | XLS-9459 | \(911\) 609-9495 | \#B9A2F5 |
| 96 | THG-2900 | \(993\) 511-5415 | \#A7A37B |
| 97 | CHC-3681 | \(065\) 802-0822 | \#47E498 |
| 98 | HKT-3552 | \(927\) 701-4276 | \#AF75D8 |
| 99 | OEW-4157 | \(365\) 419-1062 | \#5CCD95 |
| 100 | FSX-8948 | \(897\) 459-3038 | \#0F3220 |

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
| 1 | False | False | False |
| 2 | False | False | False |
| 3 | False | False | False |
| 4 | True | True | True |
| 5 | True | False | False |
| 96 | True | False | True |
| 97 | True | False | True |
| 98 | False | False | False |
| 99 | False | False | False |
| 100 | False | False | False |

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
| 1                     | 1986-01-03        | 2024-12-25 04:22:08   |
| 2                     | 1967-06-30        | 2024-10-29 16:22:23   |
| 3                     | 1961-07-13        | 2024-04-22 14:13:08   |
| 4                     | 1987-07-09        | 2024-12-12 14:04:53   |
| 5                     | 1998-01-06        | 2024-11-18 04:49:47   |
| 96                    | 1969-04-14        | 2024-07-29 13:15:44   |
| 97                    | 1975-03-23        | 2024-04-28 08:49:29   |
| 98                    | 1981-05-29        | 2024-12-13 09:42:37   |
| 99                    | 1982-09-14        | 2024-10-28 23:35:39   |
| 100                   | 1968-12-21        | 2024-06-25 14:22:27   |

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
| 1                     | 7188536481533917197 | Doris Martin       |
| 2                     | 2674009078779859984 | Nancy Gonzalez     |
| 3                     | 7652102777077138151 | Jessica Turner     |
| 4                     | 157503859921753049  | George Evans       |
| 5                     | 2829213282471975080 | Patricia Williams  |
| 96                    | 7027508096731143831 | Isaiah Murphy      |
| 97                    | 6055996548456656575 | Brittany Rodriguez |
| 98                    | 3822709996092631588 | Megan Stevens      |
| 99                    | 1522653102058131295 | Pamela Jenkins     |
| 100                   | 5690877051669225499 | Stephanie Santos   |

If your workflow uses Pandas, simply specify `output="pandas"` to get a **Pandas DataFrame**:

``` python
pandas_df = pb.generate_dataset(schema, n=100, seed=23, output="pandas")

pb.preview(pandas_df)
```

| PandasRows100Columns2 |                     |                    |
|-----------------------|---------------------|--------------------|
|                       | id *int64*          | name *str*         |
| 1                     | 7188536481533917197 | Doris Martin       |
| 2                     | 2674009078779859984 | Nancy Gonzalez     |
| 3                     | 7652102777077138151 | Jessica Turner     |
| 4                     | 157503859921753049  | George Evans       |
| 5                     | 2829213282471975080 | Patricia Williams  |
| 96                    | 7027508096731143831 | Isaiah Murphy      |
| 97                    | 6055996548456656575 | Brittany Rodriguez |
| 98                    | 3822709996092631588 | Megan Stevens      |
| 99                    | 1522653102058131295 | Pamela Jenkins     |
| 100                   | 5690877051669225499 | Stephanie Santos   |

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
