# Test Data Generation

Pointblank provides a built-in test data generation system that creates realistic, locale-aware synthetic data based on schema definitions. This is useful for testing validation rules, creating sample datasets, and generating fixture data for development.

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
| [date_field()](../../reference/date_field.md#pointblank.date_field) | Date columns | `min_val`, `max_val` |
| [datetime_field()](../../reference/datetime_field.md#pointblank.datetime_field) | Datetime columns | `min_val`, `max_val` |
| [time_field()](../../reference/time_field.md#pointblank.time_field) | Time columns | `min_val`, `max_val` |
| [duration_field()](../../reference/duration_field.md#pointblank.duration_field) | Duration columns | `min_val`, `max_val` |
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

## Profile Fields

When generating person-profile data, you often need several related presets together: a name, an email derived from that name, an address, a phone number, and so on. Rather than wiring up each column individually, the [profile_fields()](../../reference/profile_fields.md#pointblank.profile_fields) helper returns a ready-made dictionary of `StringField` objects that you can unpack directly into a [Schema()](../../reference/Schema.md#pointblank.Schema).

### Basic Usage

With no arguments, [profile_fields()](../../reference/profile_fields.md#pointblank.profile_fields) returns the **standard** set of seven columns: `first_name`, `last_name`, `email`, `city`, `state`, `postcode`, and `phone_number`. All coherence rules apply automatically: emails are derived from names, and city/state/postcode/phone are internally consistent.

``` python
schema = pb.Schema(
    user_id=pb.int_field(unique=True, min_val=1),
    **pb.profile_fields(),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | user_id *Int64* | first_name *String* | last_name *String* | email *String* | city *String* | state *String* | postcode *String* | phone_number *String* |
| 1 | 7188536481533917197 | Weston | Parker | weston.parker23@gmail.com | Lubbock | Texas | 79404 | \(832\) 960-5399 |
| 2 | 2674009078779859984 | Hazel | Torres | hazel723@hotmail.com | Anaheim | California | 92873 | \(805\) 988-7427 |
| 3 | 7652102777077138151 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | Phoenix | Arizona | 85027 | \(928\) 782-5894 |
| 4 | 157503859921753049 | Maria | Garcia | m_garcia@hotmail.com | Denver | Colorado | 80277 | \(303\) 846-6634 |
| 5 | 2829213282471975080 | Michael | Hoffman | michael.hoffman@gmail.com | San Antonio | Texas | 78208 | \(214\) 901-0009 |
| 96 | 7027508096731143831 | Daniel | Torres | daniel_torres@icloud.com | El Paso | Texas | 79944 | \(281\) 769-2210 |
| 97 | 6055996548456656575 | Helen | Simpson | hsimpson20@yandex.com | El Paso | Texas | 79930 | \(210\) 345-8803 |
| 98 | 3822709996092631588 | Mark | Graham | mark.graham65@mail.com | Charlotte | North Carolina | 28222 | \(336\) 732-4609 |
| 99 | 1522653102058131295 | Brian | Moore | bmoore95@zoho.com | Los Angeles | California | 90058 | \(707\) 702-7905 |
| 100 | 5690877051669225499 | Michael | Ward | michael_ward@yahoo.com | San Diego | California | 92147 | \(619\) 940-2614 |

The `**` operator unpacks the dictionary into keyword arguments, as if you had written each `string_field(preset=...)` call by hand.

### Choosing a Set

Three built-in sets control how many columns are generated:

| Set | Columns |
|----|----|
| `"minimal"` | `first_name`, `last_name`, `email`, `phone_number` |
| `"standard"` | `first_name`, `last_name`, `email`, `city`, `state`, `postcode`, `phone_number` |
| `"full"` | `first_name`, `last_name`, `email`, `address`, `city`, `state`, `postcode`, `phone_number`, `company`, `job` |

``` python
# Minimal profile: just name, email, and phone
pb.preview(
    pb.generate_dataset(
        pb.Schema(**pb.profile_fields(set="minimal")),
        n=100, seed=23,
    )
)
```

| PolarsRows100Columns4 |  |  |  |  |
|----|----|----|----|----|
|  | first_name *String* | last_name *String* | email *String* | phone_number *String* |
| 1 | Weston | Parker | weston.parker23@gmail.com | \(214\) 673-2777 |
| 2 | Hazel | Torres | hazel723@hotmail.com | \(213\) 828-0231 |
| 3 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | \(623\) 370-0982 |
| 4 | Maria | Garcia | m_garcia@hotmail.com | \(970\) 694-0476 |
| 5 | Michael | Hoffman | michael.hoffman@gmail.com | \(682\) 391-2143 |
| 96 | Daniel | Torres | daniel_torres@icloud.com | \(915\) 504-6717 |
| 97 | Helen | Simpson | hsimpson20@yandex.com | \(940\) 869-0705 |
| 98 | Mark | Graham | mark.graham65@mail.com | \(984\) 800-0885 |
| 99 | Brian | Moore | bmoore95@zoho.com | \(650\) 955-2724 |
| 100 | Michael | Ward | michael_ward@yahoo.com | \(949\) 771-2134 |

``` python
# Full profile: includes address, company, and job title
pb.preview(
    pb.generate_dataset(
        pb.Schema(**pb.profile_fields(set="full")),
        n=100, seed=23,
    )
)
```

| PolarsRows100Columns10 |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
|  | first_name *String* | last_name *String* | email *String* | address *String* | city *String* | state *String* | postcode *String* | phone_number *String* | company *String* | job *String* |
| 1 | Weston | Parker | wparker89@outlook.com | 3365 Richmond Avenue, Suite 422, Lubbock, Texas 79421 | Lubbock | Texas | 79470 | \(210\) 460-3732 | Patterson Networks | System Administrator |
| 2 | Hazel | Torres | htorres35@aol.com | 353 East South Street, Anaheim, California 92898 | Anaheim | California | 92835 | \(858\) 514-0644 | Anaheim Real Estate | Insurance Agent |
| 3 | Lawrence | Mitchell | lawrence.mitchell1@hotmail.com | 2647 Tatum Boulevard, Phoenix, Arizona 85023 | Phoenix | Arizona | 85041 | \(520\) 301-6749 | Ford | Interior Designer |
| 4 | Maria | Garcia | mgarcia@aol.com | 5145 Arizona Avenue, Apt 761, Denver, Colorado 80255 | Denver | Colorado | 80287 | \(970\) 458-9509 | Carter Electric | HVAC Technician |
| 5 | Michael | Hoffman | m_hoffman@outlook.com | 5209 Grayson Street, Suite 333, San Antonio, Texas 78299 | San Antonio | Texas | 78248 | \(512\) 529-7287 | San Antonio Engineering Group | Construction Manager |
| 96 | Daniel | Torres | daniel.torres48@icloud.com | 3678 Rim Road, Suite 33, El Paso, Texas 79943 | El Paso | Texas | 79962 | \(972\) 892-8539 | Pioneer Strategy Group | Consultant |
| 97 | Helen | Simpson | helen.simpson1@hotmail.com | 1411 Mesa Street, El Paso, Texas 79930 | El Paso | Texas | 79955 | \(512\) 481-0603 | Powell & Gordon | Electrical Engineer |
| 98 | Mark | Graham | mgraham26@mail.com | 2065 Craighead Road, Apt 521, Charlotte, North Carolina 28289 | Charlotte | North Carolina | 28261 | \(980\) 608-7729 | Graham & Allen | Mechanical Engineer |
| 99 | Brian | Moore | brian263@icloud.com | 2072 2nd Street Los Angeles, Suite 893, Los Angeles, California 90034 | Los Angeles | California | 90037 | \(408\) 625-2327 | The Quality Hotels | Chef |
| 100 | Michael | Ward | m_ward@protonmail.com | 7966 K Street, San Diego, California 92109 | San Diego | California | 92194 | \(510\) 252-2820 | Morgan Stanley | Bank Teller |

### Combined vs. Split Names

By default, names are split into `first_name` and `last_name` columns. Set `split_name=False` to get a single `name` column instead:

``` python
pb.preview(
    pb.generate_dataset(
        pb.Schema(**pb.profile_fields(set="minimal", split_name=False)),
        n=100, seed=23,
    )
)
```

| PolarsRows100Columns3 |  |  |  |
|----|----|----|----|
|  | name *String* | email *String* | phone_number *String* |
| 1 | Weston Parker | weston.parker23@gmail.com | \(214\) 673-2777 |
| 2 | Hazel Torres | hazel723@hotmail.com | \(213\) 828-0231 |
| 3 | Lawrence Mitchell | lawrence_mitchell@zoho.com | \(623\) 370-0982 |
| 4 | Maria Garcia | m_garcia@hotmail.com | \(970\) 694-0476 |
| 5 | Michael Hoffman | michael.hoffman@gmail.com | \(682\) 391-2143 |
| 96 | Daniel Torres | daniel_torres@icloud.com | \(915\) 504-6717 |
| 97 | Helen Simpson | hsimpson20@yandex.com | \(940\) 869-0705 |
| 98 | Mark Graham | mark.graham65@mail.com | \(984\) 800-0885 |
| 99 | Brian Moore | bmoore95@zoho.com | \(650\) 955-2724 |
| 100 | Michael Ward | michael_ward@yahoo.com | \(949\) 771-2134 |

### Adding and Removing Columns

Use `include=` to add presets to the base set and `exclude=` to remove them. Both accept lists of preset names. The available profile presets are: `first_name`, `last_name`, `name`, `email`, `address`, `city`, `state`, `postcode`, `phone_number`, `company`, and `job`.

``` python
# Standard set + company column
pb.preview(
    pb.generate_dataset(
        pb.Schema(**pb.profile_fields(include=["company"])),
        n=100, seed=23,
    )
)
```

| PolarsRows100Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | first_name *String* | last_name *String* | email *String* | city *String* | state *String* | postcode *String* | phone_number *String* | company *String* |
| 1 | Weston | Parker | weston.parker23@gmail.com | Lubbock | Texas | 79404 | \(832\) 960-5399 | Innovative Systems Solutions |
| 2 | Hazel | Torres | hazel723@hotmail.com | Anaheim | California | 92873 | \(805\) 988-7427 | Sterling Engineering |
| 3 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | Phoenix | Arizona | 85027 | \(928\) 782-5894 | Goldman Sachs |
| 4 | Maria | Garcia | m_garcia@hotmail.com | Denver | Colorado | 80277 | \(303\) 846-6634 | Evans Group |
| 5 | Michael | Hoffman | michael.hoffman@gmail.com | San Antonio | Texas | 78208 | \(214\) 901-0009 | Goodwin and Garrett |
| 96 | Daniel | Torres | daniel_torres@icloud.com | El Paso | Texas | 79944 | \(281\) 769-2210 | Henry Construction |
| 97 | Helen | Simpson | hsimpson20@yandex.com | El Paso | Texas | 79930 | \(210\) 345-8803 | Thompson Technologies |
| 98 | Mark | Graham | mark.graham65@mail.com | Charlotte | North Carolina | 28222 | \(336\) 732-4609 | Universal Consulting |
| 99 | Brian | Moore | bmoore95@zoho.com | Los Angeles | California | 90058 | \(707\) 702-7905 | Long Industries |
| 100 | Michael | Ward | michael_ward@yahoo.com | San Diego | California | 92147 | \(619\) 940-2614 | Pioneer Solutions |

``` python
# Standard set without city and state
pb.preview(
    pb.generate_dataset(
        pb.Schema(**pb.profile_fields(exclude=["city", "state"])),
        n=100, seed=23,
    )
)
```

| PolarsRows100Columns5 |  |  |  |  |  |
|----|----|----|----|----|----|
|  | first_name *String* | last_name *String* | email *String* | postcode *String* | phone_number *String* |
| 1 | Weston | Parker | weston.parker23@gmail.com | 79404 | \(832\) 960-5399 |
| 2 | Hazel | Torres | hazel723@hotmail.com | 92873 | \(805\) 988-7427 |
| 3 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | 85027 | \(928\) 782-5894 |
| 4 | Maria | Garcia | m_garcia@hotmail.com | 80277 | \(303\) 846-6634 |
| 5 | Michael | Hoffman | michael.hoffman@gmail.com | 78208 | \(214\) 901-0009 |
| 96 | Daniel | Torres | daniel_torres@icloud.com | 79944 | \(281\) 769-2210 |
| 97 | Helen | Simpson | hsimpson20@yandex.com | 79930 | \(210\) 345-8803 |
| 98 | Mark | Graham | mark.graham65@mail.com | 28222 | \(336\) 732-4609 |
| 99 | Brian | Moore | bmoore95@zoho.com | 90058 | \(707\) 702-7905 |
| 100 | Michael | Ward | michael_ward@yahoo.com | 92147 | \(619\) 940-2614 |

You can combine `include=` and `exclude=` in the same call, as long as the same preset does not appear in both.

### Column Prefixes

The `prefix=` parameter prepends a string to every column name. This is especially useful when a schema needs two independent profiles (e.g., sender and recipient):

``` python
schema = pb.Schema(
    **pb.profile_fields(set="minimal", prefix="sender_"),
    **pb.profile_fields(set="minimal", prefix="recipient_"),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23))
```

| PolarsRows100Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | sender_first_name *String* | sender_last_name *String* | sender_email *String* | sender_phone_number *String* | recipient_first_name *String* | recipient_last_name *String* | recipient_email *String* | recipient_phone_number *String* |
| 1 | Weston | Parker | weston.parker23@gmail.com | \(214\) 673-2777 | Weston | Parker | weston.parker@outlook.com | \(915\) 827-6385 |
| 2 | Hazel | Torres | hazel723@hotmail.com | \(213\) 828-0231 | Hazel | Torres | hazeltorres@yandex.com | \(510\) 430-0382 |
| 3 | Lawrence | Mitchell | lawrence_mitchell@zoho.com | \(623\) 370-0982 | Lawrence | Mitchell | lmitchell@hotmail.com | \(480\) 609-6977 |
| 4 | Maria | Garcia | m_garcia@hotmail.com | \(970\) 694-0476 | Maria | Garcia | maria993@zoho.com | \(303\) 564-2441 |
| 5 | Michael | Hoffman | michael.hoffman@gmail.com | \(682\) 391-2143 | Michael | Hoffman | michael.hoffman84@gmail.com | \(940\) 486-1122 |
| 96 | Daniel | Torres | daniel_torres@icloud.com | \(915\) 504-6717 | Daniel | Torres | daniel.torres@gmail.com | \(832\) 313-0333 |
| 97 | Helen | Simpson | hsimpson20@yandex.com | \(940\) 869-0705 | Helen | Simpson | helensimpson@gmail.com | \(512\) 818-5803 |
| 98 | Mark | Graham | mark.graham65@mail.com | \(984\) 800-0885 | Mark | Graham | markgraham@gmail.com | \(984\) 656-7349 |
| 99 | Brian | Moore | bmoore95@zoho.com | \(650\) 955-2724 | Brian | Moore | bmoore93@yahoo.com | \(408\) 902-9175 |
| 100 | Michael | Ward | michael_ward@yahoo.com | \(949\) 771-2134 | Michael | Ward | michael.ward4@hotmail.com | \(650\) 916-0907 |

Each prefixed group maintains its own coherence: the sender's email is derived from the sender's name, and the recipient's email from the recipient's name.

### Combining with Other Field Types

Since [profile_fields()](../../reference/profile_fields.md#pointblank.profile_fields) returns a plain dictionary, it composes naturally with any other field types:

``` python
schema = pb.Schema(
    id=pb.int_field(unique=True, min_val=1000),
    **pb.profile_fields(),
    active=pb.bool_field(p_true=0.8),
    signup_date=pb.date_field(
        min_date="2024-01-01",
        max_date="2025-12-31",
    ),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23, country="DE"))
```

| PolarsRows100Columns10 |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
|  | id *Int64* | first_name *String* | last_name *String* | email *String* | city *String* | state *String* | postcode *String* | phone_number *String* | active *Boolean* | signup_date *Date* |
| 1 | 7188536481533918196 | Agathe | Kramer | akramer@gmail.com | Potsdam | Brandenburg | 14820 | \(0335\) 989-6751 | False | 2024-10-23 |
| 2 | 2674009078779860983 | Rüdiger | Altmann | ruediger.altmann69@gmail.com | Halle (Saale) | Sachsen-Anhalt | 06210 | \(0345\) 239-7780 | False | 2024-03-26 |
| 3 | 7652102777077139150 | Bodo | Meyer | bodo723@web.de | Frankfurt am Main | Hessen | 60473 | \(0611\) 231-3262 | False | 2024-01-18 |
| 4 | 157503859921754048 | Stefanie | Meyer | stefanie_meyer@posteo.de | Leipzig | Sachsen | 04277 | \(0371\) 874-7863 | True | 2025-08-29 |
| 5 | 2829213282471976079 | Fabian | Thomas | f_thomas@web.de | Cologne | Nordrhein-Westfalen | 50708 | \(0203\) 870-5463 | True | 2024-11-10 |
| 96 | 7027508096731144830 | Maike | Fuchs | maike_fuchs@yahoo.de | Frankfurt am Main | Hessen | 60188 | \(0611\) 017-4531 | True | 2024-06-20 |
| 97 | 6055996548456657574 | Sebastian | König | sebastian_koenig@t-online.de | Augsburg | Bayern | 86110 | \(0841\) 916-6870 | True | 2025-04-05 |
| 98 | 3822709996092632587 | Diana | Henkel | dhenkel20@arcor.de | Frankfurt am Main | Hessen | 60198 | \(06151\) 561-5880 | False | 2024-02-04 |
| 99 | 1522653102058132294 | Matthias | Keßler | matthias.kessler65@outlook.de | Zwickau | Sachsen | 08826 | \(0351\) 850-7507 | False | 2024-02-27 |
| 100 | 5690877051669226498 | Dirk | Neumann | dneumann95@posteo.de | Nuremberg | Bayern | 90954 | \(0941\) 885-4226 | False | 2025-10-07 |

## Country-Specific Data

One of the most powerful features is generating locale-aware data. Use the `country=` parameter to generate data specific to a country. This affects names, cities, addresses, and other locale-sensitive presets.

Let's create a schema that includes several location-related fields. When generating data for a specific country, Pointblank ensures *consistency across related fields*. The city, address, postcode, and coordinates will all correspond to the same location:

``` python
# Schema with linked location fields
schema = pb.Schema(
    name=pb.string_field(preset="name"),
    city=pb.string_field(preset="city"),
    address=pb.string_field(preset="address"),
    postcode=pb.string_field(preset="postcode"),
    latitude=pb.string_field(preset="latitude"),
    longitude=pb.string_field(preset="longitude"),
)
```

Here's German data with authentic names and addresses from cities like Berlin, Munich, and Hamburg. Notice how the latitude/longitude coordinates match real locations in Germany:

``` python
pb.preview(pb.generate_dataset(schema, n=200, seed=23, country="DE"))
```

| PolarsRows200Columns6 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | name *String* | city *String* | address *String* | postcode *String* | latitude *String* | longitude *String* |
| 1 | Niklas Schulte | Potsdam | Jägertor 8211, Whg. 737, 14097 Potsdam | 14448 | 52.428914 | 13.064566 |
| 2 | Erik Becker | Halle (Saale) | Hansering 9509, Whg. 636, 06678 Halle (Saale) | 06101 | 51.484572 | 11.937119 |
| 3 | Marco Albrecht | Frankfurt am Main | Friedensstraße 9713, Whg. 474, 60597 Frankfurt am Main | 60674 | 50.212245 | 8.711472 |
| 4 | Juliane Münz | Leipzig | Lindenauer Markt 6249, Whg. 489, 04541 Leipzig | 04992 | 51.276862 | 12.458890 |
| 5 | Anton Baumann | Cologne | Aachener Straße 7203, 50125 Köln | 50589 | 50.967264 | 6.795838 |
| 196 | Franziska Wendt | Ulm | Marktplatz 6251, Whg. 535, 89984 Ulm | 89226 | 48.395296 | 10.001962 |
| 197 | Lennart Berger | Munich | Brienner Straße 1390, Whg. 389, 80255 München | 80835 | 48.206882 | 11.674262 |
| 198 | Julia Knecht | Ludwigshafen am Rhein | Friedrichstraße 3204, 67944 Ludwigshafen am Rhein | 67305 | 49.473668 | 8.437782 |
| 199 | Sebastian Thiel | Gelsenkirchen | Husemannstraße 453, Whg. 273, 45732 Gelsenkirchen | 45992 | 51.568689 | 7.082531 |
| 200 | Trude Kaiser | Kassel | Königstraße 1394, 34406 Kassel | 34736 | 51.326544 | 9.494319 |

Japanese data includes names in romanized form and addresses from cities like Tokyo, Osaka, and Kyoto. The coordinates fall within Japan's geographic boundaries:

``` python
pb.preview(pb.generate_dataset(schema, n=200, seed=23, country="JP"))
```

| PolarsRows200Columns6 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | name *String* | city *String* | address *String* | postcode *String* | latitude *String* | longitude *String* |
| 1 | Kenji Ozawa | Kasuga | 816-3132 Fukuoka Kasuga Shirane 2869-411 | 816-5387 | 33.532717 | 130.479907 |
| 2 | Yuki Sugimoto | Higashihiroshima | 739-7478 Hiroshima Higashihiroshima Kouchi-dori 5279-446 | 739-6297 | 34.402366 | 132.767214 |
| 3 | Osamu Takahashi | Fukuoka | 810-8705 Fukuoka Fukuoka Jonan-dori 4932-302 | 810-3425 | 33.676810 | 130.414218 |
| 4 | Yuki Sakai | Saitama | 330-3255 Saitama Saitama Minuma-dori 3157 | 330-4163 | 35.927124 | 139.649673 |
| 5 | Haruka Yamazaki | Kitakyushu | 802-7756 Fukuoka Kitakyushu Yamada-dori 281 | 802-7878 | 33.908803 | 130.911548 |
| 196 | Mayumi Takahashi | Fukuoka | 810-7017 Fukuoka Fukuoka Imajuku-dori 9700 | 810-1106 | 33.558128 | 130.415678 |
| 197 | Katsuya Kawamura | Hiroshima | 730-1406 Hiroshima Hiroshima Ujina-dori 281 | 730-6070 | 34.398047 | 132.440192 |
| 198 | Kazue Nakata | Sendai | 980-1935 Miyagi Sendai Kawauchi 1182 | 980-1544 | 38.255431 | 140.876065 |
| 199 | Nozomi Shiraishi | Matsudo | 271-5711 Chiba Matsudo Akiyama-dori 6082-279 | 271-5161 | 35.818062 | 139.882260 |
| 200 | Ayaka Asano | Kure | 737-9841 Hiroshima Kure Tsukiji-dori 3763 | 737-7152 | 34.261593 | 132.556836 |

Brazilian data features Portuguese names and addresses from cities like São Paulo, Rio de Janeiro, and Brasília. The postal codes follow Brazil's CEP format:

``` python
pb.preview(pb.generate_dataset(schema, n=200, seed=23, country="BR"))
```

| PolarsRows200Columns6 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | name *String* | city *String* | address *String* | postcode *String* | latitude *String* | longitude *String* |
| 1 | Fábio Santos | Campinas | Avenida das Amoreiras, 1624, Apto 677, 13239-778 Campinas - SP | 13441-282 | -22.925008 | -47.075037 |
| 2 | João Vieira | Porto Alegre | Rua Vasco da Gama, 2869, 90313-262 Porto Alegre - RS | 90439-595 | -30.006369 | -51.232939 |
| 3 | Letícia Costa | Rio de Janeiro | Avenida Maracanã, 4893, Apto 709, 20863-487 Rio de Janeiro - RJ | 20351-144 | -22.881665 | -43.185310 |
| 4 | Regina Ferreira | Belo Horizonte | Rua Guajajaras, 5690, 30633-255 Belo Horizonte - MG | 30803-422 | -19.962432 | -43.907671 |
| 5 | Francisco Rodrigues | Belo Horizonte | Rua Turquesa, 281, Apto 515, 30775-668 Belo Horizonte - MG | 30822-683 | -19.958575 | -43.936621 |
| 196 | Júlia Martins | Sao Paulo | Rua São Bento, 5410, 01605-295 São Paulo - SP | 01655-287 | -23.597987 | -46.673967 |
| 197 | Júlia Moreira | Recife | Rua do Giriquiti, 7340, 50146-087 Recife - PE | 50893-805 | -8.025160 | -34.880996 |
| 198 | Rodrigo Moura | Salvador | Rua Francisco Muniz Barreto, 5926, 40584-681 Salvador - BA | 40442-463 | -12.921934 | -38.521806 |
| 199 | Iraci Lima | Sao Paulo | Rua Celso Garcia, 3181, Apto 872, 01417-237 São Paulo - SP | 01650-013 | -23.506813 | -46.649849 |
| 200 | Bianca Medeiros | Curitiba | Rua Marechal Hermes, 3301, Apto 281, 80229-139 Curitiba - PR | 80703-569 | -25.477946 | -49.278213 |

This location coherence is valuable when testing geospatial applications, address validation systems, or any scenario where realistic, internally-consistent location data matters.

### Data Coherence

Pointblank automatically links related columns to produce realistic rows. There are three coherence systems that activate based on which presets appear together in a schema:

**Address coherence** activates when *any* address-related preset is present (`address`, `city`, `state`, `postcode`, `latitude`, `longitude`, `phone_number`, `license_plate`). All of these fields will refer to the same location within each row.

**Person coherence** activates when *any* person-related preset is present (`name`, `name_full`, `first_name`, `last_name`, `email`, `user_name`). The email and username are derived from the person's name.

**Business coherence** activates when *both* `job` and `company` are present. When active:

- the company and job title are drawn from the same industry (e.g., a nurse will work at a hospital, not a law firm).
- `name_full` gains profession-matched titles: a doctor may appear as "Dr. Ana Sousa" and a professor as "Prof. Tanaka Yuki". For German-speaking countries (DE, AT, CH), the honorific stacks before the professional title (e.g., "Herr Dr. med. Klaus Weber").
- integer columns whose name contains `age` (e.g., `age`, `person_age`) are automatically constrained to working-age range (22-65).

Here's an example showing all three coherence systems working together:

``` python
schema = pb.Schema(
    name=pb.string_field(preset="name_full"),
    email=pb.string_field(preset="email"),
    company=pb.string_field(preset="company"),
    job=pb.string_field(preset="job"),
    city=pb.string_field(preset="city"),
    state=pb.string_field(preset="state"),
    license_plate=pb.string_field(preset="license_plate"),
    age=pb.int_field(),
)

pb.preview(pb.generate_dataset(schema, n=100, seed=23, country="DE"))
```

| PolarsRows100Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | name *String* | email *String* | company *String* | job *String* | city *String* | state *String* | license_plate *String* | age *Int64* |
| 1 | Frau Agathe Kramer | agathekramer@mail.de | Berliner Systeme Software | Produktmanager | Potsdam | Brandenburg | P-ZL 931 | 40 |
| 2 | Herr Rüdiger Altmann | ruediger.altmann@mail.de | KPMG | Business Analyst | Halle (Saale) | Sachsen-Anhalt | HAL-D 6183 | 27 |
| 3 | Herr Bodo Meyer | b_meyer@outlook.de | Internationale Strom Netzwerke | Systemadministrator | Frankfurt am Main | Hessen | F-DZ 0091 | 23 |
| 4 | Frau Stefanie Meyer | stefanie.meyer@posteo.de | Frankfurter Sicherheit | Cloud-Architekt | Leipzig | Sachsen | L-X 1095 | 59 |
| 5 | Herr Fabian Thomas | f_thomas@freenet.de | Scherer Immobilien | Immobilienmakler | Cologne | Nordrhein-Westfalen | K-P 299 | 41 |
| 96 | Frau Prof. Maike Fuchs | maike.fuchs37@arcor.de | Technische Universität Frankfurt am Main | Professor | Frankfurt am Main | Hessen | F-A 2325 | 24 |
| 97 | Herr Sebastian König | sebastiankoenig@web.de | Premium Kreativ | Grafikdesigner | Augsburg | Bayern | A-MY 8326 | 25 |
| 98 | Frau Diana Henkel | diana990@web.de | Sparkasse Frankfurt am Main | Finanzanalyst | Frankfurt am Main | Hessen | F-F 1377 | 62 |
| 99 | Herr Matthias Keßler | matthiaskessler@yahoo.de | Schrader Gruppe | Grafikdesigner | Zwickau | Sachsen | Z-W 569 | 48 |
| 100 | Herr Dirk Neumann | dirk_neumann@arcor.de | Deutsche Software Digital | Produktmanager | Nuremberg | Bayern | N-K 1620 | 47 |

**License plate coherence** is part of address coherence. For `CA`, `US`, `DE`, `AU`, and `GB`, license plates follow real subregion-specific formats when location fields are present. For example, an Ontario row produces plates like `"CABC 123"` while a British Columbia row produces `"AB1 23C"`. Letters I, O, Q, and U are excluded from plate generation, matching real-world restrictions.

### Supported Countries

Pointblank currently supports 100 countries with full locale data for realistic test data generation. You can use either ISO 3166-1 alpha-2 codes (e.g., `"US"`) or alpha-3 codes (e.g., `"USA"`).

**Europe (38 countries):**

- Armenia (`AM`), Austria (`AT`), Azerbaijan (`AZ`), Belgium (`BE`), Bulgaria (`BG`), Croatia (`HR`), Cyprus (`CY`), Czech Republic (`CZ`), Denmark (`DK`), Estonia (`EE`), Finland (`FI`), France (`FR`), Georgia (`GE`), Germany (`DE`), Greece (`GR`), Hungary (`HU`), Iceland (`IS`), Ireland (`IE`), Italy (`IT`), Latvia (`LV`), Lithuania (`LT`), Luxembourg (`LU`), Malta (`MT`), Moldova (`MD`), Netherlands (`NL`), Norway (`NO`), Poland (`PL`), Portugal (`PT`), Romania (`RO`), Russia (`RU`), Serbia (`RS`), Slovakia (`SK`), Slovenia (`SI`), Spain (`ES`), Sweden (`SE`), Switzerland (`CH`), Ukraine (`UA`), United Kingdom (`GB`)

**Americas (19 countries):**

- Argentina (`AR`), Bolivia (`BO`), Brazil (`BR`), Canada (`CA`), Chile (`CL`), Colombia (`CO`), Costa Rica (`CR`), Dominican Republic (`DO`), Ecuador (`EC`), El Salvador (`SV`), Guatemala (`GT`), Honduras (`HN`), Jamaica (`JM`), Mexico (`MX`), Panama (`PA`), Paraguay (`PY`), Peru (`PE`), United States (`US`), Uruguay (`UY`)

**Asia-Pacific (22 countries):**

- Australia (`AU`), Bangladesh (`BD`), Cambodia (`KH`), China (`CN`), Hong Kong (`HK`), India (`IN`), Indonesia (`ID`), Japan (`JP`), Kazakhstan (`KZ`), Malaysia (`MY`), Myanmar (`MM`), Nepal (`NP`), New Zealand (`NZ`), Pakistan (`PK`), Philippines (`PH`), Singapore (`SG`), South Korea (`KR`), Sri Lanka (`LK`), Taiwan (`TW`), Thailand (`TH`), Uzbekistan (`UZ`), Vietnam (`VN`)

**Middle East & Africa (21 countries):**

- Algeria (`DZ`), Cameroon (`CM`), Egypt (`EG`), Ethiopia (`ET`), Ghana (`GH`), Israel (`IL`), Jordan (`JO`), Kenya (`KE`), Lebanon (`LB`), Morocco (`MA`), Mozambique (`MZ`), Nigeria (`NG`), Rwanda (`RW`), Saudi Arabia (`SA`), Senegal (`SN`), South Africa (`ZA`), Tanzania (`TZ`), Tunisia (`TN`), Turkey (`TR`), Uganda (`UG`), United Arab Emirates (`AE`)

Additional countries and expanded coverage are planned for future releases.

### Mixing Multiple Countries

When you need test data that spans multiple locales (e.g., simulating an international customer base), you can pass a list or dict to the `country=` parameter instead of a single string.

Passing a list of country codes splits rows equally across those countries. Here, 200 rows are divided evenly among the US, Germany, and Japan (~67 each):

``` python
schema = pb.Schema(
    name=pb.string_field(preset="name"),
    city=pb.string_field(preset="city"),
    postcode=pb.string_field(preset="postcode"),
)

pb.preview(pb.generate_dataset(schema, n=200, seed=23, country=["US", "DE", "JP"]))
```

| PolarsRows200Columns3 |                   |               |                   |
|-----------------------|-------------------|---------------|-------------------|
|                       | name *String*     | city *String* | postcode *String* |
| 1                     | Sigrid Naumann    | Neumunster    | 24838             |
| 2                     | Shiori Sekiguchi  | Kanazawa      | 920-4517          |
| 3                     | Kayden Mitchell   | Columbus      | 31906             |
| 4                     | Carol Hudson      | San Antonio   | 78238             |
| 5                     | Jürgen Kern       | Dortmund      | 44376             |
| 196                   | Katsuya Fukuda    | Hamamatsu     | 430-7697          |
| 197                   | Isabella Scott    | Indianapolis  | 46276             |
| 198                   | Theodore Mcintyre | Los Angeles   | 90028             |
| 199                   | Nadine Schmitz    | Bonn          | 53176             |
| 200                   | Tomoya Yamazaki   | Saitama       | 330-8941          |

To control the proportion of rows per country, pass a dict mapping country codes to weights. The following generates 200 rows with 70% from the US, 20% from Germany, and 10% from France:

``` python
pb.preview(
    pb.generate_dataset(
        schema, n=200, seed=23,
        country={"US": 0.7, "DE": 0.2, "FR": 0.1},
    )
)
```

| PolarsRows200Columns3 |                 |               |                   |
|-----------------------|-----------------|---------------|-------------------|
|                       | name *String*   | city *String* | postcode *String* |
| 1                     | Carl Thompson   | Plano         | 75013             |
| 2                     | Falk Ackermann  | Leipzig       | 04566             |
| 3                     | Janet Flores    | Columbus      | 31994             |
| 4                     | Carol Brown     | San Antonio   | 78280             |
| 5                     | David Roberts   | Nashville     | 37248             |
| 196                   | Kerstin Schulze | Mainz         | 55063             |
| 197                   | Anthony Wright  | Indianapolis  | 46299             |
| 198                   | Bobby Hernandez | Los Angeles   | 90029             |
| 199                   | Donald Perry    | Phoenix       | 85030             |
| 200                   | Samuel Gaillard | Bordeaux      | 33032             |

Weights are auto-normalized, so `{"US": 7, "DE": 2, "FR": 1}` is equivalent to the example above. Row counts are allocated using largest-remainder apportionment, ensuring they always sum to exactly `n`.

By default, rows from different countries are interleaved randomly (`shuffle=True`). Set `shuffle=False` to keep rows grouped by country in the order the countries are listed:

``` python
pb.preview(
    pb.generate_dataset(
        schema, n=120, seed=23,
        country=["US", "DE", "JP"], shuffle=False,
    )
)
```

| PolarsRows120Columns3 |                 |               |                   |
|-----------------------|-----------------|---------------|-------------------|
|                       | name *String*   | city *String* | postcode *String* |
| 1                     | Weston Martinez | Houston       | 77047             |
| 2                     | Ruth Reynolds   | El Paso       | 79923             |
| 3                     | Marilyn Lane    | Indianapolis  | 46286             |
| 4                     | Rachel Black    | Detroit       | 48219             |
| 5                     | Laura Moore     | Indianapolis  | 46208             |
| 116                   | Tatsuya Maeda   | Chiba         | 260-0498          |
| 117                   | Takahiro Yasuda | Ishinomaki    | 986-1198          |
| 118                   | Koichi Miyasaka | Hachioji      | 192-4961          |
| 119                   | Ayaka Katayama  | Fukuchiyama   | 620-3866          |
| 120                   | Shinji Abe      | Niigata       | 950-2721          |

All coherence systems (address, person, business) work correctly within each country's batch of rows. A French row will have a French name with a matching French email; a Japanese row will have a Japanese name with a matching Japanese email. Non-preset columns (integers, floats, booleans, dates) are generated independently for each batch but still respect their field constraints.

### Frequency-Weighted Sampling

By default, names and cities are sampled uniformly at random from the locale data, giving every entry the same probability of being selected. Real-world distributions are far from uniform though: "James" and "Maria" appear orders of magnitude more often than "Thaddeus" or "Xiomara", and more people live in New York City than in Flagstaff. The `weighted=True` parameter makes generated data reflect this natural skew.

``` python
schema = pb.Schema(
    name=pb.string_field(preset="name"),
    city=pb.string_field(preset="city"),
)

pb.preview(pb.generate_dataset(schema, n=200, seed=23, country="US", weighted=True))
```

| PolarsRows200Columns2 |                    |               |
|-----------------------|--------------------|---------------|
|                       | name *String*      | city *String* |
| 1                     | Adam Knight        | Lubbock       |
| 2                     | Noah Gibson        | Anaheim       |
| 3                     | Jackson Morales    | Phoenix       |
| 4                     | Mark Wilson        | Denver        |
| 5                     | Maisie Perkins     | San Antonio   |
| 196                   | Daniel Woods       | Philadelphia  |
| 197                   | Christina Anderson | Los Angeles   |
| 198                   | Thea Woods         | Joliet        |
| 199                   | Anthony Campbell   | San Diego     |
| 200                   | Daniel Wagner      | Chicago       |

With weighting enabled you will see popular names like James, John, Mary, and Patricia appear more frequently, while unusual names surface only occasionally. Similarly, cities like New York, Los Angeles, and Chicago dominate the output while smaller cities appear less often.

The feature works by organizing locale data into four frequency tiers. Each tier has a sampling probability that determines how likely its members are to be selected:

| Tier        | Probability | Contents                                        |
|-------------|-------------|-------------------------------------------------|
| very_common | 45%         | The top ~10% of entries by real-world frequency |
| common      | 30%         | The next ~20% of entries                        |
| uncommon    | 20%         | The next ~30% of entries                        |
| rare        | 5%          | The remaining ~40% of entries                   |

When a value is needed, a tier is first chosen according to these probabilities and then a single entry is picked uniformly at random within that tier. This two-step approach keeps sampling fast while producing a realistic long-tail distribution. Setting `weighted=False` pools all entries across every tier and samples them uniformly, which can be useful when you want an even spread rather than a realistic distribution.

Weighted sampling combines seamlessly with multi-country mixing. Each country's batch uses its own tiered data independently, so a mixed dataset will have weighted US names alongside weighted German names:

``` python
pb.preview(
    pb.generate_dataset(
        schema,
        n=200,
        seed=23,
        country={"US": 0.6, "DE": 0.4},
        weighted=True,
    )
)
```

| PolarsRows200Columns2 |                |               |
|-----------------------|----------------|---------------|
|                       | name *String*  | city *String* |
| 1                     | Aurora Roberts | Plano         |
| 2                     | Petra Roth     | Mannheim      |
| 3                     | Ronald Cruz    | Columbus      |
| 4                     | Emma Lopez     | San Antonio   |
| 5                     | Steven Webb    | Nashville     |
| 196                   | Zoe Kaufmann   | Leipzig       |
| 197                   | John Ford      | Indianapolis  |
| 198                   | Sharon Barnes  | Los Angeles   |
| 199                   | Shirley Grant  | Phoenix       |
| 200                   | Charlotte Held | Cologne       |

All 100 supported country locales have tiered name and location data, so `weighted=True` produces realistic frequency distributions for every country.

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
|                       | id *int64*          | name *object*      |
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

## Pytest Fixture

When Pointblank is installed, a [generate_dataset](../../reference/generate_dataset.md#pointblank.generate_dataset) **pytest fixture** is automatically available in all your test files. There is no need to import anything or add configuration to `conftest.py`: the fixture is registered via pytest's plugin system.

The fixture works identically to `pb.generate_dataset()`, but with one key difference: when you don't supply a `seed=` parameter, a deterministic seed is automatically derived from the test's fully-qualified name. This means:

- the **same test** always produces the **same data**: no manual seed management required.
- *different tests* get different seeds, so they exercise different datasets.
- you can still pass an explicit `seed=` to override the automatic seed when needed.

### Basic Usage

Use it by adding [generate_dataset](../../reference/generate_dataset.md#pointblank.generate_dataset) to your test function's parameter list:

``` python
import pointblank as pb

def test_etl_handles_nulls(generate_dataset):
    schema = pb.Schema(
        user_id=pb.int_field(unique=True),
        email=pb.string_field(preset="email", nullable=True, null_probability=0.3),
        age=pb.int_field(min_val=0, max_val=120),
    )

    df = generate_dataset(schema, n=500)
    result = my_etl_pipeline(df)
    assert result.filter(pl.col("email").is_null()).shape[0] == 0
```

All parameters from [generate_dataset()](../../reference/generate_dataset.md#pointblank.generate_dataset) are supported: `n=`, `seed=`, `output=`, and `country=`:

``` python
def test_german_data(generate_dataset):
    schema = pb.Schema(
        name=pb.string_field(preset="name"),
        city=pb.string_field(preset="city"),
    )

    df = generate_dataset(schema, n=200, country="DE", output="pandas")
    assert len(df) == 200
```

### Multiple Datasets in One Test

Calling the fixture multiple times within the same test produces different (but still deterministic) data on each call:

``` python
def test_merge_pipeline(generate_dataset):
    customers = generate_dataset(customer_schema, n=1000, country="US")
    orders = generate_dataset(order_schema, n=5000)

    # Each call gets a unique seed derived from the test name + call index,
    # so both DataFrames are deterministic and different from each other.
    result = merge_pipeline(customers, orders)
    assert result.shape[0] > 0
```

### Testing Across Locales

The fixture makes locale testing particularly concise when combined with `pytest.mark.parametrize`:

``` python
import pytest
import pointblank as pb

@pytest.mark.parametrize("country", ["US", "DE", "JP", "BR"])
def test_name_normalizer(generate_dataset, country):
    schema = pb.Schema(name=pb.string_field(preset="name_full"))
    df = generate_dataset(schema, n=100, country=country)
    result = normalize_names(df)
    assert result["name"].str.len_chars().min() > 0
```

### Sharing Schemas Across Tests

Define schemas as fixtures in `conftest.py` and compose them with [generate_dataset](../../reference/generate_dataset.md#pointblank.generate_dataset):

``` python
import pytest
import pointblank as pb

@pytest.fixture
def customer_schema():
    return pb.Schema(
        id=pb.int_field(unique=True),
        name=pb.string_field(preset="name"),
        email=pb.string_field(preset="email"),
        city=pb.string_field(preset="city"),
    )
```

``` python
def test_customer_validation(generate_dataset, customer_schema):
    df = generate_dataset(customer_schema, n=200, country="DE")
    validation = pb.Validate(df).col_vals_not_null(columns="email").interrogate()
    assert validation.all_passed()
```

``` python
def test_customer_export(generate_dataset, customer_schema):
    df = generate_dataset(customer_schema, n=50, country="JP")
    exported = export_to_parquet(df)
    assert exported.exists()
```

### Debugging with Seed Introspection

The fixture callable exposes two attributes that make debugging failed tests straightforward:

- `generate_dataset.default_seed`: the base seed derived from the test name (available before any call)
- `generate_dataset.last_seed`: the seed actually used for the most recent call (accounts for the call counter and explicit overrides)

Include `.last_seed` in assertion messages so failures are immediately reproducible:

``` python
def test_age_range(generate_dataset):
    schema = pb.Schema(age=pb.int_field(min_val=18, max_val=100))
    df = generate_dataset(schema, n=500)
    min_age = df["age"].min()
    assert min_age >= 18, (
        f"Expected min age >= 18, got {min_age} (seed={generate_dataset.last_seed})"
    )
```

You can also use `.default_seed` to reproduce the exact dataset outside of pytest:

``` python
# In a REPL or notebook, reproduce the data from a failed test:
import pointblank as pb
df = pb.generate_dataset(schema, n=500, seed=<default_seed_from_output>)
```

### Seed Stability

A given seed (whether explicit or auto-derived) is guaranteed to produce identical output **within the same Pointblank version**. Across versions, changes to country data files or generator logic may alter the output for a given seed.

For CI pipelines that require bit-exact data across library upgrades, we recommend saving generated DataFrames as Parquet or CSV snapshot files rather than relying on cross-version seed stability. This is the same approach used by snapshot-testing tools like `pytest-snapshot` and `syrupy`.

## Conclusion

Test data generation provides a convenient way to create realistic synthetic datasets directly from schema definitions. While the concept is straightforward (defining field types and constraints, then generating matching data), the feature can be invaluable in many development and testing workflows. By incorporating test data generation into your process, you can:

- quickly prototype validation rules before working with production data
- create reproducible test fixtures for automated testing and CI/CD pipelines
- generate locale-specific data for internationalization testing across 100 countries
- ensure coherent relationships between related fields like names, emails, addresses, jobs, and license plates
- produce datasets of any size with consistent, realistic values

Whether you're building validation logic, testing data pipelines, or simply need sample data for development, the schema-based generation approach gives you precise control over data characteristics while maintaining the realism needed to uncover edge cases and validate your assumptions about data quality.
