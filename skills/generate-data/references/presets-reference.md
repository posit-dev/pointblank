# String presets reference

Available preset values for `string_field(preset="...")`. The data
behind them ships with Pointblank (no Faker needed) and is
country-aware. An unknown preset raises a `ValueError` listing these.

The example outputs are for `country="US"`.

## Person and contact

| Preset          | Example output                      |
|-----------------|-------------------------------------|
| `name`          | "Kenneth Cole"                      |
| `name_full`     | "Mr. Kenneth Cole"                  |
| `first_name`    | "Kenneth"                           |
| `last_name`     | "Cole"                              |
| `gender`        | "male"                              |
| `email`         | "kenneth.cole50@mail.com"           |
| `phone_number`  | "(423) 886-9589"                    |
| `user_name`     | "kenneth_cole"                      |

## Location

| Preset           | Example output                                   |
|------------------|--------------------------------------------------|
| `address`        | "7705 Chapel Avenue, Suite 407, Nashville, Tennessee 37287" |
| `city`           | "Nashville"                                      |
| `state`          | "Tennessee"                                      |
| `postcode`       | "37287"                                          |
| `country`        | "United States"                                  |
| `country_code_2` | "US"                                             |
| `country_code_3` | "USA"                                            |
| `latitude`       | "36.218438"                                      |
| `longitude`      | "-86.968814"                                     |
| `locale_code`    | "en_US"                                          |

## Business

| Preset          | Example output                              |
|-----------------|---------------------------------------------|
| `company`       | "Classic Data"                              |
| `job`           | "Software Engineer"                         |
| `catch_phrase`  | "Seamless technologies that revolutionize"  |

## Internet

| Preset          | Example output                               |
|-----------------|----------------------------------------------|
| `url`           | "https://www.been.site"                      |
| `domain_name`   | "would.io"                                   |
| `ipv4`          | "120.233.157.90"                             |
| `ipv6`          | "c984:fe56:9e90:ef80:ee3e:4924:567f:e3a2"    |
| `password`      | "0EAIMIEDU7d2"                               |
| `user_agent`    | "Mozilla/5.0 (iPad; CPU OS 17_3_1 like ..."  |

## Identifiers and barcodes

| Preset          | Example output                           |
|-----------------|------------------------------------------|
| `uuid4`         | "fa9abf41-41cc-41a9-aeee-074eeb6b7fae"   |
| `md5`           | "249d006e264f848f313d204597f680d0"       |
| `sha1`          | "19d20679952d7d83593766d778d93c873850969a" |
| `sha256`        | "7dd89f77d239bf2f79ed44ef5372b8b4..."    |
| `ssn`           | "166-55-6405"                            |
| `license_plate` | "W72-51B"                                |
| `ean8`          | "12500925"                               |
| `ean13`         | "1171082961608"                          |

## Finance

| Preset                 | Example output              |
|------------------------|-----------------------------|
| `credit_card_number`   | "5232860129040479"          |
| `credit_card_provider` | "Mastercard"                |
| `iban`                 | "US52947179637462169994"    |
| `currency_code`        | "USD"                       |

## Text

| Preset          | Example output                                   |
|-----------------|--------------------------------------------------|
| `text`          | "His number all come then water two them ..."    |
| `sentence`      | "See their was long get she more two work."      |
| `paragraph`     | "Place which only point him might still ..."     |
| `word`          | "most"                                           |

## Dates and times (as strings)

| Preset             | Example output              |
|--------------------|-----------------------------|
| `date_this_year`   | "2026-06-28"                |
| `date_this_decade` | "2021-04-27"                |
| `date_between`     | "2025-03-17"                |
| `date_range`       | "2007-08-01 – 2010-02-22"   |
| `future_date`      | "2027-07-27"                |
| `past_date`        | "2020-07-28"                |
| `time`             | "07:09:32"                  |

For date and time columns with a real type, use `date_field()`,
`datetime_field()`, or `time_field()` instead.

## Files and misc

| Preset           | Example output        |
|------------------|-----------------------|
| `color_name`     | "Coral"               |
| `file_name`      | "look.tar"            |
| `file_extension` | "yaml"                |
| `mime_type`      | "video/x-matroska"    |

## Coherent rows

Related presets in one row agree with each other: the `email` and
`user_name` are built from the name, the `address`, `city`, `state`,
and `postcode` describe one place, and the credit card number matches
its provider. With `profile_fields(prefix=...)`, each prefix is a
separate person.

## Country-specific behavior

Presets produce locale-appropriate data based on the `country`
parameter passed to `generate()`:

```python
schema = pb.Schema(
    name=pb.string_field(preset="name"),
    city=pb.string_field(preset="city"),
    phone=pb.string_field(preset="phone_number"),
)

# US data
us_df = schema.generate(n=100, country="US")

# German data
de_df = schema.generate(n=100, country="DE")

# Japanese data
jp_df = schema.generate(n=100, country="JP")

# Several countries
mixed_df = schema.generate(n=100, country={"DE": 0.5, "FR": 0.3, "IT": 0.2})
```

Over 100 country codes are supported, using standard ISO 3166-1
alpha-2 codes.
