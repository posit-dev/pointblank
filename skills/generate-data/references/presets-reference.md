# String presets reference

Available preset values for `string_field(preset="...")`. Presets
generate realistic, locale-aware data using Faker under the hood.

## Person

| Preset          | Example output             |
|-----------------|----------------------------|
| `name`          | "John Smith"               |
| `first_name`    | "John"                     |
| `last_name`     | "Smith"                    |
| `prefix`        | "Mr."                      |
| `suffix`        | "Jr."                      |

## Contact

| Preset          | Example output             |
|-----------------|----------------------------|
| `email`         | "john.smith@example.com"   |
| `phone_number`  | "+1-555-123-4567"          |
| `url`           | "https://example.com"      |

## Address

| Preset          | Example output             |
|-----------------|----------------------------|
| `address`       | "123 Main St, Apt 4"       |
| `city`          | "New York"                 |
| `state`         | "California"               |
| `zipcode`       | "90210"                    |
| `country`       | "United States"            |
| `street_address`| "123 Main Street"          |

## Business

| Preset          | Example output             |
|-----------------|----------------------------|
| `company`       | "Acme Corporation"         |
| `job`           | "Software Engineer"        |
| `catch_phrase`  | "Innovative solutions"     |

## Internet

| Preset          | Example output             |
|-----------------|----------------------------|
| `user_name`     | "jsmith42"                 |
| `domain_name`   | "example.com"              |
| `ipv4`          | "192.168.1.1"              |
| `ipv6`          | "2001:db8::1"              |
| `mac_address`   | "00:1A:2B:3C:4D:5E"       |

## Identifiers

| Preset          | Example output             |
|-----------------|----------------------------|
| `uuid4`         | "a1b2c3d4-e5f6-..."       |
| `iban`          | "DE89 3704 0044 0532 ..."  |
| `ssn`           | "123-45-6789"              |
| `license_plate` | "ABC-1234"                 |

## Text

| Preset          | Example output             |
|-----------------|----------------------------|
| `text`          | "Lorem ipsum dolor..."     |
| `sentence`      | "The quick brown fox."     |
| `paragraph`     | "Lorem ipsum dolor sit..." |
| `word`          | "lorem"                    |

## Finance

| Preset              | Example output         |
|---------------------|------------------------|
| `credit_card_number` | "4111111111111111"     |
| `currency_code`      | "USD"                 |
| `cryptocurrency_code`| "BTC"                 |

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
```

Over 100 country codes are supported, using standard ISO 3166-1
alpha-2 codes.
