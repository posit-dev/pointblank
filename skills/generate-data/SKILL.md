---
name: generate-data
description: >
  Generate synthetic datasets with Pointblank using Schema and field
  classes. Covers IntField, FloatField, StringField, BoolField,
  DateField, DatetimeField, and more. Supports presets (name, email,
  address, etc.), country-specific data, nullable columns, unique
  constraints, profile_fields() for person data, schemas inferred from
  a real table, and check_synthetic() for comparing synthetic data with
  the real data it imitates. Use when creating test data, fixtures, or
  synthetic datasets for validation testing.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - data-generation
    - synthetic-data
    - test-data
    - schema
---

# Generate Data

Skill for creating synthetic datasets from schema definitions and
field specifications. Useful for testing validation rules, creating
fixtures, generating demo data, and prototyping pipelines.

## Quick start

```python
import pointblank as pb

schema = pb.Schema(
    id=pb.int_field(min_val=1, max_val=10000, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=95),
    score=pb.float_field(min_val=0.0, max_val=100.0, precision=2),
    active=pb.bool_field(p_true=0.8),
)

df = schema.generate(n=1000, seed=42)
```

## Skill directory structure

```
skills/generate-data/
+-- SKILL.md                    <- This file
+-- references/
    +-- field-reference.md      <- All field types and parameters
    +-- presets-reference.md    <- Available string presets
```

## When to use what

| I want to...                        | Use                                   |
| ----------------------------------- | ------------------------------------- |
| Generate a dataset from a schema    | `schema.generate()`                   |
| Generate without creating a Schema  | `pb.generate_dataset()`               |
| Define integer columns              | `int_field()`                         |
| Define float columns                | `float_field()`                       |
| Define string columns with patterns | `string_field(pattern=)`              |
| Define string columns with presets  | `string_field(preset=)`               |
| Define boolean columns              | `bool_field()`                        |
| Define date columns                 | `date_field()`                        |
| Define datetime columns             | `datetime_field()`                    |
| Define time columns                 | `time_field()`                        |
| Define duration columns             | `duration_field()`                    |
| Add person profile fields           | `profile_fields()`                    |
| Generate country-specific data      | `generate(country="DE")`              |
| Mix several countries               | `generate(country=["DE", "FR"])`      |
| Imitate an existing table           | `schema_from_tbl(df).generate()`      |
| Check synthetic data against real   | `check_synthetic()`                   |
| Make columns nullable               | `nullable=True, null_probability=0.1` |
| Ensure unique values                | `unique=True`                         |
| Use a custom generator function     | `generator=my_func`                   |

## Core concepts

### Schema-based generation

Define columns using field classes, then generate:

```python
schema = pb.Schema(
    order_id=pb.int_field(min_val=1, max_val=99999, unique=True),
    product=pb.string_field(allowed=["Widget A", "Widget B", "Gadget"]),
    quantity=pb.int_field(min_val=1, max_val=100),
    price=pb.float_field(min_val=0.99, max_val=999.99, precision=2),
    shipped=pb.bool_field(p_true=0.7),
    order_date=pb.date_field(min_date="2024-01-01", max_date="2024-12-31"),
)

df = schema.generate(n=500, seed=42, output="polars")
```

### generate() parameters

| Parameter  | Default    | Description                             |
| ---------- | ---------- | --------------------------------------- |
| `n`        | `100`      | Number of rows to generate              |
| `seed`     | `None`     | Random seed for reproducibility         |
| `output`   | `"polars"` | `"polars"`, `"pandas"`, or `"dict"`     |
| `country`  | `"US"`     | A country code, a list of them, or a dict of codes to weights |
| `shuffle`  | `True`     | Shuffle rows after generation           |
| `weighted` | `True`     | Use weighted distributions              |

### generate_dataset() convenience function

```python
df = pb.generate_dataset(schema, n=500, seed=42)
```

It takes the same parameters as `generate()`.

### Nullable columns

Any field type supports nulls (`null_probability` needs `nullable=True`):

```python
pb.int_field(min_val=0, max_val=100, nullable=True, null_probability=0.1)
pb.string_field(preset="email", nullable=True, null_probability=0.05)
```

### Unique constraints

Ensure all generated values are distinct:

```python
pb.int_field(min_val=1, max_val=10000, unique=True)
pb.string_field(preset="email", unique=True)
```

### Allowed values (categorical)

Restrict to a specific set of values:

```python
pb.int_field(allowed=[1, 2, 3, 5, 8, 13])
pb.float_field(allowed=[0.5, 1.0, 1.5, 2.0])
pb.string_field(allowed=["low", "medium", "high"])
```

### String patterns

Generate strings matching a pattern:

```python
pb.string_field(pattern=r"[A-Z]{3}-\d{4}")    # "ABC-1234"
pb.string_field(pattern=r"INV-\d{6}")          # "INV-003847"
pb.string_field(pattern=r"[a-z]{5,10}")        # random lowercase
```

### String presets

Use built-in presets for realistic data:

```python
pb.string_field(preset="name")           # full names
pb.string_field(preset="email")          # email addresses
pb.string_field(preset="address")        # full addresses
pb.string_field(preset="city")           # city names
pb.string_field(preset="postcode")       # postal codes
pb.string_field(preset="phone_number")   # phone numbers
pb.string_field(preset="company")        # company names
pb.string_field(preset="job")            # job titles
pb.string_field(preset="url")            # URLs
pb.string_field(preset="uuid4")          # UUIDs
pb.string_field(preset="iban")           # IBANs
pb.string_field(preset="ssn")            # SSNs
```

Presets produce country-specific data when `country` is set, and the
presets in a row agree with each other (the email is built from the
name, the city, state, and postcode go together). See
`references/presets-reference.md` for the full list; an unknown preset
raises a `ValueError` that lists them.

### Profile fields

Generate person-related fields as a group:

```python
fields = pb.profile_fields(
    set="standard",        # "minimal", "standard" or "full"
    split_name=True,       # first_name + last_name vs full name
    include=None,          # specific fields to include
    exclude=None,          # specific fields to exclude
    prefix=None,           # prefix for field names (each prefix is a separate person)
)

schema = pb.Schema(
    id=pb.int_field(min_val=1, max_val=99999, unique=True),
    **fields,
)

df = schema.generate(n=100, country="US")
```

### Custom generators

Supply your own generator function:

```python
import random

def custom_sku():
    return f"SKU-{random.randint(1000, 9999)}"

schema = pb.Schema(
    sku=pb.string_field(generator=custom_sku),
)
```

### Country-specific generation

Over 100 countries supported (ISO 3166-1 alpha-2 codes):

```python
# German names, addresses, phone numbers
df = schema.generate(n=100, country="DE")

# Japanese
df = schema.generate(n=100, country="JP")

# Brazilian
df = schema.generate(n=100, country="BR")

# A mix: rows spread evenly over the countries, or by weight
df = schema.generate(n=100, country=["DE", "FR"])
df = schema.generate(n=100, country={"JP": 0.7, "BR": 0.3})
```

## Workflows

### Creating test data for validation rules

1. Define the schema matching your production table.
2. Use field constraints to generate realistic ranges.
3. Add some nullable columns to test null handling.
4. Generate the dataset.
5. Run your validation plan against it.

```python
schema = pb.Schema(
    id=pb.int_field(min_val=1, max_val=10000, unique=True),
    amount=pb.float_field(min_val=-10, max_val=1000, precision=2),
    status=pb.string_field(allowed=["active", "inactive", "INVALID"]),
    email=pb.string_field(preset="email", nullable=True, null_probability=0.1),
)

test_df = schema.generate(n=500, seed=42)

validation = (
    pb.Validate(data=test_df)
    .col_vals_gt(columns="amount", value=0)
    .col_vals_in_set(columns="status", set=["active", "inactive"])
    .col_vals_not_null(columns="email")
    .interrogate()
)
```

### Generating fixtures from an existing table

```python
# Infer field constraints from real data
schema = pb.schema_from_tbl(
    production_df,
    infer_constraints=True,
    categorical_threshold=20,
)

# Generate synthetic version
fixture = schema.generate(n=100, seed=1)
```

`pb.Schema.from_table()` is the same as `schema_from_tbl()`. Each field
gets its range, null rate, uniqueness, and (for few distinct values) an
`allowed=` list from the data, and string columns are matched to
presets by name and values.

### Checking synthetic data against the real data

`check_synthetic()` compares synthetic data with its seed (the real
data it imitates) for fidelity (each column's distribution, by PSI and
KS), diversity (repeated rows, and distinct-n for `text=` columns), and
copies of seed rows (exact, or near copies of `text=` columns):

```python
real = pb.load_dataset("small_table")
synthetic = pb.schema_from_tbl(real).generate(n=50, seed=1)

check = pb.check_synthetic(synthetic, real)

check.ok                    # False if there's any issue
check.issues                # e.g. "The distribution of 'd' ... differs ... (PSI 2.65)."
check.fidelity              # per-column psi, ks, level, missing shares
check.get_tabular_report()
check.get_copies()          # synthetic rows that copy a seed row
```

Inferred fields keep each column's range but not its shape, so numeric
columns often fail fidelity; tighten those fields (or use `allowed=`)
when the distribution matters. See the `check-training-data` skill for
more on synthetic data used for training.

## Gotchas

1. **`unique=True` needs a large enough range.** If `max_val - min_val`
   < `n`, generation will fail for integer fields.
2. **Only one of `preset`, `pattern`, `allowed` per StringField.**
   They are mutually exclusive.
3. **Presets don't need Faker.** Their data ships with Pointblank,
   so no extra install is needed.
4. **`seed` makes generation reproducible for a given `n`.** With a
   larger `n`, most columns start with the smaller dataset's rows, but
   not all (`email` addresses come out differently), so don't rely on
   a smaller dataset being the first rows of a larger one.
5. **`output` is `"polars"`, `"pandas"`, or `"dict"`.** For other
   formats, convert after generation.
6. **`null_probability` needs `nullable=True`.** A probability above 0
   without it raises a `ValueError`, and `nullable=True` with the
   default `null_probability=0.0` generates no nulls.

## Related skills

| Skill               | When to use it                                   |
| ------------------- | ------------------------------------------------ |
| pointblank          | Validate the generated data                      |
| scan-and-profile    | Profile the real table the schema imitates       |
| write-validation    | Write the plan that the test data exercises      |
| check-training-data | Check synthetic data used for training a model   |
