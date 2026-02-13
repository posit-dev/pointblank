from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass

__all__ = [
    # Helper functions (primary API)
    "int_field",
    "float_field",
    "string_field",
    "bool_field",
    "date_field",
    "datetime_field",
    "time_field",
    "duration_field",
    # Classes (for type hints and advanced usage)
    "Field",
    "IntField",
    "FloatField",
    "StringField",
    "BoolField",
    "DateField",
    "DatetimeField",
    "TimeField",
    "DurationField",
]


# Available presets for realistic data generation
AVAILABLE_PRESETS = frozenset(
    {
        # Personal
        "name",
        "name_full",
        "first_name",
        "last_name",
        "email",
        "phone_number",
        "address",
        "city",
        "state",
        "country",
        "postcode",
        "latitude",
        "longitude",
        # Business
        "company",
        "job",
        "catch_phrase",
        # Internet
        "url",
        "domain_name",
        "ipv4",
        "ipv6",
        "user_name",
        "password",
        # Text
        "text",
        "sentence",
        "paragraph",
        "word",
        # Financial
        "credit_card_number",
        "iban",
        "currency_code",
        # Identifiers
        "uuid4",
        "ssn",
        "license_plate",
        # Date/Time (for string representations)
        "date_this_year",
        "date_this_decade",
        "time",
        # Misc
        "color_name",
        "file_name",
        "file_extension",
        "mime_type",
    }
)


# =============================================================================
# Base Field Class
# =============================================================================


@dataclass
class Field:
    """
    Base class for column specifications in schema definition.

    This is the base class used internally. For creating fields, use the
    purpose-built field classes or helper functions:

    - `int_field()` / `IntField` for integer columns
    - `float_field()` / `FloatField` for floating-point columns
    - `string_field()` / `StringField` for string columns
    - `bool_field()` / `BoolField` for boolean columns
    - `date_field()` / `DateField` for date columns
    - `datetime_field()` / `DatetimeField` for datetime columns
    - `time_field()` / `TimeField` for time columns
    - `duration_field()` / `DurationField` for duration columns
    """

    dtype: str

    # Nullability
    nullable: bool = False
    null_probability: float = 0.0

    # Uniqueness
    unique: bool = False

    # Custom generator
    generator: Callable[[], Any] | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate field constraints after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate that all field constraints are consistent and valid."""
        # Validate null_probability
        if not 0.0 <= self.null_probability <= 1.0:
            raise ValueError(
                f"null_probability must be between 0.0 and 1.0, got {self.null_probability}"
            )

        if self.null_probability > 0.0 and not self.nullable:
            raise ValueError("null_probability > 0 requires nullable=True")

    def is_numeric(self) -> bool:
        """Check if this field has a numeric dtype."""
        return self.dtype in {
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Float32",
            "Float64",
        }

    def is_integer(self) -> bool:
        """Check if this field has an integer dtype."""
        return self.dtype in {
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
        }

    def is_float(self) -> bool:
        """Check if this field has a float dtype."""
        return self.dtype in {"Float32", "Float64"}

    def is_string(self) -> bool:
        """Check if this field has a string dtype."""
        return self.dtype == "String"

    def is_boolean(self) -> bool:
        """Check if this field has a boolean dtype."""
        return self.dtype == "Boolean"

    def is_temporal(self) -> bool:
        """Check if this field has a temporal dtype."""
        return self.dtype in {"Date", "Datetime", "Time", "Duration"}

    def has_custom_generator(self) -> bool:
        """Check if this field uses a custom generator."""
        return self.generator is not None


# =============================================================================
# Integer Field
# =============================================================================

# Valid integer dtypes
INT_DTYPES = frozenset({"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"})


@dataclass
class IntField(Field):
    """
    Integer column specification for schema definition.

    Parameters
    ----------
    min_val
        Minimum value (inclusive). Default is `None` (no minimum).
    max_val
        Maximum value (inclusive). Default is `None` (no maximum).
    allowed
        List of allowed values (categorical constraint). When provided,
        values are sampled from this list.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating null when `nullable=True`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`.
    generator
        Custom callable that generates values. Overrides other settings.
    dtype
        Integer dtype. Default is `"Int64"`. Options: `"Int8"`, `"Int16"`,
        `"Int32"`, `"Int64"`, `"UInt8"`, `"UInt16"`, `"UInt32"`, `"UInt64"`.

    Raises
    ------
    ValueError
        If constraints are invalid (e.g., `min_val > max_val`).

    Examples
    --------
    Define a schema with integer fields and generate test data:

    ```python
    import pointblank as pb

    # Define a schema with integer field specifications
    schema = pb.Schema(
        user_id=pb.int_field(min_val=1, unique=True),
        age=pb.int_field(min_val=0, max_val=120),
        rating=pb.int_field(allowed=[1, 2, 3, 4, 5]),
    )

    # Generate 100 rows of test data
    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    The generated data will have unique user IDs starting from `1`, ages between `0`-`120`,
    and ratings sampled from the allowed values.
    """

    # Integer-specific constraints
    min_val: int | None = None
    max_val: int | None = None
    allowed: list[int] | None = field(default=None)

    # Override dtype with default
    dtype: str = "Int64"

    def _validate(self) -> None:
        """Validate integer field constraints."""
        super()._validate()

        # Validate dtype
        if self.dtype not in INT_DTYPES:
            raise ValueError(
                f"Invalid dtype '{self.dtype}' for IntField. Valid options: {sorted(INT_DTYPES)}"
            )

        # Validate min/max
        if self.min_val is not None and self.max_val is not None:
            if self.min_val > self.max_val:
                raise ValueError(
                    f"min_val ({self.min_val}) cannot be greater than max_val ({self.max_val})"
                )

        # Validate allowed list
        if self.allowed is not None:
            if len(self.allowed) == 0:
                raise ValueError("allowed list cannot be empty")

    def has_allowed_values(self) -> bool:
        """Check if this field has a set of allowed values."""
        return self.allowed is not None


def int_field(
    min_val: int | None = None,
    max_val: int | None = None,
    allowed: list[int] | None = None,
    nullable: bool = False,
    null_probability: float = 0.0,
    unique: bool = False,
    generator: Callable[[], Any] | None = None,
    dtype: str = "Int64",
) -> IntField:
    """
    Create an integer column specification for use in a schema.

    The `int_field()` function defines the constraints and behavior for an integer column when
    generating synthetic data with `generate_dataset()`. You can control the range of values
    with `min_val=` and `max_val=`, restrict values to a specific set with `allowed=`, enforce
    uniqueness with `unique=True`, and introduce null values with `nullable=True` and
    `null_probability=`. The `dtype=` parameter lets you choose the specific integer type (e.g.,
    `"Int8"`, `"UInt16"`, `"Int64"`), which also determines the valid range of values.

    When no constraints are specified, values are drawn uniformly from the full range of the
    chosen integer dtype. If both `min_val=` and `max_val=` are provided, values are drawn
    uniformly from that range. If `allowed=` is provided, values are sampled from that specific
    list.

    Parameters
    ----------
    min_val
        Minimum value (inclusive). Default is `None` (no minimum, uses dtype lower bound).
    max_val
        Maximum value (inclusive). Default is `None` (no maximum, uses dtype upper bound).
    allowed
        List of allowed values (categorical constraint). When provided, values are sampled from
        this list. Cannot be combined with `min_val=`/`max_val=`.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating a null value for each row when `nullable=True`. Must be
        between `0.0` and `1.0`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`. When `True`, the generator will
        retry until it produces `n` distinct values (subject to retry limits).
    generator
        Custom callable that generates values. When provided, this overrides all other
        constraints (`min_val=`, `max_val=`, `allowed=`, etc.). The callable should take no
        arguments and return a single integer value.
    dtype
        Integer dtype. Default is `"Int64"`. Options: `"Int8"`, `"Int16"`, `"Int32"`,
        `"Int64"`, `"UInt8"`, `"UInt16"`, `"UInt32"`, `"UInt64"`.

    Returns
    -------
    IntField
        An integer field specification that can be passed to `Schema()`.

    Raises
    ------
    ValueError
        If `min_val` is greater than `max_val`, if `allowed` is an empty list, if
        `null_probability` is not between `0.0` and `1.0`, or if `dtype` is not a valid
        integer type.

    Examples
    --------
    The `min_val=` and `max_val=` parameters constrain generated ranges, while `allowed=`
    restricts values to a specific set:

    ```{python}
    import pointblank as pb

    schema = pb.Schema(
        user_id=pb.int_field(min_val=1, unique=True),
        age=pb.int_field(min_val=0, max_val=120),
        rating=pb.int_field(allowed=[1, 2, 3, 4, 5]),
    )

    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    It's possible to introduce missing values with `nullable=True` and `null_probability=`,
    and to select a smaller dtype with `dtype=`:

    ```{python}
    schema = pb.Schema(
        score=pb.int_field(min_val=0, max_val=255, dtype="UInt8"),
        optional_val=pb.int_field(
            min_val=1, max_val=50,
            nullable=True, null_probability=0.3,
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=50, seed=42))
    ```

    We can also enforce uniqueness with `unique=True` to produce distinct identifiers within
    a range:

    ```{python}
    schema = pb.Schema(
        record_id=pb.int_field(min_val=1000, max_val=9999, unique=True),
        priority=pb.int_field(allowed=[1, 2, 3]),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=10))
    ```

    For complete control, a custom `generator=` callable can be provided:

    ```{python}
    import random

    rng = random.Random(0)

    schema = pb.Schema(
        even_numbers=pb.int_field(generator=lambda: rng.choice(range(0, 100, 2))),
    )

    pb.preview(pb.generate_dataset(schema, n=20, seed=5))
    ```
    """
    return IntField(
        min_val=min_val,
        max_val=max_val,
        allowed=allowed,
        nullable=nullable,
        null_probability=null_probability,
        unique=unique,
        generator=generator,
        dtype=dtype,
    )


# =============================================================================
# Float Field
# =============================================================================

FLOAT_DTYPES = frozenset({"Float32", "Float64"})


@dataclass
class FloatField(Field):
    """
    Floating-point column specification for schema definition.

    Parameters
    ----------
    min_val
        Minimum value (inclusive). Default is `None` (no minimum).
    max_val
        Maximum value (inclusive). Default is `None` (no maximum).
    allowed
        List of allowed values (categorical constraint). When provided,
        values are sampled from this list.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating null when `nullable=True`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`.
    generator
        Custom callable that generates values. Overrides other settings.
    dtype
        Float dtype. Default is `"Float64"`. Options: `"Float32"`, `"Float64"`.

    Raises
    ------
    ValueError
        If constraints are invalid (e.g., `min_val > max_val`).

    Examples
    --------
    Define a schema with float fields and generate test data:

    ```python
    import pointblank as pb

    # Define a schema with float field specifications
    schema = pb.Schema(
        price=pb.float_field(min_val=0.01, max_val=9999.99),
        probability=pb.float_field(min_val=0.0, max_val=1.0),
        temperature=pb.float_field(min_val=-40.0, max_val=50.0),
    )

    # Generate 100 rows of test data
    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    Values are uniformly distributed across the specified ranges.
    """

    # Float-specific constraints
    min_val: float | None = None
    max_val: float | None = None
    allowed: list[float] | None = field(default=None)

    # Override dtype with default
    dtype: str = "Float64"

    def _validate(self) -> None:
        """Validate float field constraints."""
        super()._validate()

        # Validate dtype
        if self.dtype not in FLOAT_DTYPES:
            raise ValueError(
                f"Invalid dtype '{self.dtype}' for FloatField. "
                f"Valid options: {sorted(FLOAT_DTYPES)}"
            )

        # Validate min/max
        if self.min_val is not None and self.max_val is not None:
            if self.min_val > self.max_val:
                raise ValueError(
                    f"min_val ({self.min_val}) cannot be greater than max_val ({self.max_val})"
                )

        # Validate allowed list
        if self.allowed is not None:
            if len(self.allowed) == 0:
                raise ValueError("allowed list cannot be empty")

    def has_allowed_values(self) -> bool:
        """Check if this field has a set of allowed values."""
        return self.allowed is not None


def float_field(
    min_val: float | None = None,
    max_val: float | None = None,
    allowed: list[float] | None = None,
    nullable: bool = False,
    null_probability: float = 0.0,
    unique: bool = False,
    generator: Callable[[], Any] | None = None,
    dtype: str = "Float64",
) -> FloatField:
    """
    Create a floating-point column specification for use in a schema.

    The `float_field()` function defines the constraints and behavior for a floating-point column
    when generating synthetic data with `generate_dataset()`. You can control the range of values
    with `min_val=` and `max_val=`, restrict values to a specific set with `allowed=`, enforce
    uniqueness with `unique=True`, and introduce null values with `nullable=True` and
    `null_probability=`. The `dtype=` parameter lets you choose between `"Float32"` and
    `"Float64"` precision.

    When both `min_val=` and `max_val=` are provided, values are drawn from a uniform
    distribution across that range. If neither is specified, values are drawn uniformly from a
    large default range. If `allowed=` is provided, values are sampled from that specific list.

    Parameters
    ----------
    min_val
        Minimum value (inclusive). Default is `None` (no minimum).
    max_val
        Maximum value (inclusive). Default is `None` (no maximum).
    allowed
        List of allowed values (categorical constraint). When provided, values are sampled from
        this list. Cannot be combined with `min_val=`/`max_val=`.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating a null value for each row when `nullable=True`. Must be
        between `0.0` and `1.0`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`. When `True`, the generator will
        retry until it produces `n` distinct values.
    generator
        Custom callable that generates values. When provided, this overrides all other
        constraints. The callable should take no arguments and return a single float value.
    dtype
        Float dtype. Default is `"Float64"`. Options: `"Float32"`, `"Float64"`.

    Returns
    -------
    FloatField
        A float field specification that can be passed to `Schema()`.

    Raises
    ------
    ValueError
        If `min_val` is greater than `max_val`, if `allowed` is an empty list, if
        `null_probability` is not between `0.0` and `1.0`, or if `dtype` is not a valid
        float type.

    Examples
    --------
    The `min_val=` and `max_val=` parameters define the generated value ranges:

    ```{python}
    import pointblank as pb

    schema = pb.Schema(
        price=pb.float_field(min_val=0.01, max_val=9999.99),
        probability=pb.float_field(min_val=0.0, max_val=1.0),
        temperature=pb.float_field(min_val=-40.0, max_val=50.0),
    )

    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    It's also possible to restrict values to a discrete set with `allowed=`, which is useful
    for fixed pricing tiers or measurement levels:

    ```{python}
    schema = pb.Schema(
        discount=pb.float_field(allowed=[0.05, 0.10, 0.15, 0.20, 0.25]),
        weight_kg=pb.float_field(min_val=0.5, max_val=100.0),
    )

    pb.preview(pb.generate_dataset(schema, n=50, seed=42))
    ```

    We can simulate missing measurements by introducing null values:

    ```{python}
    schema = pb.Schema(
        reading=pb.float_field(
            min_val=0.0, max_val=500.0,
            nullable=True, null_probability=0.2,
        ),
        calibration=pb.float_field(min_val=0.9, max_val=1.1),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=7))
    ```

    Setting `dtype="Float32"` gives reduced precision, and a custom `generator=` provides
    full control over value generation:

    ```{python}
    import random, math

    rng = random.Random(0)

    schema = pb.Schema(
        sensor_value=pb.float_field(min_val=-10.0, max_val=10.0, dtype="Float32"),
        log_value=pb.float_field(generator=lambda: math.log(rng.uniform(1, 1000))),
    )

    pb.preview(pb.generate_dataset(schema, n=20, seed=99))
    ```
    """
    return FloatField(
        min_val=min_val,
        max_val=max_val,
        allowed=allowed,
        nullable=nullable,
        null_probability=null_probability,
        unique=unique,
        generator=generator,
        dtype=dtype,
    )


# =============================================================================
# String Field
# =============================================================================


@dataclass
class StringField(Field):
    """
    String column specification for schema definition.

    Parameters
    ----------
    min_length
        Minimum string length. Default is `None` (no minimum).
    max_length
        Maximum string length. Default is `None` (no maximum).
    pattern
        Regular expression pattern for generated strings.
    preset
        Preset for realistic data (e.g., `"email"`, `"name"`, `"phone_number"`).
    allowed
        List of allowed values (categorical constraint).
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating null when `nullable=True`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`.
    generator
        Custom callable that generates values. Overrides other settings.
    dtype
        Always `"String"` for StringField.

    Raises
    ------
    ValueError
        If constraints are invalid or incompatible.

    Examples
    --------
    Define a schema with string fields and generate test data:

    ```python
    import pointblank as pb

    # Define a schema with string field specifications
    schema = pb.Schema(
        name=pb.string_field(preset="name"),
        email=pb.string_field(preset="email", unique=True),
        status=pb.string_field(allowed=["active", "pending", "inactive"]),
        code=pb.string_field(pattern=r"[A-Z]{3}-[0-9]{4}"),
    )

    # Generate 100 rows of test data
    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    The generated data will have coherent names and emails (derived from the name),
    statuses sampled from the allowed values, and codes matching the regex pattern.
    """

    # String-specific constraints
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    preset: str | None = None
    allowed: list[str] | None = field(default=None)

    # Override dtype with fixed value
    dtype: str = "String"

    def _validate(self) -> None:
        """Validate string field constraints."""
        super()._validate()

        # Validate dtype (must be String)
        if self.dtype != "String":
            raise ValueError(f"StringField dtype must be 'String', got '{self.dtype}'")

        # Validate length constraints
        if self.min_length is not None and self.min_length < 0:
            raise ValueError(f"min_length must be non-negative, got {self.min_length}")

        if self.max_length is not None and self.max_length < 0:
            raise ValueError(f"max_length must be non-negative, got {self.max_length}")

        if self.min_length is not None and self.max_length is not None:
            if self.min_length > self.max_length:
                raise ValueError(
                    f"min_length ({self.min_length}) cannot be greater than "
                    f"max_length ({self.max_length})"
                )

        # Validate preset
        if self.preset is not None and self.preset not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset '{self.preset}'. Available presets: {sorted(AVAILABLE_PRESETS)}"
            )

        # Validate allowed list
        if self.allowed is not None:
            if len(self.allowed) == 0:
                raise ValueError("allowed list cannot be empty")

        # Validate incompatible combinations
        specified = []
        if self.preset is not None:
            specified.append("preset")
        if self.pattern is not None:
            specified.append("pattern")
        if self.allowed is not None:
            specified.append("allowed")

        if len(specified) > 1:
            raise ValueError(
                f"Only one of preset, pattern, or allowed can be specified. "
                f"Got: {', '.join(specified)}"
            )

    def has_preset(self) -> bool:
        """Check if this field uses a preset for generation."""
        return self.preset is not None

    def has_allowed_values(self) -> bool:
        """Check if this field has a set of allowed values."""
        return self.allowed is not None

    def has_pattern(self) -> bool:
        """Check if this field has a regex pattern constraint."""
        return self.pattern is not None


def string_field(
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    preset: str | None = None,
    allowed: list[str] | None = None,
    nullable: bool = False,
    null_probability: float = 0.0,
    unique: bool = False,
    generator: Callable[[], Any] | None = None,
) -> StringField:
    """
    Create a string column specification for use in a schema.

    The `string_field()` function defines the constraints and behavior for a string column when
    generating synthetic data with `generate_dataset()`. It provides three main modes of string
    generation: (1) controlled random strings with `min_length=`/`max_length=`, (2) strings
    matching a regular expression via `pattern=`, or (3) realistic data using `preset=` (e.g.,
    `"email"`, `"name"`, `"address"`). You can also restrict values to a fixed set with
    `allowed=`. Only one of `preset=`, `pattern=`, or `allowed=` can be specified at a time.

    When no special mode is selected, random alphanumeric strings are generated with lengths
    between `min_length=` and `max_length=` (defaulting to 1--20 characters).

    Parameters
    ----------
    min_length
        Minimum string length (for random string generation). Default is `None` (defaults to
        `1`). Only applies when `preset=`, `pattern=`, and `allowed=` are all `None`.
    max_length
        Maximum string length (for random string generation). Default is `None` (defaults to
        `20`). Only applies when `preset=`, `pattern=`, and `allowed=` are all `None`.
    pattern
        Regular expression pattern that generated strings must match. Supports character
        classes (e.g., `[A-Z]`, `[0-9]`), quantifiers (e.g., `{3}`, `{2,5}`), alternation,
        and groups. Cannot be combined with `preset=` or `allowed=`.
    preset
        Preset name for generating realistic data. When specified, values are produced using
        locale-aware data generation, and the `country=` parameter of `generate_dataset()`
        controls the locale. Cannot be combined with `pattern=` or `allowed=`. See the
        **Available Presets** section below for the full list.
    allowed
        List of allowed string values (categorical constraint). Values are sampled uniformly
        from this list. Cannot be combined with `preset=` or `pattern=`.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating a null value for each row when `nullable=True`. Must be
        between `0.0` and `1.0`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`. When `True`, the generator will
        retry until it produces `n` distinct values.
    generator
        Custom callable that generates values. When provided, this overrides all other
        constraints. The callable should take no arguments and return a single string value.

    Returns
    -------
    StringField
        A string field specification that can be passed to `Schema()`.

    Raises
    ------
    ValueError
        If more than one of `preset=`, `pattern=`, or `allowed=` is specified; if `allowed=`
        is an empty list; if `min_length` or `max_length` is negative; if `min_length` exceeds
        `max_length`; or if `preset` is not a recognized preset name.

    Available Presets
    -----------------
    The `preset=` parameter accepts one of the following preset names, organized by category.
    When a preset is used, the `country=` parameter of `generate_dataset()` controls the locale
    for region-specific formatting (e.g., address formats, phone number patterns).

    **Personal:** `"name"` (first + last name), `"name_full"` (full name with possible prefix
    or suffix), `"first_name"`, `"last_name"`, `"email"` (realistic email address),
    `"phone_number"`, `"address"` (full street address), `"city"`, `"state"`, `"country"`,
    `"postcode"`, `"latitude"`, `"longitude"`

    **Business:** `"company"` (company name), `"job"` (job title), `"catch_phrase"`

    **Internet:** `"url"`, `"domain_name"`, `"ipv4"`, `"ipv6"`, `"user_name"`, `"password"`

    **Text:** `"text"` (paragraph of text), `"sentence"`, `"paragraph"`, `"word"`

    **Financial:** `"credit_card_number"`, `"iban"`, `"currency_code"`

    **Identifiers:** `"uuid4"`, `"ssn"` (social security number), `"license_plate"`

    **Date/Time (as strings):** `"date_this_year"`, `"date_this_decade"`, `"time"`

    **Miscellaneous:** `"color_name"`, `"file_name"`, `"file_extension"`, `"mime_type"`

    Coherent Data Generation
    ------------------------
    When multiple columns in the same schema use related presets, the generated data will be
    coherent across those columns within each row. Specifically:

    - **Person-related presets** (`"name"`, `"name_full"`, `"first_name"`, `"last_name"`,
      `"email"`, `"user_name"`): the email and username will be derived from the person's name.
    - **Address-related presets** (`"address"`, `"city"`, `"state"`, `"postcode"`,
      `"phone_number"`, `"latitude"`, `"longitude"`): the city, state, and postcode will
      correspond to the same location within the address.

    This coherence is automatic and requires no additional configuration.

    Examples
    --------
    The `preset=` parameter generates realistic personal data, while `allowed=` restricts
    values to a categorical set:

    ```{python}
    import pointblank as pb

    schema = pb.Schema(
        name=pb.string_field(preset="name"),
        email=pb.string_field(preset="email", unique=True),
        status=pb.string_field(allowed=["active", "pending", "inactive"]),
    )

    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    We can also generate strings that match a regular expression with `pattern=` (e.g.,
    product codes, identifiers):

    ```{python}
    schema = pb.Schema(
        product_code=pb.string_field(pattern=r"[A-Z]{3}-[0-9]{4}"),
        batch_id=pb.string_field(pattern=r"BATCH-[A-Z][0-9]{3}"),
        sku=pb.string_field(pattern=r"[A-Z]{2}[0-9]{6}"),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=42))
    ```

    For random alphanumeric strings, `min_length=` and `max_length=` control the length.
    Adding `nullable=True` introduces missing values:

    ```{python}
    schema = pb.Schema(
        short_code=pb.string_field(min_length=3, max_length=5),
        notes=pb.string_field(
            min_length=10, max_length=50,
            nullable=True, null_probability=0.4,
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=7))
    ```

    It's possible to combine business and internet presets to build a company directory:

    ```{python}
    schema = pb.Schema(
        company=pb.string_field(preset="company"),
        domain=pb.string_field(preset="domain_name"),
        industry_tag=pb.string_field(allowed=["tech", "finance", "health", "retail"]),
    )

    pb.preview(pb.generate_dataset(schema, n=20, seed=55))
    ```
    """
    return StringField(
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        preset=preset,
        allowed=allowed,
        nullable=nullable,
        null_probability=null_probability,
        unique=unique,
        generator=generator,
    )


# =============================================================================
# Boolean Field
# =============================================================================


@dataclass
class BoolField(Field):
    """
    Boolean column specification for schema definition.

    Parameters
    ----------
    p_true
        Probability of generating `True`. Default is `0.5` (equal probability).
        Must be between 0.0 and 1.0.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating null when `nullable=True`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`.
        Note: Boolean can only have 2 unique non-null values.
    generator
        Custom callable that generates values. Overrides other settings.
    dtype
        Always `"Boolean"` for BoolField.

    Examples
    --------
    Define a schema with boolean fields and generate test data:

    ```python
    import pointblank as pb

    # Define a schema with boolean field specifications
    schema = pb.Schema(
        is_active=pb.bool_field(p_true=0.8),      # 80% True
        is_premium=pb.bool_field(p_true=0.2),     # 20% True
        is_verified=pb.bool_field(),              # 50% True (default)
    )

    # Generate 100 rows of test data
    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    The `p_true` parameter controls the probability of generating `True` values,
    which is helpful for simulating real-world distributions.
    """

    # Boolean-specific parameter
    p_true: float = 0.5

    # Override dtype with fixed value
    dtype: str = "Boolean"

    def _validate(self) -> None:
        """Validate boolean field constraints."""
        super()._validate()

        # Validate dtype (must be Boolean)
        if self.dtype != "Boolean":
            raise ValueError(f"BoolField dtype must be 'Boolean', got '{self.dtype}'")

        # Validate p_true
        if not 0.0 <= self.p_true <= 1.0:
            raise ValueError(f"p_true must be between 0.0 and 1.0, got {self.p_true}")


def bool_field(
    p_true: float = 0.5,
    nullable: bool = False,
    null_probability: float = 0.0,
    unique: bool = False,
    generator: Callable[[], Any] | None = None,
) -> BoolField:
    """
    Create a boolean column specification for use in a schema.

    The `bool_field()` function defines the constraints and behavior for a boolean column when
    generating synthetic data with `generate_dataset()`. The `p_true=` parameter controls the
    probability of generating `True` values, which is useful for simulating real-world
    distributions where events may be rare or common (e.g., 5% fraud rate, 80% active users).

    By default, `True` and `False` are equally likely (`p_true=0.5`). Setting `p_true=0.0`
    produces all `False` values, and `p_true=1.0` produces all `True` values.

    Parameters
    ----------
    p_true
        Probability of generating `True`. Default is `0.5` (equal probability).
        Must be between `0.0` and `1.0`.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating a null value for each row when `nullable=True`. Must be
        between `0.0` and `1.0`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`. Note that boolean columns can
        only have 2 unique non-null values, so `n` must be `<= 2` when `unique=True` (or
        `<= 3` with `nullable=True`).
    generator
        Custom callable that generates values. When provided, this overrides all other
        constraints. The callable should take no arguments and return a single boolean value.

    Returns
    -------
    BoolField
        A boolean field specification that can be passed to `Schema()`.

    Raises
    ------
    ValueError
        If `p_true` is not between `0.0` and `1.0`, or if `null_probability` is not between
        `0.0` and `1.0`.

    Examples
    --------
    The `p_true=` parameter controls the distribution of `True`/`False` values, allowing
    you to simulate different probabilities:

    ```{python}
    import pointblank as pb

    schema = pb.Schema(
        is_active=pb.bool_field(p_true=0.8),
        is_premium=pb.bool_field(p_true=0.2),
        is_verified=pb.bool_field(),
    )

    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    Optional boolean flags can be simulated by combining `nullable=True` with
    `null_probability=`:

    ```{python}
    schema = pb.Schema(
        opted_in=pb.bool_field(p_true=0.6),
        has_referral=pb.bool_field(
            p_true=0.3,
            nullable=True, null_probability=0.25,
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=50, seed=42))
    ```

    Boolean fields can be combined with other field types in a realistic schema:

    ```{python}
    schema = pb.Schema(
        user_id=pb.int_field(min_val=1, unique=True),
        name=pb.string_field(preset="name"),
        email_verified=pb.bool_field(p_true=0.9),
        is_admin=pb.bool_field(p_true=0.05),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=10))
    ```
    """
    return BoolField(
        p_true=p_true,
        nullable=nullable,
        null_probability=null_probability,
        unique=unique,
        generator=generator,
    )


# =============================================================================
# Date Field
# =============================================================================


@dataclass
class DateField(Field):
    """
    Date column specification for schema definition.

    Parameters
    ----------
    min_date
        Minimum date (inclusive). Can be ISO string or `date` object.
    max_date
        Maximum date (inclusive). Can be ISO string or `date` object.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating null when `nullable=True`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`.
    generator
        Custom callable that generates values. Overrides other settings.
    dtype
        Always `"Date"` for DateField.

    Examples
    --------
    Define a schema with date fields and generate test data:

    ```python
    import pointblank as pb
    from datetime import date

    # Define a schema with date field specifications
    schema = pb.Schema(
        birth_date=pb.date_field(
            min_date=date(1960, 1, 1),
            max_date=date(2005, 12, 31)
        ),
        hire_date=pb.date_field(
            min_date=date(2020, 1, 1),
            max_date=date(2024, 12, 31)
        ),
    )

    # Generate 100 rows of test data
    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    Date values are uniformly distributed within the specified range.
    """

    # Date-specific constraints
    min_date: str | date | None = None
    max_date: str | date | None = None

    # Override dtype with fixed value
    dtype: str = "Date"

    def _validate(self) -> None:
        """Validate date field constraints."""
        super()._validate()

        # Validate dtype (must be Date)
        if self.dtype != "Date":
            raise ValueError(f"DateField dtype must be 'Date', got '{self.dtype}'")

        # Validate date range
        if self.min_date is not None and self.max_date is not None:
            min_dt = self._parse_date(self.min_date)
            max_dt = self._parse_date(self.max_date)
            if min_dt > max_dt:
                raise ValueError(
                    f"min_date ({self.min_date}) cannot be greater than max_date ({self.max_date})"
                )

    @staticmethod
    def _parse_date(value: str | date | datetime) -> datetime:
        """Parse a date value to datetime for comparison."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                raise ValueError(
                    f"Unable to parse date string '{value}'. Use ISO format (YYYY-MM-DD)."
                )
        raise ValueError(f"Invalid date type: {type(value)}")


def date_field(
    min_date: str | date | None = None,
    max_date: str | date | None = None,
    nullable: bool = False,
    null_probability: float = 0.0,
    unique: bool = False,
    generator: Callable[[], Any] | None = None,
) -> DateField:
    """
    Create a date column specification for use in a schema.

    The `date_field()` function defines the constraints and behavior for a date column when
    generating synthetic data with `generate_dataset()`. You can control the date range with
    `min_date=` and `max_date=`, enforce uniqueness with `unique=True`, and introduce null
    values with `nullable=True` and `null_probability=`.

    Dates are generated uniformly within the specified range. If no range is provided, the
    default range is 2000-01-01 to 2030-12-31. Both `min_date=` and `max_date=` accept either
    `datetime.date` objects or ISO 8601 date strings (e.g., `"2024-06-15"`).

    Parameters
    ----------
    min_date
        Minimum date (inclusive). Can be an ISO format string (e.g., `"2020-01-01"`) or a
        `datetime.date` object. Default is `None` (defaults to `2000-01-01`).
    max_date
        Maximum date (inclusive). Can be an ISO format string (e.g., `"2024-12-31"`) or a
        `datetime.date` object. Default is `None` (defaults to `2030-12-31`).
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating a null value for each row when `nullable=True`. Must be
        between `0.0` and `1.0`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`. When `True`, the generator will
        retry until it produces `n` distinct dates. Ensure the date range is large enough to
        accommodate the requested number of unique dates.
    generator
        Custom callable that generates values. When provided, this overrides all other
        constraints. The callable should take no arguments and return a single `datetime.date`
        value.

    Returns
    -------
    DateField
        A date field specification that can be passed to `Schema()`.

    Raises
    ------
    ValueError
        If `min_date` is later than `max_date`, or if a date string cannot be parsed.

    Examples
    --------
    The `min_date=` and `max_date=` parameters accept `datetime.date` objects to define date
    ranges:

    ```{python}
    import pointblank as pb
    from datetime import date

    schema = pb.Schema(
        birth_date=pb.date_field(
            min_date=date(1960, 1, 1),
            max_date=date(2005, 12, 31),
        ),
        hire_date=pb.date_field(
            min_date=date(2020, 1, 1),
            max_date=date(2024, 12, 31),
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    For convenience, ISO format strings can be used instead of `date` objects:

    ```{python}
    schema = pb.Schema(
        event_date=pb.date_field(min_date="2024-01-01", max_date="2024-12-31"),
        signup_date=pb.date_field(min_date="2023-06-01", max_date="2024-06-01"),
    )

    pb.preview(pb.generate_dataset(schema, n=50, seed=42))
    ```

    We can introduce missing dates with `nullable=True` and enforce distinct values using
    `unique=True`:

    ```{python}
    schema = pb.Schema(
        order_date=pb.date_field(
            min_date="2024-01-01", max_date="2024-03-31",
            unique=True,
        ),
        cancel_date=pb.date_field(
            min_date="2024-01-01", max_date="2024-12-31",
            nullable=True, null_probability=0.5,
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=7))
    ```
    """
    return DateField(
        min_date=min_date,
        max_date=max_date,
        nullable=nullable,
        null_probability=null_probability,
        unique=unique,
        generator=generator,
    )


# =============================================================================
# Datetime Field
# =============================================================================


@dataclass
class DatetimeField(Field):
    """
    Datetime column specification for schema definition.

    Parameters
    ----------
    min_date
        Minimum datetime (inclusive). Can be ISO string or `datetime` object.
    max_date
        Maximum datetime (inclusive). Can be ISO string or `datetime` object.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating null when `nullable=True`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`.
    generator
        Custom callable that generates values. Overrides other settings.
    dtype
        Always `"Datetime"` for DatetimeField.

    Examples
    --------
    Define a schema with datetime fields and generate test data:

    ```python
    import pointblank as pb
    from datetime import datetime

    # Define a schema with datetime field specifications
    schema = pb.Schema(
        created_at=pb.datetime_field(
            min_date=datetime(2024, 1, 1),
            max_date=datetime(2024, 12, 31)
        ),
        updated_at=pb.datetime_field(
            min_date=datetime(2024, 6, 1),
            max_date=datetime(2024, 12, 31)
        ),
    )

    # Generate 100 rows of test data
    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    Datetime values are uniformly distributed within the specified range.
    """

    # Datetime-specific constraints
    min_date: str | datetime | None = None
    max_date: str | datetime | None = None

    # Override dtype with fixed value
    dtype: str = "Datetime"

    def _validate(self) -> None:
        """Validate datetime field constraints."""
        super()._validate()

        # Validate dtype (must be Datetime)
        if self.dtype != "Datetime":
            raise ValueError(f"DatetimeField dtype must be 'Datetime', got '{self.dtype}'")

        # Validate date range
        if self.min_date is not None and self.max_date is not None:
            min_dt = self._parse_datetime(self.min_date)
            max_dt = self._parse_datetime(self.max_date)
            if min_dt > max_dt:
                raise ValueError(
                    f"min_date ({self.min_date}) cannot be greater than max_date ({self.max_date})"
                )

    @staticmethod
    def _parse_datetime(value: str | datetime) -> datetime:
        """Parse a datetime value for comparison."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                raise ValueError(
                    f"Unable to parse datetime string '{value}'. "
                    "Use ISO format (YYYY-MM-DDTHH:MM:SS)."
                )
        raise ValueError(f"Invalid datetime type: {type(value)}")


def datetime_field(
    min_date: str | datetime | None = None,
    max_date: str | datetime | None = None,
    nullable: bool = False,
    null_probability: float = 0.0,
    unique: bool = False,
    generator: Callable[[], Any] | None = None,
) -> DatetimeField:
    """
    Create a datetime column specification for use in a schema.

    The `datetime_field()` function defines the constraints and behavior for a datetime column
    when generating synthetic data with `generate_dataset()`. You can control the datetime range
    with `min_date=` and `max_date=`, enforce uniqueness with `unique=True`, and introduce null
    values with `nullable=True` and `null_probability=`.

    Datetime values are generated uniformly (at second-level resolution) within the specified
    range. If no range is provided, the default range is 2000-01-01T00:00:00 to
    2030-12-31T23:59:59. Both `min_date=` and `max_date=` accept `datetime` objects, `date`
    objects (which are converted to datetimes at midnight), or ISO 8601 datetime strings.

    Parameters
    ----------
    min_date
        Minimum datetime (inclusive). Can be an ISO format string (e.g.,
        `"2024-01-01T00:00:00"`), a `datetime.datetime` object, or a `datetime.date` object.
        Default is `None` (defaults to `2000-01-01 00:00:00`).
    max_date
        Maximum datetime (inclusive). Can be an ISO format string, a `datetime.datetime`
        object, or a `datetime.date` object. Default is `None` (defaults to
        `2030-12-31 23:59:59`).
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating a null value for each row when `nullable=True`. Must be
        between `0.0` and `1.0`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`. With second-level resolution
        over a wide range, collisions are unlikely for moderate dataset sizes.
    generator
        Custom callable that generates values. When provided, this overrides all other
        constraints. The callable should take no arguments and return a single
        `datetime.datetime` value.

    Returns
    -------
    DatetimeField
        A datetime field specification that can be passed to `Schema()`.

    Raises
    ------
    ValueError
        If `min_date` is later than `max_date`, or if a datetime string cannot be parsed.

    Examples
    --------
    The `min_date=` and `max_date=` parameters accept `datetime` objects for precise range
    definitions:

    ```{python}
    import pointblank as pb
    from datetime import datetime

    schema = pb.Schema(
        created_at=pb.datetime_field(
            min_date=datetime(2024, 1, 1),
            max_date=datetime(2024, 12, 31),
        ),
        updated_at=pb.datetime_field(
            min_date=datetime(2024, 6, 1),
            max_date=datetime(2024, 12, 31),
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    For a quick setup, ISO format strings work just as well:

    ```{python}
    schema = pb.Schema(
        event_time=pb.datetime_field(
            min_date="2024-03-01T08:00:00",
            max_date="2024-03-01T18:00:00",
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=42))
    ```

    Optional timestamps can be simulated with `nullable=True`, and datetime fields work
    nicely alongside other field types:

    ```{python}
    schema = pb.Schema(
        order_id=pb.int_field(min_val=1000, max_val=9999, unique=True),
        placed_at=pb.datetime_field(
            min_date=datetime(2024, 1, 1),
            max_date=datetime(2024, 12, 31),
        ),
        shipped_at=pb.datetime_field(
            min_date=datetime(2024, 1, 2),
            max_date=datetime(2025, 1, 15),
            nullable=True, null_probability=0.3,
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=7))
    ```
    """
    return DatetimeField(
        min_date=min_date,
        max_date=max_date,
        nullable=nullable,
        null_probability=null_probability,
        unique=unique,
        generator=generator,
    )


# =============================================================================
# Time Field
# =============================================================================


@dataclass
class TimeField(Field):
    """
    Time column specification for schema definition.

    Parameters
    ----------
    min_time
        Minimum time (inclusive). Can be ISO string or `time` object.
    max_time
        Maximum time (inclusive). Can be ISO string or `time` object.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating null when `nullable=True`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`.
    generator
        Custom callable that generates values. Overrides other settings.
    dtype
        Always `"Time"` for TimeField.

    Examples
    --------
    Define a schema with time fields and generate test data:

    ```python
    import pointblank as pb
    from datetime import time

    # Define a schema with time field specifications
    schema = pb.Schema(
        start_time=pb.time_field(
            min_time=time(9, 0, 0),
            max_time=time(12, 0, 0)
        ),
        end_time=pb.time_field(
            min_time=time(13, 0, 0),
            max_time=time(17, 0, 0)
        ),
    )

    # Generate 100 rows of test data
    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    Time values are uniformly distributed within the specified range.
    """

    # Time-specific constraints
    min_time: str | time | None = None
    max_time: str | time | None = None

    # Override dtype with fixed value
    dtype: str = "Time"

    def _validate(self) -> None:
        """Validate time field constraints."""
        super()._validate()

        # Validate dtype (must be Time)
        if self.dtype != "Time":
            raise ValueError(f"TimeField dtype must be 'Time', got '{self.dtype}'")

        # Validate time range
        if self.min_time is not None and self.max_time is not None:
            min_t = self._parse_time(self.min_time)
            max_t = self._parse_time(self.max_time)
            if min_t > max_t:
                raise ValueError(
                    f"min_time ({self.min_time}) cannot be greater than max_time ({self.max_time})"
                )

    @staticmethod
    def _parse_time(value: str | time) -> time:
        """Parse a time value for comparison."""
        if isinstance(value, time):
            return value
        if isinstance(value, str):
            try:
                return time.fromisoformat(value)
            except ValueError:
                raise ValueError(
                    f"Unable to parse time string '{value}'. Use ISO format (HH:MM:SS)."
                )
        raise ValueError(f"Invalid time type: {type(value)}")


def time_field(
    min_time: str | time | None = None,
    max_time: str | time | None = None,
    nullable: bool = False,
    null_probability: float = 0.0,
    unique: bool = False,
    generator: Callable[[], Any] | None = None,
) -> TimeField:
    """
    Create a time column specification for use in a schema.

    The `time_field()` function defines the constraints and behavior for a time-of-day column
    when generating synthetic data with `generate_dataset()`. You can control the time range
    with `min_time=` and `max_time=`, enforce uniqueness with `unique=True`, and introduce null
    values with `nullable=True` and `null_probability=`.

    Time values are generated uniformly (at second-level resolution) within the specified range.
    If no range is provided, the default range is 00:00:00 to 23:59:59. Both `min_time=` and
    `max_time=` accept `datetime.time` objects or ISO format time strings (e.g., `"09:30:00"`).

    Parameters
    ----------
    min_time
        Minimum time (inclusive). Can be an ISO format string (e.g., `"08:00:00"`) or a
        `datetime.time` object. Default is `None` (defaults to `00:00:00`).
    max_time
        Maximum time (inclusive). Can be an ISO format string (e.g., `"17:30:00"`) or a
        `datetime.time` object. Default is `None` (defaults to `23:59:59`).
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating a null value for each row when `nullable=True`. Must be
        between `0.0` and `1.0`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`. With second-level resolution
        within a time range, uniqueness is feasible for moderate dataset sizes.
    generator
        Custom callable that generates values. When provided, this overrides all other
        constraints. The callable should take no arguments and return a single value.

    Returns
    -------
    TimeField
        A time field specification that can be passed to `Schema()`.

    Raises
    ------
    ValueError
        If `min_time` is later than `max_time`, or if a time string cannot be parsed.

    Examples
    --------
    The `min_time=` and `max_time=` parameters accept `datetime.time` objects, making it
    easy to define business-hours ranges:

    ```{python}
    import pointblank as pb
    from datetime import time

    schema = pb.Schema(
        start_time=pb.time_field(
            min_time=time(9, 0, 0),
            max_time=time(12, 0, 0),
        ),
        end_time=pb.time_field(
            min_time=time(13, 0, 0),
            max_time=time(17, 0, 0),
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    ISO format strings can also be used for convenience:

    ```{python}
    schema = pb.Schema(
        login_time=pb.time_field(min_time="06:00:00", max_time="23:59:59"),
        alarm_time=pb.time_field(min_time="05:00:00", max_time="09:00:00"),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=42))
    ```

    It's possible to introduce optional time values with `nullable=True` and combine them
    with other field types:

    ```{python}
    schema = pb.Schema(
        employee_id=pb.int_field(min_val=100, max_val=999, unique=True),
        check_in=pb.time_field(min_time="07:00:00", max_time="10:00:00"),
        check_out=pb.time_field(
            min_time="16:00:00", max_time="20:00:00",
            nullable=True, null_probability=0.15,
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=7))
    ```
    """
    return TimeField(
        min_time=min_time,
        max_time=max_time,
        nullable=nullable,
        null_probability=null_probability,
        unique=unique,
        generator=generator,
    )


# =============================================================================
# Duration Field
# =============================================================================


@dataclass
class DurationField(Field):
    """
    Duration column specification for schema definition.

    Parameters
    ----------
    min_duration
        Minimum duration (inclusive). Can be ISO string or `timedelta` object.
    max_duration
        Maximum duration (inclusive). Can be ISO string or `timedelta` object.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating null when `nullable=True`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`.
    generator
        Custom callable that generates values. Overrides other settings.
    dtype
        Always `"Duration"` for DurationField.

    Examples
    --------
    Define a schema with duration fields and generate test data:

    ```python
    import pointblank as pb
    from datetime import timedelta

    # Define a schema with duration field specifications
    schema = pb.Schema(
        session_length=pb.duration_field(
            min_duration=timedelta(minutes=5),
            max_duration=timedelta(hours=2)
        ),
        wait_time=pb.duration_field(
            min_duration=timedelta(seconds=30),
            max_duration=timedelta(minutes=15)
        ),
    )

    # Generate 100 rows of test data
    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    Duration values are uniformly distributed within the specified range.
    """

    # Duration-specific constraints
    min_duration: str | timedelta | None = None
    max_duration: str | timedelta | None = None

    # Override dtype with fixed value
    dtype: str = "Duration"

    def _validate(self) -> None:
        """Validate duration field constraints."""
        super()._validate()

        # Validate dtype (must be Duration)
        if self.dtype != "Duration":
            raise ValueError(f"DurationField dtype must be 'Duration', got '{self.dtype}'")

        # Validate duration range
        if self.min_duration is not None and self.max_duration is not None:
            min_d = self._parse_duration(self.min_duration)
            max_d = self._parse_duration(self.max_duration)
            if min_d > max_d:
                raise ValueError(
                    f"min_duration ({self.min_duration}) cannot be greater than "
                    f"max_duration ({self.max_duration})"
                )

    @staticmethod
    def _parse_duration(value: str | timedelta) -> timedelta:
        """Parse a duration value for comparison."""
        if isinstance(value, timedelta):
            return value
        if isinstance(value, str):
            # Parse ISO 8601 duration format (simplified)
            # e.g., "PT1H30M" for 1 hour 30 minutes
            # For simplicity, we also accept formats like "1:30:00"
            try:
                parts = value.split(":")
                if len(parts) == 3:
                    hours, minutes, seconds = map(float, parts)
                    return timedelta(hours=hours, minutes=minutes, seconds=seconds)
                elif len(parts) == 2:
                    minutes, seconds = map(float, parts)
                    return timedelta(minutes=minutes, seconds=seconds)
            except ValueError:
                pass
            raise ValueError(
                f"Unable to parse duration string '{value}'. "
                "Use format 'HH:MM:SS' or timedelta object."
            )
        raise ValueError(f"Invalid duration type: {type(value)}")


def duration_field(
    min_duration: str | timedelta | None = None,
    max_duration: str | timedelta | None = None,
    nullable: bool = False,
    null_probability: float = 0.0,
    unique: bool = False,
    generator: Callable[[], Any] | None = None,
) -> DurationField:
    """
    Create a duration column specification for use in a schema.

    The `duration_field()` function defines the constraints and behavior for a duration
    (timedelta) column when generating synthetic data with `generate_dataset()`. You can
    control the duration range with `min_duration=` and `max_duration=`, enforce uniqueness
    with `unique=True`, and introduce null values with `nullable=True` and `null_probability=`.

    Duration values are generated uniformly (at second-level resolution) within the specified
    range. If no range is provided, the default range is 0 seconds to 30 days. Both
    `min_duration=` and `max_duration=` accept `datetime.timedelta` objects or colon-separated
    strings in `"HH:MM:SS"` or `"MM:SS"` format.

    Parameters
    ----------
    min_duration
        Minimum duration (inclusive). Can be a `"HH:MM:SS"` or `"MM:SS"` string, or a
        `datetime.timedelta` object. Default is `None` (defaults to 0 seconds).
    max_duration
        Maximum duration (inclusive). Can be a `"HH:MM:SS"` or `"MM:SS"` string, or a
        `datetime.timedelta` object. Default is `None` (defaults to 30 days).
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating a null value for each row when `nullable=True`. Must be
        between `0.0` and `1.0`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`. With second-level resolution
        within a duration range, uniqueness is feasible for moderate dataset sizes.
    generator
        Custom callable that generates values. When provided, this overrides all other
        constraints. The callable should take no arguments and return a single
        `datetime.timedelta` value.

    Returns
    -------
    DurationField
        A duration field specification that can be passed to `Schema()`.

    Raises
    ------
    ValueError
        If `min_duration` is greater than `max_duration`, or if a duration string cannot be
        parsed.

    Examples
    --------
    The `min_duration=` and `max_duration=` parameters accept `timedelta` objects for
    defining duration ranges:

    ```{python}
    import pointblank as pb
    from datetime import timedelta

    schema = pb.Schema(
        session_length=pb.duration_field(
            min_duration=timedelta(minutes=5),
            max_duration=timedelta(hours=2),
        ),
        wait_time=pb.duration_field(
            min_duration=timedelta(seconds=30),
            max_duration=timedelta(minutes=15),
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=100, seed=23))
    ```

    Colon-separated strings can also be used for quick duration definitions:

    ```{python}
    schema = pb.Schema(
        call_duration=pb.duration_field(min_duration="0:01:00", max_duration="1:30:00"),
        break_time=pb.duration_field(min_duration="0:05:00", max_duration="0:30:00"),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=42))
    ```

    Optional durations can be created with `nullable=True`, and duration fields work well
    alongside other field types:

    ```{python}
    schema = pb.Schema(
        task_id=pb.int_field(min_val=1, max_val=500, unique=True),
        time_spent=pb.duration_field(
            min_duration=timedelta(minutes=1),
            max_duration=timedelta(hours=8),
        ),
        overtime=pb.duration_field(
            min_duration=timedelta(0),
            max_duration=timedelta(hours=4),
            nullable=True, null_probability=0.6,
        ),
    )

    pb.preview(pb.generate_dataset(schema, n=30, seed=7))
    ```
    """
    return DurationField(
        min_duration=min_duration,
        max_duration=max_duration,
        nullable=nullable,
        null_probability=null_probability,
        unique=unique,
        generator=generator,
    )
