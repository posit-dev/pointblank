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
    ```python
    import pointblank as pb

    # Basic integer field
    user_id = pb.int_field()

    # With constraints
    age = pb.int_field(min_val=0, max_val=120)

    # Categorical integers
    rating = pb.int_field(allowed=[1, 2, 3, 4, 5])

    # Unsigned 8-bit integer
    byte_val = pb.int_field(min_val=0, max_val=255, dtype="UInt8")
    ```
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
    Create an integer column specification.

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

    Returns
    -------
    IntField
        An integer field specification.

    Examples
    --------
    ```python
    import pointblank as pb

    schema = pb.Schema(
        user_id=pb.int_field(min_val=1),
        age=pb.int_field(min_val=0, max_val=120),
        rating=pb.int_field(allowed=[1, 2, 3, 4, 5]),
    )

    data = schema.generate(n=100, seed=23)
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
    ```python
    import pointblank as pb

    # Basic float field
    price = pb.float_field(min_val=0.0)

    # With range
    probability = pb.float_field(min_val=0.0, max_val=1.0)

    # Single precision
    sensor_reading = pb.float_field(dtype="Float32")
    ```
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
    Create a floating-point column specification.

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

    Returns
    -------
    FloatField
        A float field specification.

    Examples
    --------
    ```python
    import pointblank as pb

    schema = pb.Schema(
        price=pb.float_field(min_val=0.01, max_val=9999.99),
        probability=pb.float_field(min_val=0.0, max_val=1.0),
    )

    data = schema.generate(n=100, seed=23)
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
    ```python
    import pointblank as pb

    # Basic string field
    name = pb.string_field(min_length=1, max_length=100)

    # With regex pattern
    code = pb.string_field(pattern=r"[A-Z]{3}-\\d{4}")

    # Realistic data preset
    email = pb.string_field(preset="email", unique=True)

    # Categorical
    status = pb.string_field(allowed=["active", "pending", "inactive"])
    ```
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
    Create a string column specification.

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

    Returns
    -------
    StringField
        A string field specification.

    Examples
    --------
    ```python
    import pointblank as pb

    schema = pb.Schema(
        name=pb.string_field(preset="name"),
        email=pb.string_field(preset="email", unique=True),
        status=pb.string_field(allowed=["active", "pending", "inactive"]),
        code=pb.string_field(pattern=r"[A-Z]{3}-\\d{4}"),
    )

    data = schema.generate(n=100, seed=23)
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
    ```python
    import pointblank as pb

    # Basic boolean field (50% True)
    is_active = pb.bool_field()

    # Boolean with 80% probability of True
    is_verified = pb.bool_field(p_true=0.8)

    # Nullable boolean
    has_subscription = pb.bool_field(nullable=True, null_probability=0.1)
    ```
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
    Create a boolean column specification.

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

    Returns
    -------
    BoolField
        A boolean field specification.

    Examples
    --------
    ```python
    import pointblank as pb

    schema = pb.Schema(
        is_active=pb.bool_field(p_true=0.8),      # 80% True
        is_premium=pb.bool_field(p_true=0.2),     # 20% True
        is_verified=pb.bool_field(),              # 50% True (default)
    )

    data = schema.generate(n=100, seed=23)
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
    ```python
    import pointblank as pb

    # Basic date field
    created_at = pb.date_field()

    # With range
    birth_date = pb.date_field(min_date="1950-01-01", max_date="2005-12-31")
    ```
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
    Create a date column specification.

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

    Returns
    -------
    DateField
        A date field specification.

    Examples
    --------
    ```python
    import pointblank as pb

    schema = pb.Schema(
        created_at=pb.date_field(),
        birth_date=pb.date_field(min_date="1950-01-01", max_date="2005-12-31"),
    )

    data = schema.generate(n=100, seed=23)
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
    ```python
    import pointblank as pb

    # Basic datetime field
    timestamp = pb.datetime_field()

    # With range
    event_time = pb.datetime_field(
        min_date="2024-01-01T00:00:00",
        max_date="2024-12-31T23:59:59"
    )
    ```
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
    Create a datetime column specification.

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

    Returns
    -------
    DatetimeField
        A datetime field specification.

    Examples
    --------
    ```python
    import pointblank as pb

    schema = pb.Schema(
        created_at=pb.datetime_field(),
        event_time=pb.datetime_field(
            min_date="2024-01-01T00:00:00",
            max_date="2024-12-31T23:59:59"
        ),
    )

    data = schema.generate(n=100, seed=23)
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
    ```python
    import pointblank as pb

    # Basic time field
    start_time = pb.time_field()

    # Business hours only
    meeting_time = pb.time_field(min_time="09:00:00", max_time="17:00:00")
    ```
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
    Create a time column specification.

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

    Returns
    -------
    TimeField
        A time field specification.

    Examples
    --------
    ```python
    import pointblank as pb

    schema = pb.Schema(
        start_time=pb.time_field(),
        meeting_time=pb.time_field(min_time="09:00:00", max_time="17:00:00"),
    )

    data = schema.generate(n=100, seed=23)
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
    ```python
    import pointblank as pb
    from datetime import timedelta

    # Basic duration field
    elapsed = pb.duration_field()

    # With range
    session_length = pb.duration_field(
        min_duration=timedelta(minutes=1),
        max_duration=timedelta(hours=2)
    )
    ```
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
    Create a duration column specification.

    Parameters
    ----------
    min_duration
        Minimum duration (inclusive). Can be string or `timedelta` object.
    max_duration
        Maximum duration (inclusive). Can be string or `timedelta` object.
    nullable
        Whether the column can contain null values. Default is `False`.
    null_probability
        Probability of generating null when `nullable=True`. Default is `0.0`.
    unique
        Whether all values must be unique. Default is `False`.
    generator
        Custom callable that generates values. Overrides other settings.

    Returns
    -------
    DurationField
        A duration field specification.

    Examples
    --------
    ```python
    import pointblank as pb
    from datetime import timedelta

    schema = pb.Schema(
        elapsed=pb.duration_field(),
        session_length=pb.duration_field(
            min_duration=timedelta(minutes=1),
            max_duration=timedelta(hours=2)
        ),
    )

    data = schema.generate(n=100, seed=23)
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
