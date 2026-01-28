"""
Per-dtype value generators for synthetic data generation.
"""

from __future__ import annotations

import random
import string
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

from pointblank._utils import _is_lib_present
from pointblank.field import Field
from pointblank.generate.base import GeneratorConfig
from pointblank.generate.regex import generate_from_regex
from pointblank.locales import LocaleGenerator

if TYPE_CHECKING:
    pass

__all__ = ["generate_column", "generate_dataframe"]


# Integer dtype bounds
INTEGER_BOUNDS = {
    "Int8": (-(2**7), 2**7 - 1),
    "Int16": (-(2**15), 2**15 - 1),
    "Int32": (-(2**31), 2**31 - 1),
    "Int64": (-(2**63), 2**63 - 1),
    "UInt8": (0, 2**8 - 1),
    "UInt16": (0, 2**16 - 1),
    "UInt32": (0, 2**32 - 1),
    "UInt64": (0, 2**64 - 1),
}


def _get_locale_generator(country: str = "US", seed: int | None = None) -> LocaleGenerator:
    """Get a LocaleGenerator instance with the specified country."""
    return LocaleGenerator(country=country, seed=seed)


def _generate_integer(field: Field, rng: random.Random, generator: Any | None = None) -> int:
    """Generate a random integer value respecting field constraints."""
    dtype_min, dtype_max = INTEGER_BOUNDS.get(field.dtype, (-(2**63), 2**63 - 1))

    min_val = getattr(field, "min_val", None)
    max_val = getattr(field, "max_val", None)

    min_val = min_val if min_val is not None else dtype_min
    max_val = max_val if max_val is not None else dtype_max

    # Clamp to dtype bounds
    min_val = max(min_val, dtype_min)
    max_val = min(max_val, dtype_max)

    return rng.randint(int(min_val), int(max_val))


def _generate_float(field: Field, rng: random.Random, generator: Any | None = None) -> float:
    """Generate a random float value respecting field constraints."""
    min_val = getattr(field, "min_val", None)
    max_val = getattr(field, "max_val", None)

    min_val = min_val if min_val is not None else -1e10
    max_val = max_val if max_val is not None else 1e10

    return rng.uniform(float(min_val), float(max_val))


def _generate_string(
    field: Field, rng: random.Random, generator: LocaleGenerator | None = None
) -> str:
    """Generate a random string value respecting field constraints."""
    # If using a preset, delegate to locale generator
    preset = getattr(field, "preset", None)
    if preset is not None:
        if generator is None:
            raise ValueError("LocaleGenerator instance required for preset generation")
        return _generate_from_preset(preset, generator)

    # If using a pattern, generate from regex
    pattern = getattr(field, "pattern", None)
    if pattern is not None:
        return _generate_from_pattern(pattern, rng)

    # Otherwise generate random alphanumeric string
    min_length = getattr(field, "min_length", None)
    max_length = getattr(field, "max_length", None)
    min_len = min_length if min_length is not None else 1
    max_len = max_length if max_length is not None else 20

    length = rng.randint(min_len, max_len)
    chars = string.ascii_letters + string.digits
    return "".join(rng.choice(chars) for _ in range(length))


def _generate_from_preset(preset: str, generator: LocaleGenerator) -> str:
    """Generate a value using a LocaleGenerator preset."""
    # Map preset names to LocaleGenerator methods
    preset_mapping = {
        # Personal
        "name": generator.name,
        "first_name": generator.first_name,
        "last_name": generator.last_name,
        "email": generator.email,
        "phone_number": generator.phone_number,
        "address": generator.address,
        "city": generator.city,
        "state": generator.state,
        "country": generator.country,
        "postcode": generator.postcode,
        "latitude": generator.latitude,
        "longitude": generator.longitude,
        # Business
        "company": generator.company,
        "job": generator.job,
        "catch_phrase": generator.catch_phrase,
        # Internet
        "url": generator.url,
        "domain_name": generator.domain_name,
        "ipv4": generator.ipv4,
        "ipv6": generator.ipv6,
        "user_name": generator.user_name,
        "password": generator.password,
        # Text
        "text": generator.text,
        "sentence": generator.sentence,
        "paragraph": generator.paragraph,
        "word": generator.word,
        # Financial
        "credit_card_number": generator.credit_card_number,
        "iban": generator.iban,
        "currency_code": generator.currency_code,
        # Identifiers
        "uuid4": generator.uuid4,
        "ssn": generator.ssn,
        "license_plate": generator.license_plate,
        # Date/Time
        "date_this_year": generator.date_this_year,
        "date_this_decade": generator.date_this_decade,
        "time": generator.time,
        # Misc
        "color_name": generator.color_name,
        "file_name": generator.file_name,
        "file_extension": generator.file_extension,
        "mime_type": generator.mime_type,
    }

    generator = preset_mapping.get(preset)
    if generator is None:
        raise ValueError(f"Unknown preset: {preset}")

    return str(generator())


def _generate_from_pattern(pattern: str, rng: random.Random) -> str:
    """Generate a string matching the given regex pattern."""
    return generate_from_regex(pattern, rng)


def _generate_boolean(field: Field, rng: random.Random, generator: Any | None = None) -> bool:
    """Generate a random boolean value."""
    return rng.choice([True, False])


def _generate_date(field: Field, rng: random.Random, generator: Any | None = None) -> date:
    """Generate a random date value respecting field constraints."""
    min_date = getattr(field, "min_date", None)
    max_date = getattr(field, "max_date", None)

    # Default date range
    if min_date is None:
        min_date = date(2000, 1, 1)
    elif isinstance(min_date, str):
        min_date = date.fromisoformat(min_date)
    elif isinstance(min_date, datetime):
        min_date = min_date.date()

    if max_date is None:
        max_date = date(2030, 12, 31)
    elif isinstance(max_date, str):
        max_date = date.fromisoformat(max_date)
    elif isinstance(max_date, datetime):
        max_date = max_date.date()

    days_between = (max_date - min_date).days
    random_days = rng.randint(0, max(0, days_between))
    return min_date + timedelta(days=random_days)


def _generate_datetime(field: Field, rng: random.Random, generator: Any | None = None) -> datetime:
    """Generate a random datetime value respecting field constraints."""
    min_date = getattr(field, "min_date", None)
    max_date = getattr(field, "max_date", None)

    # Default datetime range
    if min_date is None:
        min_dt = datetime(2000, 1, 1, 0, 0, 0)
    elif isinstance(min_date, str):
        min_dt = datetime.fromisoformat(min_date)
    elif isinstance(min_date, date) and not isinstance(min_date, datetime):
        min_dt = datetime.combine(min_date, datetime.min.time())
    else:
        min_dt = min_date

    if max_date is None:
        max_dt = datetime(2030, 12, 31, 23, 59, 59)
    elif isinstance(max_date, str):
        max_dt = datetime.fromisoformat(max_date)
    elif isinstance(max_date, date) and not isinstance(max_date, datetime):
        max_dt = datetime.combine(max_date, datetime.max.time())
    else:
        max_dt = max_date

    seconds_between = int((max_dt - min_dt).total_seconds())
    random_seconds = rng.randint(0, max(0, seconds_between))
    return min_dt + timedelta(seconds=random_seconds)


def _generate_duration(field: Field, rng: random.Random, generator: Any | None = None) -> timedelta:
    """Generate a random duration value."""
    # Generate duration between 0 and 30 days by default
    seconds = rng.randint(0, 30 * 24 * 60 * 60)
    return timedelta(seconds=seconds)


def _generate_time(field: Field, rng: random.Random, generator: Any | None = None) -> str:
    """Generate a random time value as string."""
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


# Mapping from dtype to generator function
DTYPE_GENERATORS: dict[str, Callable[[Field, random.Random, Any | None], Any]] = {
    "Int8": _generate_integer,
    "Int16": _generate_integer,
    "Int32": _generate_integer,
    "Int64": _generate_integer,
    "UInt8": _generate_integer,
    "UInt16": _generate_integer,
    "UInt32": _generate_integer,
    "UInt64": _generate_integer,
    "Float32": _generate_float,
    "Float64": _generate_float,
    "String": _generate_string,
    "Boolean": _generate_boolean,
    "Date": _generate_date,
    "Datetime": _generate_datetime,
    "Duration": _generate_duration,
    "Time": _generate_time,
}


def _generate_value(field: Field, rng: random.Random, locale_gen: Any | None = None) -> Any:
    """Generate a single value for a field."""
    # Check for custom generator first
    if field.generator is not None:
        return field.generator()

    # Check for allowed values (categorical)
    allowed = getattr(field, "allowed", None)
    if allowed is not None:
        return rng.choice(allowed)

    # Use dtype-specific generator
    generator = DTYPE_GENERATORS.get(field.dtype)
    if generator is None:
        raise ValueError(f"No generator available for dtype: {field.dtype}")

    return generator(field, rng, locale_gen)


def _generate_unique_values(
    field: Field,
    n: int,
    rng: random.Random,
    locale_gen: Any | None = None,
    max_retries: int = 1000,
) -> list[Any]:
    """Generate n unique values for a field."""
    # Check if we can even generate enough unique values
    allowed = getattr(field, "allowed", None)
    if allowed is not None and len(allowed) < n:
        raise ValueError(
            f"Cannot generate {n} unique values from {len(allowed)} allowed values "
            f"for field with allowed={allowed}"
        )

    seen: set[Any] = set()
    values: list[Any] = []
    consecutive_retries = 0

    while len(values) < n:
        value = _generate_value(field, rng, locale_gen)

        # Handle unhashable types
        try:
            value_key = value
            if isinstance(value, (list, dict)):
                value_key = str(value)

            if value_key not in seen:
                seen.add(value_key)
                values.append(value)
                consecutive_retries = 0
            else:
                consecutive_retries += 1
                if consecutive_retries > max_retries:
                    raise ValueError(
                        f"Unable to generate {n} unique values after {max_retries} "
                        f"consecutive retries. Generated {len(values)} unique values. "
                        "Consider relaxing constraints or reducing n."
                    )
        except TypeError:
            # Unhashable type, just append (can't check uniqueness easily)
            values.append(value)

    return values


def generate_column(
    field: Field,
    config: GeneratorConfig,
) -> list[Any]:
    """
    Generate a list of values for a single column.

    Parameters
    ----------
    field
        The Field specification for the column.
    config
        Generation configuration.

    Returns
    -------
    list
        List of generated values.
    """
    # Set up random number generator
    rng = random.Random(config.seed)

    # Set up locale generator if needed
    locale_gen = None
    preset = getattr(field, "preset", None)
    if preset is not None:
        # Use config country
        locale_gen = _get_locale_generator(config.country, config.seed)

    # Generate values
    if field.unique:
        values = _generate_unique_values(
            field, config.n, rng, locale_gen, config.max_unique_retries
        )
    else:
        values = [_generate_value(field, rng, locale_gen) for _ in range(config.n)]

    # Apply null probability
    if field.nullable and field.null_probability > 0:
        null_rng = random.Random(config.seed + 1 if config.seed else None)
        values = [None if null_rng.random() < field.null_probability else v for v in values]

    return values


# Presets that should share coherent context across columns
ADDRESS_RELATED_PRESETS = {"city", "state", "postcode", "phone_number"}
PERSON_RELATED_PRESETS = {"name", "first_name", "last_name", "email", "user_name"}


def _get_coherence_needs(fields: dict[str, Field]) -> tuple[bool, bool]:
    """Check what coherence is needed for the given fields."""
    needs_address = False
    needs_person = False

    for field in fields.values():
        preset = getattr(field, "preset", None)
        if preset in ADDRESS_RELATED_PRESETS:
            needs_address = True
        if preset in PERSON_RELATED_PRESETS:
            needs_person = True

    return needs_address, needs_person


def _generate_column_with_row_context(
    field: Field,
    config: GeneratorConfig,
    locale_gen: LocaleGenerator | None,
) -> list[Any]:
    """
    Generate column values with per-row context (location and/or person).

    This is used when columns need to share coherent data per row.
    """
    rng = random.Random(config.seed)

    values = []
    for i in range(config.n):
        if locale_gen is not None:
            locale_gen.set_row(i)
        values.append(_generate_value(field, rng, locale_gen))

    # Apply null probability
    if field.nullable and field.null_probability > 0:
        null_rng = random.Random(config.seed + 1 if config.seed else None)
        values = [None if null_rng.random() < field.null_probability else v for v in values]

    return values


def generate_dataframe(
    fields: dict[str, Field],
    config: GeneratorConfig,
) -> Any:
    """
    Generate a DataFrame with the specified fields.

    Parameters
    ----------
    fields
        Dictionary mapping column names to Field specifications.
    config
        Generation configuration.

    Returns
    -------
    DataFrame
        Generated DataFrame in the format specified by config.output.
    """
    # Check what coherence is needed
    needs_address, needs_person = _get_coherence_needs(fields)
    needs_coherence = needs_address or needs_person

    # Set up shared locale generator if any coherence is needed
    shared_locale_gen = None
    if needs_coherence:
        shared_locale_gen = _get_locale_generator(config.country, config.seed)
        if needs_address:
            shared_locale_gen.init_row_locations(config.n)
        if needs_person:
            shared_locale_gen.init_row_persons(config.n)

    # Determine which presets need row context
    coherent_presets = set()
    if needs_address:
        coherent_presets.update(ADDRESS_RELATED_PRESETS)
    if needs_person:
        coherent_presets.update(PERSON_RELATED_PRESETS)

    # Generate data for each column
    data: dict[str, list[Any]] = {}
    for col_name, field in fields.items():
        preset = getattr(field, "preset", None)

        # Use shared locale generator for coherent presets
        if needs_coherence and preset in coherent_presets:
            data[col_name] = _generate_column_with_row_context(field, config, shared_locale_gen)
        else:
            data[col_name] = generate_column(field, config)

    # Clean up
    if shared_locale_gen is not None:
        if needs_address:
            shared_locale_gen.clear_row_locations()
        if needs_person:
            shared_locale_gen.clear_row_persons()

    # Convert to requested output format
    if config.output == "dict":
        return data

    if config.output == "polars":
        if not _is_lib_present("polars"):
            raise ImportError(
                "The Polars library is not installed but is required when specifying "
                '`output="polars"`.'
            )
        import polars as pl

        return pl.DataFrame(data)

    if config.output == "pandas":
        if not _is_lib_present("pandas"):
            raise ImportError(
                "The Pandas library is not installed but is required when specifying "
                '`output="pandas"`.'
            )
        import pandas as pd

        return pd.DataFrame(data)

    raise ValueError(f"Unknown output format: {config.output}")
