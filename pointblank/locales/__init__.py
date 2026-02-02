"""
Country-based data generation for synthetic test data.

This module provides country-specific data generation without external dependencies.
It supports generating realistic names, addresses, emails, and other data types
with proper localization based on ISO 3166-1 country codes.
"""

from __future__ import annotations

import json
import random
import unicodedata
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

__all__ = [
    "LocaleRegistry",
    "LocaleGenerator",
    "get_generator",
    "COUNTRY_CODE_MAP",
    "COUNTRIES_WITH_FULL_DATA",
]


# ISO 3166-1 country code mappings
# Maps alpha-2 (2-letter) and alpha-3 (3-letter) codes to internal data directory names
COUNTRY_CODE_MAP: dict[str, str] = {
    # United States
    "US": "US",
    "USA": "US",
    # United Kingdom
    "GB": "GB",
    "GBR": "GB",
    "UK": "GB",  # Common alias
    # Ireland
    "IE": "IE",
    "IRL": "IE",
    # Iceland
    "IS": "IS",
    "ISL": "IS",
    # Australia
    "AU": "AU",
    "AUS": "AU",
    # Argentina
    "AR": "AR",
    "ARG": "AR",
    # Canada
    "CA": "CA",
    "CAN": "CA",
    # Germany
    "DE": "DE",
    "DEU": "DE",
    # Greece
    "GR": "GR",
    "GRC": "GR",
    # Austria
    "AT": "AT",
    "AUT": "AT",
    # Switzerland
    "CH": "CH",
    "CHE": "CH",
    # Chile
    "CL": "CL",
    "CHL": "CL",
    # France
    "FR": "FR",
    "FRA": "FR",
    # Spain
    "ES": "ES",
    "ESP": "ES",
    # Mexico
    "MX": "MX",
    "MEX": "MX",
    # Malta
    "MT": "MT",
    "MLT": "MT",
    # Portugal
    "PT": "PT",
    "PRT": "PT",
    # Brazil
    "BR": "BR",
    "BRA": "BR",
    # India
    "IN": "IN",
    "IND": "IN",
    # Italy
    "IT": "IT",
    "ITA": "IT",
    # Netherlands
    "NL": "NL",
    "NLD": "NL",
    # Belgium
    "BE": "BE",
    "BEL": "BE",
    # Bulgaria
    "BG": "BG",
    "BGR": "BG",
    # Poland
    "PL": "PL",
    "POL": "PL",
    # Romania
    "RO": "RO",
    "ROU": "RO",
    # Russia
    "RU": "RU",
    "RUS": "RU",
    # Slovakia
    "SK": "SK",
    "SVK": "SK",
    # Slovenia
    "SI": "SI",
    "SVN": "SI",
    # Japan
    "JP": "JP",
    "JPN": "JP",
    # South Korea
    "KR": "KR",
    "KOR": "KR",
    # Latvia
    "LV": "LV",
    "LVA": "LV",
    # Lithuania
    "LT": "LT",
    "LTU": "LT",
    # Luxembourg
    "LU": "LU",
    "LUX": "LU",
    # China
    "CN": "CN",
    "CHN": "CN",
    # Colombia
    "CO": "CO",
    "COL": "CO",
    # Cyprus
    "CY": "CY",
    "CYP": "CY",
    # Czech Republic
    "CZ": "CZ",
    "CZE": "CZ",
    # Estonia
    "EE": "EE",
    "EST": "EE",
    # Hong Kong
    "HK": "HK",
    "HKG": "HK",
    # Croatia
    "HR": "HR",
    "HRV": "HR",
    # Hungary
    "HU": "HU",
    "HUN": "HU",
    # Indonesia
    "ID": "ID",
    "IDN": "ID",
    # Taiwan
    "TW": "TW",
    "TWN": "TW",
    # Turkey
    "TR": "TR",
    "TUR": "TR",
    # New Zealand
    "NZ": "NZ",
    "NZL": "NZ",
    # Philippines
    "PH": "PH",
    "PHL": "PH",
}

# Countries that have complete locale data files
# These are the ISO alpha-2 codes for countries with full address, company,
# internet, misc, person, and text JSON files in the data directory
COUNTRIES_WITH_FULL_DATA: list[str] = [
    "US",  # United States
    "AR",  # Argentina
    "AT",  # Austria
    "AU",  # Australia
    "BE",  # Belgium
    "BG",  # Bulgaria
    "BR",  # Brazil
    "CA",  # Canada
    "CH",  # Switzerland
    "CL",  # Chile
    "CN",  # China
    "CO",  # Colombia
    "CY",  # Cyprus
    "CZ",  # Czech Republic
    "DE",  # Germany
    "DK",  # Denmark
    "EE",  # Estonia
    "ES",  # Spain
    "FI",  # Finland
    "FR",  # France
    "GB",  # United Kingdom
    "GR",  # Greece
    "HK",  # Hong Kong
    "HR",  # Croatia
    "HU",  # Hungary
    "ID",  # Indonesia
    "IE",  # Ireland
    "IN",  # India
    "IS",  # Iceland
    "IT",  # Italy
    "JP",  # Japan
    "KR",  # South Korea
    "LV",  # Latvia
    "LT",  # Lithuania
    "LU",  # Luxembourg
    "MT",  # Malta
    "MX",  # Mexico
    "NL",  # Netherlands
    "NO",  # Norway
    "NZ",  # New Zealand
    "PL",  # Poland
    "PH",  # Philippines
    "PT",  # Portugal
    "RO",  # Romania
    "RU",  # Russia
    "SE",  # Sweden
    "SK",  # Slovakia
    "SI",  # Slovenia
    "TR",  # Turkey
    "TW",  # Taiwan
]

# Fallback chains for countries (when a country's data is incomplete)
COUNTRY_FALLBACKS: dict[str, list[str]] = {
    # English-speaking countries fall back to US
    "GB": ["GB", "US"],
    "IE": ["IE", "GB", "US"],
    "AU": ["AU", "GB", "US"],
    "CA": ["CA", "US"],
    # German-speaking countries
    "DE": ["DE", "US"],
    "AT": ["AT", "DE", "US"],
    "CH": ["CH", "DE", "US"],
    # French-speaking
    "FR": ["FR", "US"],
    # Belgian (Dutch/French bilingual)
    "BE": ["BE", "NL", "FR", "US"],
    # Scandinavian
    "DK": ["DK", "DE", "US"],
    "NO": ["NO", "DK", "DE", "US"],
    "SE": ["SE", "DK", "DE", "US"],
    "FI": ["FI", "SE", "US"],
    # Spanish-speaking
    "ES": ["ES", "US"],
    "MX": ["MX", "ES", "US"],
    # Portuguese-speaking
    "PT": ["PT", "US"],
    "BR": ["BR", "PT", "US"],
    # Other European
    "IT": ["IT", "US"],
    "NL": ["NL", "US"],
    "PL": ["PL", "US"],
    "RU": ["RU", "US"],
    # Asian countries
    "JP": ["JP", "US"],
    "KR": ["KR", "US"],
    "CN": ["CN", "US"],
    "TW": ["TW", "CN", "US"],
    # Turkey
    "TR": ["TR", "US"],
}


@dataclass
class LocaleData:
    """Container for all locale-specific data."""

    locale: str
    person: dict[str, Any] = field(default_factory=dict)
    address: dict[str, Any] = field(default_factory=dict)
    company: dict[str, Any] = field(default_factory=dict)
    internet: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)
    misc: dict[str, Any] = field(default_factory=dict)


# Transliteration map for special characters (umlauts add 'e', others simplified)
_TRANSLITERATION_MAP: dict[str, str] = {
    # German umlauts -> add 'e'
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "Ä": "Ae",
    "Ö": "Oe",
    "Ü": "Ue",
    "ß": "ss",
    # Scandinavian
    "å": "aa",
    "Å": "Aa",
    "ø": "oe",
    "Ø": "Oe",
    "æ": "ae",
    "Æ": "Ae",
    # French/Spanish/Portuguese/Italian accents
    "à": "a",
    "á": "a",
    "â": "a",
    "ã": "a",
    "À": "A",
    "Á": "A",
    "Â": "A",
    "Ã": "A",
    "è": "e",
    "é": "e",
    "ê": "e",
    "ë": "e",
    "È": "E",
    "É": "E",
    "Ê": "E",
    "Ë": "E",
    "ì": "i",
    "í": "i",
    "î": "i",
    "ï": "i",
    "Ì": "I",
    "Í": "I",
    "Î": "I",
    "Ï": "I",
    "ò": "o",
    "ó": "o",
    "ô": "o",
    "õ": "o",
    "Ò": "O",
    "Ó": "O",
    "Ô": "O",
    "Õ": "O",
    "ù": "u",
    "ú": "u",
    "û": "u",
    "Ù": "U",
    "Ú": "U",
    "Û": "U",
    "ñ": "n",
    "Ñ": "N",
    "ç": "c",
    "Ç": "C",
    "ý": "y",
    "ÿ": "y",
    "Ý": "Y",
    # Eastern European
    "ł": "l",
    "Ł": "L",
    "ń": "n",
    "Ń": "N",
    "ś": "s",
    "Ś": "S",
    "ź": "z",
    "Ź": "Z",
    "ż": "z",
    "Ż": "Z",
    "ć": "c",
    "Ć": "C",
    "ě": "e",
    "Ě": "E",
    "š": "s",
    "Š": "S",
    "č": "c",
    "Č": "C",
    "ř": "r",
    "Ř": "R",
    "ž": "z",
    "Ž": "Z",
    "ů": "u",
    "Ů": "U",
    "ď": "d",
    "Ď": "D",
    "ť": "t",
    "Ť": "T",
    "ň": "n",
    "Ň": "N",
    # Other
    "đ": "d",
    "Đ": "D",
    "ğ": "g",
    "Ğ": "G",
    "ı": "i",
    "İ": "I",
    "ş": "s",
    "Ş": "S",
    "ț": "t",
    "Ț": "T",
    "ă": "a",
    "Ă": "A",
}


def _transliterate_to_ascii(text: str) -> str:
    """
    Transliterate text to ASCII-safe characters for email addresses and usernames.

    Handles German umlauts specially (ü -> ue, ö -> oe, ä -> ae) and converts
    other accented characters to their base ASCII equivalents.

    Parameters
    ----------
    text
        The text to transliterate.

    Returns
    -------
    str
        ASCII-safe version of the text.
    """
    # First apply our custom transliteration map
    result = []
    for char in text:
        if char in _TRANSLITERATION_MAP:
            result.append(_TRANSLITERATION_MAP[char])
        else:
            result.append(char)
    text = "".join(result)

    # Then use unicodedata to handle any remaining non-ASCII characters
    # NFD decomposes characters (e.g., é -> e + combining accent)
    # We then filter to keep only ASCII characters
    normalized = unicodedata.normalize("NFD", text)
    ascii_text = "".join(c for c in normalized if unicodedata.category(c) != "Mn")

    return ascii_text


def _normalize_country(country: str) -> str:
    """
    Normalize a country code to the standard 2-letter ISO 3166-1 alpha-2 format.

    Parameters
    ----------
    country
        Country code in alpha-2 (US), alpha-3 (USA), or legacy locale format (en_US).

    Returns
    -------
    str
        The normalized 2-letter country code.

    Raises
    ------
    ValueError
        If the country code is not recognized.
    """
    # Uppercase and strip whitespace
    code = country.strip().upper()

    # Handle legacy locale format (en_US, de-DE, etc.)
    if "_" in code or "-" in code:
        # Extract country part from locale code
        parts = code.replace("-", "_").split("_")
        if len(parts) == 2:
            code = parts[1]  # Take the country part (US from en_US)

    # Look up in the country code map
    if code in COUNTRY_CODE_MAP:
        return COUNTRY_CODE_MAP[code]

    # If already a valid 2-letter code in fallbacks, use it
    if code in COUNTRY_FALLBACKS:
        return code

    # Default to US with a warning (or raise an error)
    raise ValueError(
        f"Unknown country code: {country!r}. "
        f"Supported codes: {', '.join(sorted(set(COUNTRY_CODE_MAP.keys())))}"
    )


class LocaleRegistry:
    """Registry for country data with fallback support."""

    _instance: LocaleRegistry | None = None
    _cache: dict[str, LocaleData]

    def __new__(cls) -> LocaleRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    def get(self, country: str) -> LocaleData:
        """
        Get country data with fallback chain.

        Parameters
        ----------
        country
            Country code (e.g., "US", "DE", "USA", "DEU").
            Also accepts legacy locale codes like "en_US" for backwards compatibility.

        Returns
        -------
        LocaleData
            The country data, falling back to parent countries if needed.
        """
        # Normalize to 2-letter country code
        country_code = _normalize_country(country)

        if country_code in self._cache:
            return self._cache[country_code]

        # Get fallback chain
        fallback_chain = COUNTRY_FALLBACKS.get(country_code, [country_code, "US"])
        if country_code not in fallback_chain:
            fallback_chain = [country_code] + fallback_chain

        # Load data with fallback
        locale_data = self._load_with_fallback(fallback_chain)
        self._cache[country_code] = locale_data
        return locale_data

    def _load_with_fallback(self, fallback_chain: list[str]) -> LocaleData:
        """Load country data, falling back through the chain."""
        merged_data = LocaleData(locale=fallback_chain[0])

        # Load shared/universal data first (e.g., file extensions, MIME types)
        shared_data = self._load_country_files("_shared")
        if shared_data:
            self._merge_data(merged_data, shared_data)

        # Load in reverse order so more specific countries override
        for country in reversed(fallback_chain):
            data = self._load_country_files(country)
            if data:
                self._merge_data(merged_data, data)

        return merged_data

    def _load_country_files(self, country: str) -> dict[str, Any] | None:
        """Load all data files for a country."""
        try:
            data_path = files("pointblank.locales.data") / country
            if not data_path.is_dir():
                return None

            result: dict[str, Any] = {}
            for category in ["person", "address", "company", "internet", "text", "misc"]:
                file_path = data_path / f"{category}.json"
                try:
                    content = file_path.read_text(encoding="utf-8")
                    result[category] = json.loads(content)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass

            return result if result else None
        except (TypeError, FileNotFoundError):
            return None

    def _merge_data(self, target: LocaleData, source: dict[str, Any]) -> None:
        """Merge source data into target LocaleData."""
        for category, data in source.items():
            if hasattr(target, category):
                existing = getattr(target, category)
                if isinstance(existing, dict) and isinstance(data, dict):
                    existing.update(data)
                else:
                    setattr(target, category, data)

    def clear_cache(self) -> None:
        """Clear the country data cache."""
        self._cache.clear()


class LocaleGenerator:
    """
    Generator for country-specific test data.

    This class provides methods to generate realistic data like names, emails,
    addresses, etc. based on country-specific patterns and data.
    """

    def __init__(self, country: str = "US", seed: int | None = None):
        """
        Initialize the country data generator.

        Parameters
        ----------
        country
            Country code (e.g., "US", "DE", "USA", "DEU").
            Also accepts legacy locale codes like "en_US" for backwards compatibility.
        seed
            Random seed for reproducibility.
        """
        self.country_code = _normalize_country(country)
        self.rng = random.Random(seed)
        self._registry = LocaleRegistry()
        self._data = self._registry.get(self.country_code)

    def seed(self, seed: int) -> None:
        """Set the random seed."""
        self.rng.seed(seed)

    # =========================================================================
    # Person
    # =========================================================================

    _current_person: dict[str, str] | None = None
    _row_persons: list[dict[str, str]] | None = None

    def _get_person(self, gender: str | None = None) -> dict[str, str]:
        """Get a coherent person (first_name, last_name, gender) from the data."""
        # If no gender specified, randomly select one (weighted toward male/female)
        if gender is None:
            gender = self.rng.choice(["male", "female"])

        return {
            "first_name": self._generate_first_name(gender),
            "last_name": self._generate_last_name(),
            "gender": gender,
        }

    def _generate_first_name(self, gender: str | None = None) -> str:
        """Generate a random first name (internal, no caching)."""
        names = self._data.person.get("first_names", {})

        if gender and gender in names:
            name_list = names[gender]
        elif "neutral" in names:
            # Combine all available names
            all_names = []
            for category in ["male", "female", "neutral"]:
                all_names.extend(names.get(category, []))
            name_list = all_names if all_names else ["Alex"]
        else:
            # Flatten all categories
            all_names = []
            for category_names in names.values():
                if isinstance(category_names, list):
                    all_names.extend(category_names)
            name_list = all_names if all_names else ["Alex"]

        return self.rng.choice(name_list)

    def _generate_last_name(self) -> str:
        """Generate a random last name (internal, no caching)."""
        names = self._data.person.get("last_names", ["Smith"])
        return self.rng.choice(names)

    def init_row_persons(self, n_rows: int) -> None:
        """
        Pre-generate person data for multiple rows to ensure coherence across columns.

        This should be called before generating a dataset with person-related columns.
        When active, first_name(), last_name(), name(), email() will use the person
        for the current row (set via set_row()).

        Parameters
        ----------
        n_rows
            Number of rows to pre-generate persons for.
        """
        self._row_persons = [self._get_person() for _ in range(n_rows)]

    def clear_row_persons(self) -> None:
        """Clear all pre-generated row persons."""
        self._row_persons = None

    def new_person(self, gender: str | None = None) -> dict[str, str]:
        """
        Select a new random person and cache it for coherent generation.

        Call this before generating related person components (first_name, last_name, email)
        to ensure they all refer to the same person.

        Returns
        -------
        dict
            The selected person with first_name and last_name.
        """
        self._current_person = self._get_person(gender)
        return self._current_person

    def _get_current_person(self) -> dict[str, str]:
        """Get the current cached person, or select a new one."""
        # If row persons are active, use those
        if self._row_persons is not None and self._current_row is not None:
            return self._row_persons[self._current_row]
        # Otherwise use single cached person
        if self._current_person is None:
            self._current_person = self._get_person()
        return self._current_person

    def clear_person(self) -> None:
        """Clear the cached person so the next call will select a new one."""
        self._current_person = None

    def first_name(self, gender: str | None = None) -> str:
        """Generate a random first name (coherent with current person context)."""
        person = self._get_current_person()
        return person.get("first_name", "Alex")

    def last_name(self) -> str:
        """Generate a random last name (coherent with current person context)."""
        person = self._get_current_person()
        return person.get("last_name", "Smith")

    def name(self, gender: str | None = None) -> str:
        """Generate a simple full name (first + last, coherent with current person context).

        For names with prefixes (Mr., Ms., Dr., etc.) and occasional suffixes (Jr., III),
        use name_full() instead.
        """
        person = self._get_current_person()
        first = person.get("first_name", "Alex")
        last = person.get("last_name", "Smith")

        # Check if locale uses "last first" order (e.g., Japanese)
        formats = self._data.person.get("name_formats", ["{first_name} {last_name}"])
        # Use the simplest format (usually first one, which is typically "first last" or "last first")
        if formats and "{last_name} {first_name}" in formats[0]:
            return f"{last} {first}"
        return f"{first} {last}"

    def name_full(self, gender: str | None = None) -> str:
        """Generate a full name with optional prefix and rare suffix.

        Includes honorific prefixes with realistic frequencies:
        - Common honorifics (Mr., Ms., Mrs., etc.): ~95% of names
        - Professional titles (Dr., Prof., Rev., etc.): ~5% of names

        Suffixes (Jr., II, III) appear very rarely (~1 in 2000).
        """
        person = self._get_current_person()
        first = person.get("first_name", "Alex")
        last = person.get("last_name", "Smith")

        # Get gender for prefix selection (from person context or parameter)
        person_gender = person.get("gender", "neutral")
        if gender:
            person_gender = gender

        # Professional titles are rare (~2-3% of population for Dr., ~0.5% for Prof.)
        # These should appear infrequently
        professional_titles = {
            "Dr.",
            "Prof.",
            "Professor",
            "Rev.",
            "Pr.",
            "Prof. Dr.",
            "Rabbi",
            "Father",
            "Sister",
            "Pastor",
            "Elder",
        }

        # Get prefix based on gender from locale data
        prefixes = self._data.person.get("prefixes", {})
        prefix_list = prefixes.get(person_gender, prefixes.get("neutral", []))

        # Separate common honorifics and professional titles from locale data
        locale_common = [p for p in prefix_list if p not in professional_titles]
        locale_professional = [p for p in prefix_list if p in professional_titles]

        # Select prefix with realistic probabilities
        # ~95% common honorific, ~5% professional title
        if locale_professional and self.rng.random() < 0.05:
            prefix = self.rng.choice(locale_professional)
        elif locale_common:
            prefix = self.rng.choice(locale_common)
        else:
            # Fallback defaults if no common prefixes in locale
            fallback = {"male": "Mr.", "female": "Ms.", "neutral": "Mr."}
            prefix = fallback.get(person_gender, "")

        # Get suffix - very rare (approximately 1/2000 chance)
        suffix = ""
        if self.rng.random() < 0.0005:  # 1 in 2000
            suffixes = self._data.person.get("suffixes", [])
            # Filter out empty strings
            suffixes = [s for s in suffixes if s]
            if suffixes:
                suffix = self.rng.choice(suffixes)

        # Check if locale uses "last first" order (e.g., Japanese)
        formats = self._data.person.get("name_formats", ["{first_name} {last_name}"])
        if formats and "{last_name} {first_name}" in formats[0]:
            # For "last first" cultures, prefix typically comes before everything
            parts = [prefix, last, first] if prefix else [last, first]
        else:
            parts = [prefix, first, last] if prefix else [first, last]

        if suffix:
            parts.append(suffix)

        return " ".join(parts)

    # =========================================================================
    # Address
    # =========================================================================

    _current_location: dict[str, str] | None = None
    _row_locations: list[dict[str, str]] | None = None
    _current_row: int | None = None

    def _get_location(self) -> dict[str, str]:
        """Get a coherent location (city, state, postcode_prefix) from the data."""
        locations = self._data.address.get("locations", [])
        if locations:
            return self.rng.choice(locations)
        # Fallback for old-style data
        return {
            "city": "Springfield",
            "state": "State",
            "state_abbr": "ST",
            "postcode_prefix": "000",
        }

    def init_row_locations(self, n_rows: int) -> None:
        """
        Pre-generate locations for multiple rows to ensure coherence across columns.

        This should be called before generating a dataset with address-related columns.
        When active, city(), state(), postcode() etc. will use the location for the
        current row (set via set_row()).

        Parameters
        ----------
        n_rows
            Number of rows to pre-generate locations for.
        """
        self._row_locations = [self._get_location() for _ in range(n_rows)]
        self._current_row = None

    def set_row(self, row_index: int) -> None:
        """
        Set the current row index for location-based generation.

        When row locations are initialized, this sets which row's location to use.

        Parameters
        ----------
        row_index
            The row index (0-based).
        """
        self._current_row = row_index

    def clear_row_locations(self) -> None:
        """Clear all pre-generated row locations."""
        self._row_locations = None
        self._current_row = None

    def new_location(self) -> dict[str, str]:
        """
        Select a new random location and cache it for coherent address generation.

        Call this before generating related address components (city, state, postcode)
        to ensure they all refer to the same location.

        Returns
        -------
        dict
            The selected location with city, state, state_abbr, and postcode_prefix.
        """
        self._current_location = self._get_location()
        return self._current_location

    def _get_current_location(self) -> dict[str, str]:
        """Get the current cached location, or select a new one."""
        # If row locations are active, use those
        if self._row_locations is not None and self._current_row is not None:
            return self._row_locations[self._current_row]
        # Otherwise use single cached location
        if self._current_location is None:
            self._current_location = self._get_location()
        return self._current_location

    def clear_location(self) -> None:
        """Clear the cached location so the next call will select a new one."""
        self._current_location = None

    def city(self) -> str:
        """Generate a random city name (coherent with current location context).

        Returns the exonym (English name) if available, otherwise the native city name.
        This allows addresses to use native names while city presets use international names.
        """
        location = self._get_current_location()
        # Prefer exonym (English name) for standalone city preset
        return location.get("exonym", location.get("city", "Springfield"))

    def _city_native(self) -> str:
        """Get the native city name (used internally for addresses).

        Always returns the native name, ignoring any exonym.
        """
        location = self._get_current_location()
        return location.get("city", "Springfield")

    def state(self, abbr: bool = False) -> str:
        """Generate a random state/province name (coherent with current location context)."""
        location = self._get_current_location()
        if abbr:
            return location.get("state_abbr", "ST")
        return location.get("state", "State")

    def country(self) -> str:
        """Generate the country name for this locale."""
        return self._data.address.get("country", "United States")

    def postcode(self) -> str:
        """Generate a random postal code (coherent with current location context)."""
        location = self._get_current_location()
        prefix = location.get("postcode_prefix", "")
        postcode_format = self._data.address.get("postcode_format", "")

        # If format uses pattern characters (? for letter, # for digit), generate accordingly
        if "?" in postcode_format or "#" in postcode_format:
            # Generate the full postcode from the format pattern
            # Replace ? with random uppercase letter, # with random digit
            result = []
            prefix_idx = 0
            for char in postcode_format:
                if char == "?":
                    # Use prefix character if available, otherwise random letter
                    if prefix_idx < len(prefix) and prefix[prefix_idx].isalpha():
                        result.append(prefix[prefix_idx])
                        prefix_idx += 1
                    else:
                        result.append(self.rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
                elif char == "#":
                    # Use prefix character if available, otherwise random digit
                    if prefix_idx < len(prefix) and prefix[prefix_idx].isdigit():
                        result.append(prefix[prefix_idx])
                        prefix_idx += 1
                    else:
                        result.append(str(self.rng.randint(0, 9)))
                else:
                    # Keep literal characters (spaces, dashes, etc.)
                    result.append(char)
            return "".join(result)

        # Default: append digits to complete the postal code
        remaining = 5 - len(prefix)
        suffix = "".join(str(self.rng.randint(0, 9)) for _ in range(remaining))
        return prefix + suffix

    def street_name(self) -> str:
        """Generate a random street name.

        If the locale has `streets_by_city`, use city-specific streets.
        Otherwise, fall back to combining `street_names` and `street_suffixes`.
        """
        # Check if locale uses city-specific streets
        streets_by_city = self._data.address.get("streets_by_city")
        if streets_by_city:
            # Get current city from location
            location = self._get_current_location()
            city = location.get("city", "")
            city_streets = streets_by_city.get(city)
            if city_streets:
                return self.rng.choice(city_streets)

        # Fall back to old street_names + street_suffixes approach
        names = self._data.address.get("street_names", ["Main"])
        suffixes = self._data.address.get("street_suffixes", ["St"])
        return f"{self.rng.choice(names)} {self.rng.choice(suffixes)}"

    def building_number(self) -> str:
        """Generate a random building number."""
        return str(self.rng.randint(1, 9999))

    def address(self) -> str:
        """Generate a full coherent address (city, state, postcode are consistent)."""
        # Only select a new location if row locations are not active
        # This ensures coherence with other address-related columns (city, state, etc.)
        using_row_context = self._row_locations is not None and self._current_row is not None
        if not using_row_context:
            self.new_location()

        formats = self._data.address.get(
            "address_formats",
            ["{building_number} {street}, {city}, {state} {postcode}"],
        )
        fmt = self.rng.choice(formats)

        result = fmt.format(
            building_number=self.building_number(),
            street=self.street_name(),
            city=self._city_native(),  # Use native name in addresses
            state=self.state(abbr=False),
            state_abbr=self.state(abbr=True),
            postcode=self.postcode(),
            country=self.country(),
            unit=str(self.rng.randint(1, 999)),
        )

        # Clear location after generating full address (only if we set it)
        if not using_row_context:
            self.clear_location()
        return result

    def phone_number(self) -> str:
        """Generate a phone number with area code matching the current location's state."""
        location = self._get_current_location()
        state = location.get("state", "California")

        # Get area codes for this state
        area_codes = self._data.address.get("phone_area_codes", {})
        state_codes = area_codes.get(state, ["555"])  # 555 is fictional fallback
        area_code = self.rng.choice(state_codes)

        # Generate the rest of the number
        exchange = str(self.rng.randint(200, 999))  # Exchange can't start with 0 or 1
        subscriber = str(self.rng.randint(0, 9999)).zfill(4)

        return f"({area_code}) {exchange}-{subscriber}"

    def latitude(self) -> str:
        """Generate a random latitude (bounded by current location if available)."""
        location = self._get_current_location()
        lat_min = location.get("lat_min", -90)
        lat_max = location.get("lat_max", 90)
        return f"{self.rng.uniform(lat_min, lat_max):.6f}"

    def longitude(self) -> str:
        """Generate a random longitude (bounded by current location if available)."""
        location = self._get_current_location()
        lon_min = location.get("lon_min", -180)
        lon_max = location.get("lon_max", 180)
        return f"{self.rng.uniform(lon_min, lon_max):.6f}"

    # =========================================================================
    # Company
    # =========================================================================

    def company(self) -> str:
        """Generate a random company name.

        Has a ~15% chance to return a well-known company name, with preference
        for companies that have offices in the current city (if location context is active).
        Otherwise generates a fictional company name.
        """
        # 15% chance to use a well-known company
        if self.rng.random() < 0.15:
            well_known = self._data.company.get("well_known_companies", [])
            if well_known:
                # Get current city if location context is active
                current_city = None
                if self._row_locations is not None and self._current_row is not None:
                    current_city = self._row_locations[self._current_row].get("city")

                # Collect all companies, preferring those in the current city
                city_companies = []
                all_companies = []

                for company in well_known:
                    name = company.get("name") if isinstance(company, dict) else company
                    cities = company.get("cities", []) if isinstance(company, dict) else []
                    all_companies.append(name)
                    if current_city and current_city in cities:
                        city_companies.append(name)

                # 70% chance to use city-relevant company if available
                if city_companies and self.rng.random() < 0.7:
                    return self.rng.choice(city_companies)
                elif all_companies:
                    return self.rng.choice(all_companies)

        # Generate a fictional company name
        formats = self._data.company.get("formats", ["{last_name} {suffix}"])
        fmt = self.rng.choice(formats)

        suffixes = self._data.company.get("suffixes", ["Inc", "LLC", "Corp"])
        adjectives = self._data.company.get("adjectives", ["Global", "Advanced"])
        nouns = self._data.company.get("nouns", ["Solutions", "Systems"])

        # Count how many {last_name} placeholders are in the format
        # and generate distinct last names for each
        last_name_count = fmt.count("{last_name}")
        if last_name_count <= 1:
            company_last_name = self._generate_last_name()
            return fmt.format(
                last_name=company_last_name,
                suffix=self.rng.choice(suffixes),
                adjective=self.rng.choice(adjectives),
                noun=self.rng.choice(nouns),
            )
        else:
            # Generate distinct last names for formats like "{last_name} and {last_name}"
            last_names = []
            for _ in range(last_name_count):
                new_name = self._generate_last_name()
                # Ensure we don't repeat the same name
                attempts = 0
                while new_name in last_names and attempts < 10:
                    new_name = self._generate_last_name()
                    attempts += 1
                last_names.append(new_name)

            # Replace placeholders one at a time
            result = fmt
            for name in last_names:
                result = result.replace("{last_name}", name, 1)

            return result.format(
                suffix=self.rng.choice(suffixes),
                adjective=self.rng.choice(adjectives),
                noun=self.rng.choice(nouns),
            )

    def job(self) -> str:
        """Generate a random job title."""
        jobs = self._data.company.get("jobs", ["Manager"])
        return self.rng.choice(jobs)

    def catch_phrase(self) -> str:
        """Generate a random business catch phrase."""
        adjectives = self._data.company.get("catch_phrase_adjectives", ["Innovative", "Dynamic"])
        nouns = self._data.company.get("catch_phrase_nouns", ["solutions", "paradigms"])
        verbs = self._data.company.get("catch_phrase_verbs", ["deliver", "leverage"])
        return (
            f"{self.rng.choice(adjectives)} {self.rng.choice(nouns)} that {self.rng.choice(verbs)}"
        )

    # =========================================================================
    # Internet
    # =========================================================================

    def email(self) -> str:
        """Generate a random email address (coherent with current person context)."""
        # Get person data - uses cached person if available
        person = self._get_current_person()
        first = person.get("first_name", "user").lower()
        last = person.get("last_name", "name").lower()
        domains = self._data.internet.get("free_email_domains", ["gmail.com", "outlook.com"])

        # Transliterate to ASCII for valid email addresses
        first = _transliterate_to_ascii(first)
        last = _transliterate_to_ascii(last)

        # Clean names for email (remove non-alphanumeric)
        first = "".join(c for c in first if c.isalnum())
        last = "".join(c for c in last if c.isalnum())

        # Various realistic email patterns
        patterns = [
            f"{first}.{last}",  # john.smith
            f"{first}{last}",  # johnsmith
            f"{first}_{last}",  # john_smith
            f"{first[0]}{last}",  # jsmith
            f"{first}{self.rng.randint(1, 999)}",  # john123
            f"{first[0]}{last}{self.rng.randint(1, 99)}",  # jsmith42
            f"{first}.{last}{self.rng.randint(1, 99)}",  # john.smith99
            f"{first[0]}_{last}",  # j_smith
        ]

        return f"{self.rng.choice(patterns)}@{self.rng.choice(domains)}"

    def user_name(self) -> str:
        """Generate a random username (coherent with current person context)."""
        # Get person data - uses cached person if available
        person = self._get_current_person()
        first = person.get("first_name", "user").lower()
        last = person.get("last_name", "name").lower()

        # Transliterate to ASCII for valid usernames
        first = _transliterate_to_ascii(first)
        last = _transliterate_to_ascii(last)

        # Clean names
        first = "".join(c for c in first if c.isalnum())
        last = "".join(c for c in last if c.isalnum())

        patterns = [
            f"{first}{last}",
            f"{first}_{last}",
            f"{first}{self.rng.randint(1, 999)}",
            f"{first[0]}{last}{self.rng.randint(1, 99)}",
        ]

        return self.rng.choice(patterns)

    def password(self, length: int = 12) -> str:
        """Generate a random password."""
        import string

        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(self.rng.choice(chars) for _ in range(length))

    def url(self) -> str:
        """Generate a random URL."""
        protocols = ["https://"]
        tlds = self._data.internet.get("tlds", ["com", "org", "net"])
        words = self._data.text.get("words", ["example", "test", "sample"])

        domain = self.rng.choice(words).lower()
        domain = "".join(c for c in domain if c.isalnum())

        return f"{self.rng.choice(protocols)}www.{domain}.{self.rng.choice(tlds)}"

    def domain_name(self) -> str:
        """Generate a random domain name."""
        tlds = self._data.internet.get("tlds", ["com", "org", "net"])
        words = self._data.text.get("words", ["example", "test", "sample"])

        domain = self.rng.choice(words).lower()
        domain = "".join(c for c in domain if c.isalnum())

        return f"{domain}.{self.rng.choice(tlds)}"

    def ipv4(self) -> str:
        """Generate a random IPv4 address."""
        return ".".join(str(self.rng.randint(0, 255)) for _ in range(4))

    def ipv6(self) -> str:
        """Generate a random IPv6 address."""
        return ":".join(f"{self.rng.randint(0, 65535):04x}" for _ in range(8))

    # =========================================================================
    # Text
    # =========================================================================

    def word(self) -> str:
        """Generate a random word."""
        words = self._data.text.get("words", ["lorem", "ipsum", "dolor"])
        return self.rng.choice(words)

    def sentence(self, num_words: int | None = None) -> str:
        """Generate a random sentence."""
        if num_words is None:
            num_words = self.rng.randint(5, 15)

        words = [self.word() for _ in range(num_words)]
        words[0] = words[0].capitalize()
        return " ".join(words) + "."

    def paragraph(self, num_sentences: int | None = None) -> str:
        """Generate a random paragraph."""
        if num_sentences is None:
            num_sentences = self.rng.randint(3, 7)

        return " ".join(self.sentence() for _ in range(num_sentences))

    def text(self, max_chars: int = 200) -> str:
        """Generate random text up to max_chars."""
        result = []
        current_length = 0

        while current_length < max_chars:
            sentence = self.sentence()
            if current_length + len(sentence) + 1 > max_chars:
                break
            result.append(sentence)
            current_length += len(sentence) + 1

        return " ".join(result) if result else self.sentence()[:max_chars]

    # =========================================================================
    # Financial
    # =========================================================================

    def credit_card_number(self) -> str:
        """Generate a random credit card number (not valid for transactions)."""
        # Generate a 16-digit number with valid Luhn checksum
        prefix = self.rng.choice(["4", "5", "37", "6011"])  # Visa, MC, Amex, Discover
        length = 15 if prefix == "37" else 16

        # Generate digits (minus check digit)
        digits = list(prefix)
        while len(digits) < length - 1:
            digits.append(str(self.rng.randint(0, 9)))

        # Calculate Luhn check digit
        check_digit = self._luhn_checksum(digits)
        digits.append(str(check_digit))

        return "".join(digits)

    def _luhn_checksum(self, digits: list[str]) -> int:
        """Calculate Luhn check digit for a partial card number.

        The check digit is appended to make the full number pass the Luhn algorithm.
        We process from right to left, doubling every second digit starting from
        the rightmost digit of the partial number (since the check digit will be
        at position 0 and won't be doubled).
        """
        nums = [int(d) for d in digits]
        total = 0
        for i, d in enumerate(reversed(nums)):
            if i % 2 == 0:  # These positions get doubled (check digit at pos 0 won't be)
                d = d * 2
                if d > 9:
                    d -= 9
            total += d
        return (10 - (total % 10)) % 10

    def iban(self) -> str:
        """Generate a random IBAN."""
        # Simplified - generates a plausible-looking IBAN
        country = self._data.address.get("country_code", "US")
        if country == "US":
            # US doesn't use IBAN, use DE as example
            country = "DE"

        check_digits = f"{self.rng.randint(10, 99)}"
        bank_code = "".join(str(self.rng.randint(0, 9)) for _ in range(8))
        account = "".join(str(self.rng.randint(0, 9)) for _ in range(10))

        return f"{country}{check_digits}{bank_code}{account}"

    def currency_code(self) -> str:
        """Generate a random currency code."""
        codes = self._data.misc.get("currency_codes", ["USD", "EUR", "GBP", "JPY", "CNY"])
        return self.rng.choice(codes)

    # =========================================================================
    # Identifiers
    # =========================================================================

    def uuid4(self) -> str:
        """Generate a random UUID4."""
        import uuid

        # Use our RNG to generate deterministic UUIDs
        hex_chars = "0123456789abcdef"
        parts = [
            "".join(self.rng.choice(hex_chars) for _ in range(8)),
            "".join(self.rng.choice(hex_chars) for _ in range(4)),
            "4" + "".join(self.rng.choice(hex_chars) for _ in range(3)),  # Version 4
            self.rng.choice("89ab")
            + "".join(self.rng.choice(hex_chars) for _ in range(3)),  # Variant
            "".join(self.rng.choice(hex_chars) for _ in range(12)),
        ]
        return "-".join(parts)

    def ssn(self) -> str:
        """Generate a random SSN-like identifier."""
        # US format: XXX-XX-XXXX
        fmt = self._data.misc.get("ssn_format", "###-##-####")
        return self._generate_from_format(fmt)

    def license_plate(self) -> str:
        """Generate a random license plate."""
        fmt = self._data.misc.get("license_plate_format", "???-####")
        return self._generate_from_format(fmt)

    # =========================================================================
    # Date/Time (string representations)
    # =========================================================================

    def date_this_year(self) -> str:
        """Generate a random date from this year as ISO string."""
        from datetime import date, timedelta

        today = date.today()
        start = date(today.year, 1, 1)
        days = (today - start).days
        random_date = start + timedelta(days=self.rng.randint(0, max(days, 1)))
        return random_date.isoformat()

    def date_this_decade(self) -> str:
        """Generate a random date from this decade as ISO string."""
        from datetime import date, timedelta

        today = date.today()
        decade_start = (today.year // 10) * 10
        start = date(decade_start, 1, 1)
        days = (today - start).days
        random_date = start + timedelta(days=self.rng.randint(0, max(days, 1)))
        return random_date.isoformat()

    def time(self) -> str:
        """Generate a random time as string."""
        hour = self.rng.randint(0, 23)
        minute = self.rng.randint(0, 59)
        second = self.rng.randint(0, 59)
        return f"{hour:02d}:{minute:02d}:{second:02d}"

    # =========================================================================
    # Misc
    # =========================================================================

    def color_name(self) -> str:
        """Generate a random color name."""
        colors = self._data.misc.get(
            "colors",
            [
                "Red",
                "Blue",
                "Green",
                "Yellow",
                "Purple",
                "Orange",
                "Pink",
                "Brown",
                "Black",
                "White",
                "Gray",
                "Cyan",
                "Magenta",
            ],
        )
        return self.rng.choice(colors)

    def file_name(self) -> str:
        """Generate a random file name."""
        words = self._data.text.get("words", ["document", "file", "report"])
        extensions = self._data.misc.get("file_extensions", ["txt", "pdf", "doc", "xlsx"])
        word = self.rng.choice(words).lower()
        word = "".join(c for c in word if c.isalnum())
        return f"{word}.{self.rng.choice(extensions)}"

    def file_extension(self) -> str:
        """Generate a random file extension."""
        extensions = self._data.misc.get(
            "file_extensions", ["txt", "pdf", "doc", "xlsx", "png", "jpg"]
        )
        return self.rng.choice(extensions)

    def mime_type(self) -> str:
        """Generate a random MIME type."""
        mime_types = self._data.misc.get(
            "mime_types",
            [
                "text/plain",
                "text/html",
                "application/json",
                "application/pdf",
                "image/png",
                "image/jpeg",
            ],
        )
        return self.rng.choice(mime_types)

    # =========================================================================
    # Utilities
    # =========================================================================

    def _generate_from_format(self, fmt: str) -> str:
        """
        Generate a string from a format pattern.

        Patterns:
        - # = digit (0-9)
        - ? = uppercase letter (A-Z)
        - * = alphanumeric
        """
        result = []
        for char in fmt:
            if char == "#":
                result.append(str(self.rng.randint(0, 9)))
            elif char == "?":
                result.append(self.rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            elif char == "*":
                result.append(self.rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
            else:
                result.append(char)
        return "".join(result)


# Module-level convenience function
_default_registry = LocaleRegistry()


def get_generator(country: str = "US", seed: int | None = None) -> LocaleGenerator:
    """
    Get a country data generator instance.

    Parameters
    ----------
    country
        Country code (e.g., "US", "DE", "USA", "DEU").
        Also accepts legacy locale codes like "en_US" for backwards compatibility.
    seed
        Random seed for reproducibility.

    Returns
    -------
    LocaleGenerator
        A generator configured for the specified country.
    """
    return LocaleGenerator(country=country, seed=seed)
