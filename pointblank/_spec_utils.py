from __future__ import annotations

import re


def regex_email() -> str:
    """Regex pattern for email validation."""
    # Requires at least one dot in the domain part
    return r"^[a-zA-Z0-9.!#$%&'*+\/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"


def regex_url() -> str:
    """Regex pattern for URL validation."""
    # Simplified but comprehensive URL regex
    return r"^(https?|ftp):\/\/[^\s/$.?#].[^\s]*$"


def regex_phone() -> str:
    """Regex pattern for phone number validation."""
    # Matches various phone number formats - requires at least 7 digits total
    return r"^[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,9}([-\s\.]?[0-9]{1,9})+$"


def regex_ipv4_address() -> str:
    """Regex pattern for IPv4 address validation."""
    return r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"


def regex_ipv6_address() -> str:
    """Regex pattern for IPv6 address validation."""
    return r"^(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:))$"


def regex_mac() -> str:
    """Regex pattern for MAC address validation."""
    return r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$"


def regex_swift_bic() -> str:
    """Regex pattern for SWIFT/BIC code validation."""
    # Bank code (4 letters) + Country code (2 letters) + Location code (2 letters/digits) + optional Branch code (3 letters/digits)
    return r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$"


def regex_vin() -> str:
    """Regex pattern for VIN validation (basic format check)."""
    return r"^[A-HJ-NPR-Z0-9]{17}$"


def regex_credit_card_1() -> str:
    """Get first regex pattern for credit card validation."""
    return r"^[0-9\s\-]+$"


def regex_credit_card_2() -> str:
    """Get second regex pattern for credit card validation."""
    return r"^[0-9]{13,19}$"


def regex_iban(country: str | None = None) -> str:
    """
    Regex pattern for IBAN validation.

    Parameters
    ----------
    country
        Optional two or three-letter country code. If provided, returns country-specific pattern.
    """
    if country is None:
        # Generic IBAN pattern: 2 letters, 2 digits, up to 30 alphanumeric
        return r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$"

    # Country-specific patterns
    iban_patterns = {
        "AL": r"^AL[0-9]{10}[A-Z0-9]{16}$",
        "ALB": r"^AL[0-9]{10}[A-Z0-9]{16}$",
        "AD": r"^AD[0-9]{10}[A-Z0-9]{12}$",
        "AND": r"^AD[0-9]{10}[A-Z0-9]{12}$",
        "AT": r"^AT[0-9]{18}$",
        "AUT": r"^AT[0-9]{18}$",
        "BE": r"^BE[0-9]{14}$",
        "BEL": r"^BE[0-9]{14}$",
        "BA": r"^BA[0-9]{18}$",
        "BIH": r"^BA[0-9]{18}$",
        "BG": r"^BG[0-9]{2}[A-Z]{4}[0-9]{6}[A-Z0-9]{8}$",
        "BGR": r"^BG[0-9]{2}[A-Z]{4}[0-9]{6}[A-Z0-9]{8}$",
        "BR": r"^BR[0-9]{25}[A-Z]{1}[A-Z0-9]{1}$",
        "BRA": r"^BR[0-9]{25}[A-Z]{1}[A-Z0-9]{1}$",
        "HR": r"^HR[0-9]{19}$",
        "HRV": r"^HR[0-9]{19}$",
        "CY": r"^CY[0-9]{10}[A-Z0-9]{16}$",
        "CYP": r"^CY[0-9]{10}[A-Z0-9]{16}$",
        "CZ": r"^CZ[0-9]{22}$",
        "CZE": r"^CZ[0-9]{22}$",
        "DK": r"^DK[0-9]{16}$",
        "DNK": r"^DK[0-9]{16}$",
        "EE": r"^EE[0-9]{18}$",
        "EST": r"^EE[0-9]{18}$",
        "FO": r"^FO[0-9]{16}$",
        "FRO": r"^FO[0-9]{16}$",
        "FI": r"^FI[0-9]{16}$",
        "FIN": r"^FI[0-9]{16}$",
        "FR": r"^FR[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "FRA": r"^FR[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "PF": r"^PF[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "PYF": r"^PF[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "TF": r"^TF[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "ATF": r"^TF[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "DE": r"^DE[0-9]{20}$",
        "DEU": r"^DE[0-9]{20}$",
        "GI": r"^GI[0-9]{2}[A-Z]{4}[A-Z0-9]{15}$",
        "GIB": r"^GI[0-9]{2}[A-Z]{4}[A-Z0-9]{15}$",
        "GE": r"^GE[0-9]{2}[A-Z]{2}[0-9]{16}$",
        "GEO": r"^GE[0-9]{2}[A-Z]{2}[0-9]{16}$",
        "GR": r"^GR[0-9]{9}[A-Z0-9]{16}$",
        "GRC": r"^GR[0-9]{9}[A-Z0-9]{16}$",
        "GL": r"^GL[0-9]{16}$",
        "GRL": r"^GL[0-9]{16}$",
        "HU": r"^HU[0-9]{26}$",
        "HUN": r"^HU[0-9]{26}$",
        "IS": r"^IS[0-9]{24}$",
        "ISL": r"^IS[0-9]{24}$",
        "IE": r"^IE[0-9]{2}[A-Z]{4}[0-9]{14}$",
        "IRL": r"^IE[0-9]{2}[A-Z]{4}[0-9]{14}$",
        "IL": r"^IL[0-9]{21}$",
        "ISR": r"^IL[0-9]{21}$",
        "IT": r"^IT[0-9]{2}[A-Z]{1}[0-9]{10}[A-Z0-9]{12}$",
        "ITA": r"^IT[0-9]{2}[A-Z]{1}[0-9]{10}[A-Z0-9]{12}$",
        "LV": r"^LV[0-9]{2}[A-Z]{4}[A-Z0-9]{13}$",
        "LVA": r"^LV[0-9]{2}[A-Z]{4}[A-Z0-9]{13}$",
        "LB": r"^LB[0-9]{6}[A-Z0-9]{20}$",
        "LBN": r"^LB[0-9]{6}[A-Z0-9]{20}$",
        "LI": r"^LI[0-9]{7}[A-Z0-9]{12}$",
        "LIE": r"^LI[0-9]{7}[A-Z0-9]{12}$",
        "LT": r"^LT[0-9]{18}$",
        "LTU": r"^LT[0-9]{18}$",
        "LU": r"^LU[0-9]{5}[A-Z0-9]{13}$",
        "LUX": r"^LU[0-9]{5}[A-Z0-9]{13}$",
        "MK": r"^MK[0-9]{5}[A-Z0-9]{10}[0-9]{2}$",
        "MKD": r"^MK[0-9]{5}[A-Z0-9]{10}[0-9]{2}$",
        "MT": r"^MT[0-9]{2}[A-Z]{4}[0-9]{5}[A-Z0-9]{18}$",
        "MLT": r"^MT[0-9]{2}[A-Z]{4}[0-9]{5}[A-Z0-9]{18}$",
        "MU": r"^MU[0-9]{2}[A-Z]{4}[0-9]{19}[A-Z]{3}$",
        "MUS": r"^MU[0-9]{2}[A-Z]{4}[0-9]{19}[A-Z]{3}$",
        "YT": r"^YT[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "MYT": r"^YT[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "MC": r"^MC[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "MCO": r"^MC[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "ME": r"^ME[0-9]{20}$",
        "MNE": r"^ME[0-9]{20}$",
        "NL": r"^NL[0-9]{2}[A-Z]{4}[0-9]{10}$",
        "NLD": r"^NL[0-9]{2}[A-Z]{4}[0-9]{10}$",
        "NC": r"^NC[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "NCL": r"^NC[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "NO": r"^NO[0-9]{13}$",
        "NOR": r"^NO[0-9]{13}$",
        "PL": r"^PL[0-9]{26}$",
        "POL": r"^PL[0-9]{26}$",
        "PT": r"^PT[0-9]{23}$",
        "PRT": r"^PT[0-9]{23}$",
        "RO": r"^RO[0-9]{2}[A-Z]{4}[A-Z0-9]{16}$",
        "ROU": r"^RO[0-9]{2}[A-Z]{4}[A-Z0-9]{16}$",
        "PM": r"^PM[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "SPM": r"^PM[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "SM": r"^SM[0-9]{2}[A-Z]{1}[0-9]{10}[A-Z0-9]{12}$",
        "SMR": r"^SM[0-9]{2}[A-Z]{1}[0-9]{10}[A-Z0-9]{12}$",
        "SA": r"^SA[0-9]{4}[A-Z0-9]{18}$",
        "SAU": r"^SA[0-9]{4}[A-Z0-9]{18}$",
        "RS": r"^RS[0-9]{20}$",
        "SRB": r"^RS[0-9]{20}$",
        "SK": r"^SK[0-9]{22}$",
        "SVK": r"^SK[0-9]{22}$",
        "SI": r"^SI[0-9]{17}$",
        "SVN": r"^SI[0-9]{17}$",
        "ES": r"^ES[0-9]{22}$",
        "ESP": r"^ES[0-9]{22}$",
        "SE": r"^SE[0-9]{22}$",
        "SWE": r"^SE[0-9]{22}$",
        "CH": r"^CH[0-9]{7}[A-Z0-9]{12}$",
        "CHE": r"^CH[0-9]{7}[A-Z0-9]{12}$",
        "TN": r"^TN[0-9]{22}$",
        "TUN": r"^TN[0-9]{22}$",
        "TR": r"^TR[0-9]{8}[A-Z0-9]{16}$",
        "TUR": r"^TR[0-9]{8}[A-Z0-9]{16}$",
        "GB": r"^GB[0-9]{2}[A-Z]{4}[0-9]{14}$",
        "GBR": r"^GB[0-9]{2}[A-Z]{4}[0-9]{14}$",
        "WF": r"^WF[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
        "WLF": r"^WF[0-9]{12}[A-Z0-9]{11}[0-9]{2}$",
    }

    return iban_patterns.get(country.upper(), r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$")


def regex_postal_code(country: str) -> str:
    """
    Get regex pattern for postal code validation for a specific country.

    Parameters
    ----------
    country
        Two or three-letter country code.
    """
    postal_patterns = {
        "AD": r"^AD[0-9]{3}$",
        "AND": r"^AD[0-9]{3}$",
        "AF": r"^[0-9]{4}$",
        "AFG": r"^[0-9]{4}$",
        "AI": r"^AI-2640$",
        "AIA": r"^AI-2640$",
        "AL": r"^[0-9]{4}$",
        "ALB": r"^[0-9]{4}$",
        "AM": r"^[0-9]{4}$",
        "ARM": r"^[0-9]{4}$",
        "AR": r"^([A-Z][0-9]{4}[A-Z]{3}|[A-Z][0-9]{4})$",
        "ARG": r"^([A-Z][0-9]{4}[A-Z]{3}|[A-Z][0-9]{4})$",
        "AS": r"^96799$",
        "ASM": r"^96799$",
        "AT": r"^[0-9]{4}$",
        "AUT": r"^[0-9]{4}$",
        "AU": r"^[0-9]{4}$",
        "AUS": r"^[0-9]{4}$",
        "AZ": r"^AZ[0-9]{4}$",
        "AZE": r"^AZ[0-9]{4}$",
        "BA": r"^[0-9]{5}$",
        "BIH": r"^[0-9]{5}$",
        "BB": r"^BB[0-9]{5}$",
        "BRB": r"^BB[0-9]{5}$",
        "BD": r"^[0-9]{4}$",
        "BGD": r"^[0-9]{4}$",
        "BE": r"^[0-9]{4}$",
        "BEL": r"^[0-9]{4}$",
        "BG": r"^[0-9]{4}$",
        "BGR": r"^[0-9]{4}$",
        "BH": r"^[0-9]{3,4}$",
        "BHR": r"^[0-9]{3,4}$",
        "BL": r"^97133$",
        "BLM": r"^97133$",
        "BM": r"^[A-Z]{2}\s?[0-9]{2}$",
        "BMU": r"^[A-Z]{2}\s?[0-9]{2}$",
        "BN": r"^[A-Z]{2}\s?[0-9]{4}$",
        "BRN": r"^[A-Z]{2}\s?[0-9]{4}$",
        "BR": r"^[0-9]{5}-?[0-9]{3}$",
        "BRA": r"^[0-9]{5}-?[0-9]{3}$",
        "BT": r"^[0-9]{5}$",
        "BTN": r"^[0-9]{5}$",
        "BY": r"^[0-9]{6}$",
        "BLR": r"^[0-9]{6}$",
        "CA": r"^[A-Z][0-9][A-Z]\s?[0-9][A-Z][0-9]$",
        "CAN": r"^[A-Z][0-9][A-Z]\s?[0-9][A-Z][0-9]$",
        "CC": r"^6799$",
        "CCK": r"^6799$",
        "CH": r"^[0-9]{4}$",
        "CHE": r"^[0-9]{4}$",
        "CL": r"^[0-9]{7}$",
        "CHL": r"^[0-9]{7}$",
        "CN": r"^[0-9]{6}$",
        "CHN": r"^[0-9]{6}$",
        "CO": r"^[0-9]{6}$",
        "COL": r"^[0-9]{6}$",
        "CR": r"^[0-9]{5}$",
        "CRI": r"^[0-9]{5}$",
        "CU": r"^[0-9]{5}$",
        "CUB": r"^[0-9]{5}$",
        "CV": r"^[0-9]{4}$",
        "CPV": r"^[0-9]{4}$",
        "CX": r"^6798$",
        "CXR": r"^6798$",
        "CY": r"^[0-9]{4}$",
        "CYP": r"^[0-9]{4}$",
        "CZ": r"^[0-9]{3}\s?[0-9]{2}$",
        "CZE": r"^[0-9]{3}\s?[0-9]{2}$",
        "DE": r"^[0-9]{5}$",
        "DEU": r"^[0-9]{5}$",
        "DK": r"^[0-9]{4}$",
        "DNK": r"^[0-9]{4}$",
        "DO": r"^[0-9]{5}$",
        "DOM": r"^[0-9]{5}$",
        "DZ": r"^[0-9]{5}$",
        "DZA": r"^[0-9]{5}$",
        "EC": r"^[0-9]{6}$",
        "ECU": r"^[0-9]{6}$",
        "EE": r"^[0-9]{5}$",
        "EST": r"^[0-9]{5}$",
        "EG": r"^[0-9]{5}$",
        "EGY": r"^[0-9]{5}$",
        "ES": r"^[0-9]{5}$",
        "ESP": r"^[0-9]{5}$",
        "ET": r"^[0-9]{4}$",
        "ETH": r"^[0-9]{4}$",
        "FI": r"^[0-9]{5}$",
        "FIN": r"^[0-9]{5}$",
        "FK": r"^FIQQ 1ZZ$",
        "FLK": r"^FIQQ 1ZZ$",
        "FM": r"^(96941|96942|96943|96944)$",
        "FSM": r"^(96941|96942|96943|96944)$",
        "FO": r"^[0-9]{3}$",
        "FRO": r"^[0-9]{3}$",
        "FR": r"^[0-9]{5}$",
        "FRA": r"^[0-9]{5}$",
        "GB": r"^([A-Z]{1,2}[0-9]{1,2}[A-Z]?)\s?([0-9][A-Z]{2})$",
        "GBR": r"^([A-Z]{1,2}[0-9]{1,2}[A-Z]?)\s?([0-9][A-Z]{2})$",
        "GF": r"^973[0-9]{2}$",
        "GUF": r"^973[0-9]{2}$",
        "GI": r"^GX11 1AA$",
        "GIB": r"^GX11 1AA$",
        "GL": r"^39[0-9]{2}$",
        "GRL": r"^39[0-9]{2}$",
        "GP": r"^971[0-9]{2}$",
        "GLP": r"^971[0-9]{2}$",
        "GR": r"^[0-9]{3}\s?[0-9]{2}$",
        "GRC": r"^[0-9]{3}\s?[0-9]{2}$",
        "GT": r"^[0-9]{5}$",
        "GTM": r"^[0-9]{5}$",
        "GU": r"^969[0-9]{2}$",
        "GUM": r"^969[0-9]{2}$",
        "HR": r"^[0-9]{5}$",
        "HRV": r"^[0-9]{5}$",
        "HT": r"^[0-9]{4}$",
        "HTI": r"^[0-9]{4}$",
        "HU": r"^[0-9]{4}$",
        "HUN": r"^[0-9]{4}$",
        "ID": r"^[0-9]{5}$",
        "IDN": r"^[0-9]{5}$",
        "IE": r"^[A-Z][0-9]{2}\s?[A-Z0-9]{4}$",
        "IRL": r"^[A-Z][0-9]{2}\s?[A-Z0-9]{4}$",
        "IN": r"^[0-9]{6}$",
        "IND": r"^[0-9]{6}$",
        "IO": r"^BBND 1ZZ$",
        "IOT": r"^BBND 1ZZ$",
        "IQ": r"^[0-9]{5}$",
        "IRQ": r"^[0-9]{5}$",
        "IR": r"^[0-9]{10}$",
        "IRN": r"^[0-9]{10}$",
        "IS": r"^[0-9]{3}$",
        "ISL": r"^[0-9]{3}$",
        "IT": r"^[0-9]{5}$",
        "ITA": r"^[0-9]{5}$",
        "JP": r"^[0-9]{3}-?[0-9]{4}$",
        "JPN": r"^[0-9]{3}-?[0-9]{4}$",
        "KR": r"^[0-9]{5}$",
        "KOR": r"^[0-9]{5}$",
        "KY": r"^KY[0-9]-[0-9]{4}$",
        "CYM": r"^KY[0-9]-[0-9]{4}$",
        "LI": r"^948[5-9]|949[0-7]$",
        "LIE": r"^948[5-9]|949[0-7]$",
        "LK": r"^[0-9]{5}$",
        "LKA": r"^[0-9]{5}$",
        "LT": r"^LT-?[0-9]{5}$",
        "LTU": r"^LT-?[0-9]{5}$",
        "LU": r"^[0-9]{4}$",
        "LUX": r"^[0-9]{4}$",
        "LV": r"^LV-?[0-9]{4}$",
        "LVA": r"^LV-?[0-9]{4}$",
        "MC": r"^980[0-9]{2}$",
        "MCO": r"^980[0-9]{2}$",
        "MD": r"^MD-?[0-9]{4}$",
        "MDA": r"^MD-?[0-9]{4}$",
        "MH": r"^(96960|96970)$",
        "MHL": r"^(96960|96970)$",
        "MK": r"^[0-9]{4}$",
        "MKD": r"^[0-9]{4}$",
        "MP": r"^9695[0-2]$",
        "MNP": r"^9695[0-2]$",
        "MQ": r"^972[0-9]{2}$",
        "MTQ": r"^972[0-9]{2}$",
        "MX": r"^[0-9]{5}$",
        "MEX": r"^[0-9]{5}$",
        "MY": r"^[0-9]{5}$",
        "MYS": r"^[0-9]{5}$",
        "NC": r"^988[0-9]{2}$",
        "NCL": r"^988[0-9]{2}$",
        "NE": r"^[0-9]{4}$",
        "NER": r"^[0-9]{4}$",
        "NF": r"^2899$",
        "NFK": r"^2899$",
        "NG": r"^[0-9]{6}$",
        "NGA": r"^[0-9]{6}$",
        "NI": r"^[0-9]{5}$",
        "NIC": r"^[0-9]{5}$",
        "NL": r"^[0-9]{4}\s?[A-Z]{2}$",
        "NLD": r"^[0-9]{4}\s?[A-Z]{2}$",
        "NO": r"^[0-9]{4}$",
        "NOR": r"^[0-9]{4}$",
        "NP": r"^[0-9]{5}$",
        "NPL": r"^[0-9]{5}$",
        "NZ": r"^[0-9]{4}$",
        "NZL": r"^[0-9]{4}$",
        "OM": r"^[0-9]{3}$",
        "OMN": r"^[0-9]{3}$",
        "PE": r"^([A-Z]{4,5}\s?[0-9]{2}|[0-9]{5})$",
        "PER": r"^([A-Z]{4,5}\s?[0-9]{2}|[0-9]{5})$",
        "PF": r"^987[0-9]{2}$",
        "PYF": r"^987[0-9]{2}$",
        "PG": r"^[0-9]{3}$",
        "PNG": r"^[0-9]{3}$",
        "PH": r"^[0-9]{4}$",
        "PHL": r"^[0-9]{4}$",
        "PK": r"^[0-9]{5}$",
        "PAK": r"^[0-9]{5}$",
        "PL": r"^[0-9]{2}-?[0-9]{3}$",
        "POL": r"^[0-9]{2}-?[0-9]{3}$",
        "PM": r"^97500$",
        "SPM": r"^97500$",
        "PN": r"^PCRN 1ZZ$",
        "PCN": r"^PCRN 1ZZ$",
        "PR": r"^00[679][0-9]{2}$",
        "PRI": r"^00[679][0-9]{2}$",
        "PT": r"^[0-9]{4}-?[0-9]{3}$",
        "PRT": r"^[0-9]{4}-?[0-9]{3}$",
        "PW": r"^96940$",
        "PLW": r"^96940$",
        "PY": r"^[0-9]{4}$",
        "PRY": r"^[0-9]{4}$",
        "RE": r"^974[0-9]{2}$",
        "REU": r"^974[0-9]{2}$",
        "RO": r"^[0-9]{6}$",
        "ROU": r"^[0-9]{6}$",
        "RS": r"^[0-9]{5}$",
        "SRB": r"^[0-9]{5}$",
        "RU": r"^[0-9]{6}$",
        "RUS": r"^[0-9]{6}$",
        "SA": r"^[0-9]{5}$",
        "SAU": r"^[0-9]{5}$",
        "SD": r"^[0-9]{5}$",
        "SDN": r"^[0-9]{5}$",
        "SE": r"^[0-9]{3}\s?[0-9]{2}$",
        "SWE": r"^[0-9]{3}\s?[0-9]{2}$",
        "SG": r"^[0-9]{6}$",
        "SGP": r"^[0-9]{6}$",
        "SH": r"^(STHL 1ZZ|ASCN 1ZZ)$",
        "SHN": r"^(STHL 1ZZ|ASCN 1ZZ)$",
        "SI": r"^[0-9]{4}$",
        "SVN": r"^[0-9]{4}$",
        "SJ": r"^[0-9]{4}$",
        "SJM": r"^[0-9]{4}$",
        "SK": r"^[0-9]{3}\s?[0-9]{2}$",
        "SVK": r"^[0-9]{3}\s?[0-9]{2}$",
        "SM": r"^4789[0-9]$",
        "SMR": r"^4789[0-9]$",
        "SN": r"^[0-9]{5}$",
        "SEN": r"^[0-9]{5}$",
        "SO": r"^[A-Z]{2}\s?[0-9]{5}$",
        "SOM": r"^[A-Z]{2}\s?[0-9]{5}$",
        "SV": r"^CP\s?[0-9]{4}$",
        "SLV": r"^CP\s?[0-9]{4}$",
        "SZ": r"^[A-Z][0-9]{3}$",
        "SWZ": r"^[A-Z][0-9]{3}$",
        "TC": r"^TKCA 1ZZ$",
        "TCA": r"^TKCA 1ZZ$",
        "TH": r"^[0-9]{5}$",
        "THA": r"^[0-9]{5}$",
        "TJ": r"^[0-9]{6}$",
        "TJK": r"^[0-9]{6}$",
        "TM": r"^[0-9]{6}$",
        "TKM": r"^[0-9]{6}$",
        "TN": r"^[0-9]{4}$",
        "TUN": r"^[0-9]{4}$",
        "TR": r"^[0-9]{5}$",
        "TUR": r"^[0-9]{5}$",
        "TW": r"^[0-9]{3}([0-9]{2})?$",
        "TWN": r"^[0-9]{3}([0-9]{2})?$",
        "TZ": r"^[0-9]{5}$",
        "TZA": r"^[0-9]{5}$",
        "UA": r"^[0-9]{5}$",
        "UKR": r"^[0-9]{5}$",
        "UM": r"^96898$",
        "UMI": r"^96898$",
        "US": r"^[0-9]{5}(-[0-9]{4})?$",
        "USA": r"^[0-9]{5}(-[0-9]{4})?$",
        "ZIP": r"^[0-9]{5}(-[0-9]{4})?$",  # Alias for US
        "UY": r"^[0-9]{5}$",
        "URY": r"^[0-9]{5}$",
        "UZ": r"^[0-9]{6}$",
        "UZB": r"^[0-9]{6}$",
        "VA": r"^00120$",
        "VAT": r"^00120$",
        "VC": r"^VC[0-9]{4}$",
        "VCT": r"^VC[0-9]{4}$",
        "VE": r"^[0-9]{4}$",
        "VEN": r"^[0-9]{4}$",
        "VG": r"^VG[0-9]{4}$",
        "VGB": r"^VG[0-9]{4}$",
        "VI": r"^008[0-9]{2}$",
        "VIR": r"^008[0-9]{2}$",
        "VN": r"^[0-9]{6}$",
        "VNM": r"^[0-9]{6}$",
        "WF": r"^986[0-9]{2}$",
        "WLF": r"^986[0-9]{2}$",
        "YT": r"^976[0-9]{2}$",
        "MYT": r"^976[0-9]{2}$",
        "ZA": r"^[0-9]{4}$",
        "ZAF": r"^[0-9]{4}$",
        "ZM": r"^[0-9]{5}$",
        "ZMB": r"^[0-9]{5}$",
    }

    return postal_patterns.get(country.upper(), r"^[0-9A-Z\s-]+$")


# Helper functions for string cleaning


def remove_hyphens(x: str, replacement: str = "") -> str:
    """Remove hyphens from a string."""
    return x.replace("-", replacement)


def remove_spaces(x: str, replacement: str = "") -> str:
    """Remove spaces from a string."""
    return x.replace(" ", replacement)


def remove_letters(x: str, replacement: str = "") -> str:
    """Remove letters from a string."""
    return re.sub(r"[a-zA-Z]", replacement, x)


def remove_punctuation(x: str, replacement: str = " ") -> str:
    """Remove punctuation from a string."""
    return re.sub(r"[^\w\s]", replacement, x)


# Validation functions


def is_isbn_10(x: str) -> bool:
    """
    Check if a string is a valid ISBN-10.

    Parameters
    ----------
    x
        String to validate.

    Returns
    -------
    bool
        True if valid ISBN-10, False otherwise.
    """
    x = remove_hyphens(x)
    x = remove_punctuation(x)
    x = x.lower()
    x = remove_spaces(x)

    if not re.match(r"\d{9}[0-9x]", x):
        return False

    digits = list(x)

    # If the check digit is "x" then substitute that for "10"
    if digits[9] == "x":
        digits[9] = "10"

    # Recast as integer values
    try:
        digits = [int(d) for d in digits]
    except ValueError:
        return False

    # The sum of vector multiplication of digits by the digit
    # weights (10 to 1 across the digits) should be
    # divided evenly by 11 for this to be a valid ISBN-10
    return sum(d * w for d, w in zip(digits, range(10, 0, -1))) % 11 == 0


def is_isbn_13(x: str) -> bool:
    """
    Check if a string is a valid ISBN-13.

    Parameters
    ----------
    x
        String to validate.

    Returns
    -------
    bool
        True if valid ISBN-13, False otherwise.
    """
    x = remove_hyphens(x)

    if not re.match(r"\d{13}", x):
        return False

    try:
        digits = [int(d) for d in x]
    except ValueError:
        return False

    check = digits[12]
    remainder = sum(d * w for d, w in zip(digits[:12], [1, 3] * 6)) % 10

    return (remainder == 0 and check == 0) or (10 - remainder == check)


def check_isbn(x: list[str]) -> list[bool]:
    """
    Check if strings are valid ISBNs (10 or 13 digit).

    Parameters
    ----------
    x
        List of strings to validate.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    results = []
    for val in x:
        if val is None or (isinstance(val, float) and val != val):  # Check for None or NaN
            results.append(False)
            continue

        val_clean = remove_hyphens(str(val))
        val_clean = remove_punctuation(val_clean)
        val_clean = val_clean.lower()
        val_clean = remove_spaces(val_clean)

        isbn_length = len(val_clean)

        if isbn_length == 10:
            results.append(is_isbn_10(val_clean))
        elif isbn_length == 13:
            results.append(is_isbn_13(val_clean))
        else:
            results.append(False)

    return results


def is_vin(x: str) -> bool:
    """
    Check if a string is a valid VIN (Vehicle Identification Number).

    Parameters
    ----------
    x
        String to validate.

    Returns
    -------
    bool
        True if valid VIN, False otherwise.
    """
    if not re.match(regex_vin(), x.upper()):
        return False

    x_lower = x.lower()
    digits = list(x_lower)

    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

    letter_vals = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
        "j": 1,
        "k": 2,
        "l": 3,
        "m": 4,
        "n": 5,
        "p": 7,
        "r": 9,
        "s": 2,
        "t": 3,
        "u": 4,
        "v": 5,
        "w": 6,
        "x": 7,
        "y": 8,
        "z": 9,
    }

    total = 0
    for i in range(17):
        if not digits[i].isdigit():
            total += letter_vals.get(digits[i], 0) * weights[i]
        else:
            total += int(digits[i]) * weights[i]

    check = total % 11

    if check == 10:
        check_str = "x"
    else:
        check_str = str(check)

    return check_str == digits[8]


def check_vin(x: list[str]) -> list[bool]:
    """
    Check if strings are valid VINs.

    Parameters
    ----------
    x
        List of strings to validate.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    results = []
    for val in x:
        if val is None or (isinstance(val, float) and val != val):  # Check for None or NaN
            results.append(False)
            continue

        val_str = str(val)
        val_str = remove_hyphens(val_str)
        val_str = remove_punctuation(val_str)
        val_str = val_str.lower()
        val_str = remove_spaces(val_str)

        results.append(is_vin(val_str))

    return results


def luhn(x: str) -> bool:
    """
    Check if a string passes the Luhn algorithm (for credit cards).

    Parameters
    ----------
    x
        String to validate.

    Returns
    -------
    bool
        True if passes Luhn check, False otherwise.
    """
    try:
        digits = [int(d) for d in reversed(x)]
    except ValueError:
        return False

    odd_sum = sum(digits[::2])
    even_digits = [d * 2 for d in digits[1::2]]
    even_digits = [d - 9 if d > 9 else d for d in even_digits]
    even_sum = sum(even_digits)

    total = odd_sum + even_sum

    return total % 10 == 0


def is_credit_card(x: str) -> bool:
    """
    Check if a string is a valid credit card number.

    Parameters
    ----------
    x
        String to validate.

    Returns
    -------
    bool
        True if valid credit card number, False otherwise.
    """
    if not re.match(regex_credit_card_1(), x):
        return False

    x_clean = remove_hyphens(x)
    x_clean = remove_punctuation(x_clean)
    x_clean = remove_spaces(x_clean)

    if not re.match(regex_credit_card_2(), x_clean):
        return False

    return luhn(x_clean)


def check_credit_card(x: list[str]) -> list[bool]:
    """
    Check if strings are valid credit card numbers.

    Parameters
    ----------
    x
        List of strings to validate.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    return [
        is_credit_card(str(val))
        if val is not None and (not isinstance(val, float) or val == val)
        else False
        for val in x
    ]


def check_iban(x: list[str], country: str | None = None) -> list[bool]:
    """
    Check if strings are valid IBANs.

    Parameters
    ----------
    x
        List of strings to validate.
    country
        Optional country code for country-specific validation.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    pattern = regex_iban(country=country)
    return [
        bool(re.match(pattern, str(val).upper()))
        if val is not None and (not isinstance(val, float) or val == val)
        else False
        for val in x
    ]


def check_postal_code(x: list[str], country: str) -> list[bool]:
    """
    Check if strings are valid postal codes for a given country.

    Parameters
    ----------
    x
        List of strings to validate.
    country
        Country code (2 or 3 letter).

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    pattern = regex_postal_code(country=country)
    return [
        bool(re.match(pattern, str(val).upper()))
        if val is not None and (not isinstance(val, float) or val == val)
        else False
        for val in x
    ]


def check_url(x: list[str]) -> list[bool]:
    """
    Check if strings are valid URLs.

    Parameters
    ----------
    x
        List of strings to validate.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    pattern = regex_url()
    return [
        bool(re.match(pattern, str(val)))
        if val is not None and (not isinstance(val, float) or val == val)
        else False
        for val in x
    ]


def check_ipv4_address(x: list[str]) -> list[bool]:
    """
    Check if strings are valid IPv4 addresses.

    Parameters
    ----------
    x
        List of strings to validate.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    pattern = regex_ipv4_address()
    return [
        bool(re.match(pattern, str(val)))
        if val is not None and (not isinstance(val, float) or val == val)
        else False
        for val in x
    ]


def check_ipv6_address(x: list[str]) -> list[bool]:
    """
    Check if strings are valid IPv6 addresses.

    Parameters
    ----------
    x
        List of strings to validate.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    pattern = regex_ipv6_address()
    return [
        bool(re.match(pattern, str(val)))
        if val is not None and (not isinstance(val, float) or val == val)
        else False
        for val in x
    ]


def check_email(x: list[str]) -> list[bool]:
    """
    Check if strings are valid email addresses.

    Parameters
    ----------
    x
        List of strings to validate.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    pattern = regex_email()
    return [
        bool(re.match(pattern, str(val)))
        if val is not None and (not isinstance(val, float) or val == val)
        else False
        for val in x
    ]


def check_phone(x: list[str]) -> list[bool]:
    """
    Check if strings are valid phone numbers.

    Parameters
    ----------
    x
        List of strings to validate.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    pattern = regex_phone()
    return [
        bool(re.match(pattern, str(val)))
        if val is not None and (not isinstance(val, float) or val == val)
        else False
        for val in x
    ]


def check_mac(x: list[str]) -> list[bool]:
    """
    Check if strings are valid MAC addresses.

    Parameters
    ----------
    x
        List of strings to validate.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    pattern = regex_mac()
    return [
        bool(re.match(pattern, str(val)))
        if val is not None and (not isinstance(val, float) or val == val)
        else False
        for val in x
    ]


def check_swift_bic(x: list[str]) -> list[bool]:
    """
    Check if strings are valid SWIFT/BIC codes.

    Parameters
    ----------
    x
        List of strings to validate.

    Returns
    -------
    list[bool]
        List of boolean values indicating validity.
    """
    pattern = regex_swift_bic()
    return [
        bool(re.match(pattern, str(val).upper()))
        if val is not None and (not isinstance(val, float) or val == val)
        else False
        for val in x
    ]
