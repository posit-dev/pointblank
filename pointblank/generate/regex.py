from __future__ import annotations

import random
import string
from re import U

try:
    import re._parser as sre_parse  # Python 3.11+
except ImportError:
    from re import sre_parse  # type: ignore[attr-defined]


# Character categories for regex character classes
def _build_categories() -> dict:
    """Build character category mappings for regex classes like \\d, \\w, \\s."""
    # Use ASCII-only characters for predictable, readable output
    word_chars = list(string.ascii_letters + string.digits + "_")
    non_word_chars = [chr(x) for x in range(32, 127) if chr(x) not in word_chars]
    digit_chars = list(string.digits)
    non_digit_chars = [chr(x) for x in range(32, 127) if chr(x) not in digit_chars]
    space_chars = list(" \t\n\r\f\v")
    non_space_chars = [chr(x) for x in range(32, 127) if chr(x) not in space_chars]

    return {
        sre_parse.CATEGORY_SPACE: space_chars,
        sre_parse.CATEGORY_NOT_SPACE: non_space_chars,
        sre_parse.CATEGORY_DIGIT: digit_chars,
        sre_parse.CATEGORY_NOT_DIGIT: non_digit_chars,
        sre_parse.CATEGORY_WORD: word_chars,
        sre_parse.CATEGORY_NOT_WORD: non_word_chars,
        "category_any": [chr(x) for x in range(32, 127)],  # Printable ASCII
    }


CATEGORIES = _build_categories()


def _parse_in(items: list, rng: random.Random) -> str:
    """Handle character class [...] by returning a random matching character."""
    chars: list[str] = []
    negate = False

    for item in items:
        op = item[0]

        if op == sre_parse.NEGATE:
            chars = list(CATEGORIES["category_any"])
            negate = True

        elif op == sre_parse.RANGE:
            # Character range like a-z
            range_chars = [chr(x) for x in range(item[1][0], item[1][1] + 1)]
            if negate:
                for char in range_chars:
                    if char in chars:
                        chars.remove(char)
            else:
                chars.extend(range_chars)

        elif op == sre_parse.LITERAL:
            char = chr(item[1])
            if negate:
                if char in chars:
                    chars.remove(char)
            else:
                chars.append(char)

        elif op == sre_parse.CATEGORY:
            category_chars = CATEGORIES.get(item[1], [""])
            if negate:
                for char in category_chars:
                    if char in chars:
                        chars.remove(char)
            else:
                chars.extend(category_chars)

    return rng.choice(chars) if chars else ""


def _generate_one(
    parsed: list, rng: random.Random, limit: int = 20, grouprefs: dict | None = None
) -> str:
    """Generate a single random string from a parsed regex pattern.

    Parameters
    ----------
    parsed
        The parsed regex structure from sre_parse.parse().
    rng
        Random number generator for reproducibility.
    limit
        Maximum number of repetitions for unbounded quantifiers like * or +.
    grouprefs
        Dictionary to store captured group values for backreferences.

    Returns
    -------
    str
        A random string matching the pattern.
    """
    if grouprefs is None:
        grouprefs = {}

    result = ""

    for item in parsed:
        op = item[0]

        if op == sre_parse.IN:
            # Character class [...]
            result += _parse_in(item[1], rng)

        elif op == sre_parse.LITERAL:
            # Literal character
            result += chr(item[1])

        elif op == sre_parse.CATEGORY:
            # Character category like \d, \w, \s
            chars = CATEGORIES.get(item[1], [""])
            result += rng.choice(chars)

        elif op == sre_parse.ANY:
            # . (any character)
            result += rng.choice(CATEGORIES["category_any"])

        elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
            # Quantifiers: *, +, ?, {n}, {n,m}
            min_count, max_count = item[1][0], item[1][1]
            sub_pattern = list(item[1][2])

            # Limit unbounded repetitions
            if max_count - min_count >= limit:
                max_count = min_count + limit - 1

            count = rng.randint(min_count, max_count)
            for _ in range(count):
                result += _generate_one(sub_pattern, rng, limit, grouprefs)

        elif op == sre_parse.BRANCH:
            # Alternation: a|b|c
            branches = item[1][1]
            chosen_branch = rng.choice(branches)
            result += _generate_one(chosen_branch, rng, limit, grouprefs)

        elif op == sre_parse.SUBPATTERN:
            # Capturing group (...)
            group_id = item[1][0]
            # Python 3.6+ has different structure
            sub_pattern = item[1][3] if len(item[1]) > 3 else item[1][1]
            sub_result = _generate_one(sub_pattern, rng, limit, grouprefs)
            if group_id:
                grouprefs[group_id] = sub_result
            result += sub_result

        elif op == sre_parse.ASSERT:
            # Lookahead assertion (?=...) - generate the content
            sub_pattern = item[1][1]
            result += _generate_one(sub_pattern, rng, limit, grouprefs)

        elif op == sre_parse.AT:
            # Anchors ^ and $ - ignore them
            continue

        elif op == sre_parse.NOT_LITERAL:
            # [^x] - any character except x
            chars = list(CATEGORIES["category_any"])
            excluded = chr(item[1])
            if excluded in chars:
                chars.remove(excluded)
            result += rng.choice(chars)

        elif op == sre_parse.GROUPREF:
            # Backreference \1, \2, etc.
            result += grouprefs.get(item[1], "")

        elif op == sre_parse.ASSERT_NOT:
            # Negative lookahead (?!...) - skip
            pass

        else:
            # Unknown operation - skip with warning (could also raise)
            pass

    return result


def generate_from_regex(pattern: str, rng: random.Random, limit: int = 20) -> str:
    """Generate a random string matching the given regular expression pattern.

    Parameters
    ----------
    pattern
        A regular expression pattern string.
    rng
        Random number generator instance for reproducibility.
    limit
        Maximum number of repetitions for unbounded quantifiers (*, +).
        Default is 20.

    Returns
    -------
    str
        A random string that matches the pattern.

    Examples
    --------
    >>> import random
    >>> rng = random.Random(23)
    >>> generate_from_regex(r"[A-Z]{3}-\\d{4}", rng)
    'CAS-6685'
    >>> generate_from_regex(r"(foo|bar|baz)", rng)
    'foo'
    >>> generate_from_regex(r"\\w+@\\w+\\.com", rng)
    'rCaoND5@g.com'
    """
    parsed = list(sre_parse.parse(pattern, flags=U))
    return _generate_one(parsed, rng, limit)
