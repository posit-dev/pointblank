"""
The text engine of the text checks: normalization, word n-grams, MinHash with locality-sensitive
hashing, near-duplicate clusters, n-gram matching against reference texts, the checks of chat
conversations, and the statistics of the heuristic quality filters.

Everything here is plain Python (NumPy, when it's installed, only makes MinHash faster: the
signatures are the same with and without it), and deterministic: the MinHash permutations come
from a fixed seed, so the same texts always give the same clusters.
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------------------------
# Normalization and n-grams
# ---------------------------------------------------------------------------------------------

_NOT_WORD = re.compile(r"[^\w\s]+")


def normalize(text: str) -> str:
    """
    Normalize a text for comparing: Unicode NFKC, case folded, punctuation and symbols replaced by
    spaces, and runs of whitespace collapsed to one space.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_NOT_WORD.sub(" ", text).split())


def words(text: str) -> list[str]:
    """The words of a normalized text."""
    return normalize(text).split()


def ngrams(tokens: list[str], n: int) -> list[str]:
    """
    The word n-grams of a text, each as a string. A text with fewer than `n` words gives one
    n-gram, of all of its words (an empty text gives the empty string).
    """
    if len(tokens) < n:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def hash32(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(), "little")


def hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def join_values(values: Sequence[Any]) -> str:
    """The text of a row for several columns: the values joined by newlines."""
    return "\n".join(v if isinstance(v, str) else str(v) for v in values)


# ---------------------------------------------------------------------------------------------
# MinHash and LSH
# ---------------------------------------------------------------------------------------------

_MERSENNE = (1 << 61) - 1
_MAX32 = (1 << 32) - 1
_SEED = 20_240_601


@functools.lru_cache(maxsize=8)
def _permutations(num_perm: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # `a` and `b` are below 2^32 (as are the hashes), so `a * x + b` fits in 64 bits and NumPy
    # gives the same signatures as Python's integers
    rng = random.Random(_SEED)
    a = tuple(rng.randint(1, _MAX32) for _ in range(num_perm))
    b = tuple(rng.randint(0, _MAX32) for _ in range(num_perm))
    return a, b


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    return np


def signatures(shingle_sets: list[set[int]], num_perm: int) -> list[tuple[int, ...]]:
    """The MinHash signatures of sets of 32-bit shingle hashes (none of them empty)."""
    a, b = _permutations(num_perm)
    np = _numpy()
    if np is not None:
        a_arr = np.array(a, dtype=np.uint64)[:, None]
        b_arr = np.array(b, dtype=np.uint64)[:, None]
        p = np.uint64(_MERSENNE)
        mask = np.uint64(_MAX32)
        out = []
        for shingles in shingle_sets:
            hv = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))[None, :]
            values = ((a_arr * hv + b_arr) % p) & mask
            out.append(tuple(int(v) for v in values.min(axis=1)))
        return out
    params = list(zip(a, b))  # pragma: no cover
    return [  # pragma: no cover
        tuple(min(((ai * x + bi) % _MERSENNE) & _MAX32 for x in shingles) for ai, bi in params)
        for shingles in shingle_sets
    ]


@functools.lru_cache(maxsize=64)
def lsh_params(threshold: float, num_perm: int) -> tuple[int, int]:
    """
    Choose the number of bands and of rows per band for LSH, so that pairs with a Jaccard
    similarity at the threshold are as likely to be missed as pairs below it are to be compared
    (the false positives and false negatives are weighted equally, as in `datasketch`).
    """

    def integrate(f: Any, low: float, high: float, steps: int = 100) -> float:
        if high <= low:
            return 0.0
        width = (high - low) / steps
        total = 0.5 * (f(low) + f(high))
        total += sum(f(low + k * width) for k in range(1, steps))
        return total * width

    best = (1, num_perm)
    best_error = float("inf")
    for bands in range(1, num_perm + 1):
        rows = num_perm // bands
        if rows < 1:
            break

        def p_candidate(s: float, r: int = rows, b: int = bands) -> float:
            return 1.0 - (1.0 - s**r) ** b

        fp = integrate(p_candidate, 0.0, threshold)
        fn = integrate(lambda s, f=p_candidate: 1.0 - f(s), threshold, 1.0)
        error = fp + fn
        if error < best_error - 1e-12:
            best, best_error = (bands, rows), error
    return best


def jaccard(x: set[int], y: set[int]) -> float:
    if not x and not y:
        return 1.0
    inter = len(x & y)
    return inter / (len(x) + len(y) - inter)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        parent = self.parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            # The smaller index is the root, so clusters are numbered by their first member
            if ri < rj:
                self.parent[rj] = ri
            else:
                self.parent[ri] = rj


def shingle_hashes(text: str, ngram: int) -> set[int]:
    return {hash32(g) for g in ngrams(words(text), ngram)}


def near_duplicate_groups(
    texts: list[str],
    method: str,
    threshold: float,
    ngram: int,
    num_perm: int,
) -> tuple[list[int], int]:
    """
    Group texts that are near duplicates of each other.

    `method="exact"` groups texts that are the same once normalized. `method="minhash"` also
    groups texts whose sets of word n-grams have a Jaccard similarity of at least `threshold`:
    candidate pairs come from MinHash with LSH, and each candidate pair is confirmed with the
    exact Jaccard similarity of the n-grams, so no pair below the threshold is grouped (a pair
    just above it can be missed, rarely). Groups are connected: if A is like B and B is like C,
    then A, B, and C are one group.

    Returns the group of each text (numbered from 0, in the order of their first text) and the
    number of candidate pairs that were compared.
    """
    n = len(texts)
    by_normalized: dict[str, int] = {}
    distinct_of = []
    for text in texts:
        distinct_of.append(by_normalized.setdefault(normalize(text), len(by_normalized)))
    n_distinct = len(by_normalized)

    n_compared = 0
    uf = _UnionFind(n_distinct)
    if method == "minhash" and n_distinct > 1:
        normalized = list(by_normalized)
        shingles = [{hash32(g) for g in ngrams(text.split(), ngram)} for text in normalized]
        sigs = signatures(shingles, num_perm)
        bands, rows = lsh_params(threshold, num_perm)
        for band in range(bands):
            buckets: dict[tuple[int, ...], list[int]] = {}
            start = band * rows
            for k, sig in enumerate(sigs):
                buckets.setdefault(sig[start : start + rows], []).append(k)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                # Compare each member with one member of each group found in the bucket so far
                # (members of a bucket are usually alike, so there are few groups to compare with)
                reps: list[int] = []
                for k in members:
                    joined = False
                    for rep in reps:
                        if uf.find(rep) == uf.find(k):
                            joined = True
                            break
                        n_compared += 1
                        if jaccard(shingles[k], shingles[rep]) >= threshold:
                            uf.union(k, rep)
                            joined = True
                            break
                    if not joined:
                        reps.append(k)

    # Number the groups in the order of their first text
    roots = [uf.find(k) for k in range(n_distinct)]
    numbers: dict[int, int] = {}
    groups = []
    for i in range(n):
        root = roots[distinct_of[i]]
        if root not in numbers:
            numbers[root] = len(numbers)
        groups.append(numbers[root])
    return groups, n_compared


# ---------------------------------------------------------------------------------------------
# Matching against reference texts (contamination)
# ---------------------------------------------------------------------------------------------


class ReferenceIndex:
    """
    An index of reference texts (e.g., the questions of a benchmark), to find the texts of a table
    that overlap with them.

    - `method="ngram"`: a text overlaps if it shares a word n-gram with a reference text, if a
      reference text shorter than `ngram` words appears whole in it, or if it's shorter than
      `ngram` words itself and is the same as a reference text (all compared once normalized)
    - `method="exact"`: a text overlaps if it's the same as a reference text, once normalized
    - `method="minhash"`: a text overlaps if the Jaccard similarity of its word n-grams with those
      of a reference text is at least `threshold` (found with MinHash and LSH, and confirmed)

    `match()` gives the position of the first reference text that a text overlaps with, or
    `None`.
    """

    def __init__(
        self,
        texts: list[str],
        method: str,
        ngram: int,
        threshold: float = 0.8,
        num_perm: int = 128,
    ):
        self.method = method
        self.ngram = ngram
        self.threshold = threshold
        self.num_perm = num_perm
        self.exact: dict[str, int] = {}
        self.grams: dict[int, int] = {}
        self.short: dict[int, dict[int, int]] = {}
        normalized = [normalize(t) for t in texts]
        for k, text in enumerate(normalized):
            self.exact.setdefault(text, k)
        if method == "ngram":
            for k, text in enumerate(normalized):
                tokens = text.split()
                if not tokens:
                    continue
                if len(tokens) < ngram:
                    self.short.setdefault(len(tokens), {}).setdefault(hash64(text), k)
                    continue
                for g in ngrams(tokens, ngram):
                    self.grams.setdefault(hash64(g), k)
        elif method == "minhash":
            self.shingles = [{hash32(g) for g in ngrams(t.split(), ngram)} for t in normalized]
            self.sigs = signatures(self.shingles, num_perm)
            self.bands, self.rows = lsh_params(threshold, num_perm)
            self.buckets: list[dict[tuple[int, ...], list[int]]] = []
            for band in range(self.bands):
                start = band * self.rows
                buckets: dict[tuple[int, ...], list[int]] = {}
                for k, sig in enumerate(self.sigs):
                    buckets.setdefault(sig[start : start + self.rows], []).append(k)
                self.buckets.append(buckets)

    def match(self, text: str) -> int | None:
        normalized = normalize(text)
        if self.method == "exact":
            return self.exact.get(normalized)
        tokens = normalized.split()
        if self.method == "ngram":
            if len(tokens) < self.ngram:
                if not tokens:
                    return None
                found = self.exact.get(normalized)
                if found is not None:
                    return found
            else:
                for g in ngrams(tokens, self.ngram):
                    found = self.grams.get(hash64(g))
                    if found is not None:
                        return found
            # A short reference text that appears whole in the text
            for size, index in self.short.items():
                if size > len(tokens):
                    continue
                for g in ngrams(tokens, size):
                    found = index.get(hash64(g))
                    if found is not None:
                        return found
            return None
        # MinHash
        shingles = {hash32(g) for g in ngrams(tokens, self.ngram)}
        sig = signatures([shingles], self.num_perm)[0]
        seen: set[int] = set()
        best = None
        for band, buckets in enumerate(self.buckets):
            start = band * self.rows
            for k in buckets.get(sig[start : start + self.rows], ()):
                if k in seen:
                    continue
                seen.add(k)
                if jaccard(shingles, self.shingles[k]) >= self.threshold and (
                    best is None or k < best
                ):
                    best = k
        return best


# ---------------------------------------------------------------------------------------------
# Chat conversations
# ---------------------------------------------------------------------------------------------

# The reasons that a conversation can fail `col_vals_chat_format()`, by code
CHAT_REASONS = {
    "not_messages": "isn't a list of messages",
    "no_messages": "has no messages",
    "bad_message": "has a message without a role",
    "unknown_role": "has a role that isn't allowed",
    "system_position": "has a system message after the first message",
    "first_role": "starts with the wrong role",
    "not_alternating": "has roles that don't alternate",
    "last_role": "ends with the wrong role",
    "empty_content": "has a message with empty content",
}


def _as_messages(value: Any) -> Any:
    """Turn a value of a chat column into a list of messages (or `None` if it isn't one)."""
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if isinstance(value, Mapping) and "messages" in value:
        value = value["messages"]
    if not isinstance(value, (list, tuple)):
        return None
    return list(value)


def chat_reason(
    value: Any,
    roles: Sequence[str],
    system_role: str | None,
    alternate: bool,
    first_role: str | None,
    last_role: str | None,
    allow_empty: bool,
    role_key: str,
    content_key: str,
) -> str | None:
    """
    Check a conversation, giving the code of the first problem found (or `None` if it's well
    formed). The checks are made in the order of `CHAT_REASONS`.
    """
    messages = _as_messages(value)
    if messages is None:
        return "not_messages"
    if not messages:
        return "no_messages"
    found_roles = []
    contents = []
    for message in messages:
        if hasattr(message, "items") and not isinstance(message, Mapping):  # pragma: no cover
            message = dict(message.items())
        if not isinstance(message, Mapping) or role_key not in message:
            return "bad_message"
        role = message[role_key]
        if not isinstance(role, str):
            return "bad_message"
        found_roles.append(role)
        contents.append(message.get(content_key))
    if any(role not in roles for role in found_roles):
        return "unknown_role"
    if system_role is not None and any(role == system_role for role in found_roles[1:]):
        return "system_position"
    turns = (
        found_roles[1:]
        if system_role is not None and found_roles[0] == system_role
        else found_roles
    )
    if first_role is not None and (not turns or turns[0] != first_role):
        return "first_role"
    if alternate and any(r1 == r2 for r1, r2 in zip(turns, turns[1:])):
        return "not_alternating"
    if last_role is not None and found_roles[-1] != last_role:
        return "last_role"
    if not allow_empty:
        for content in contents:
            if content is None or (isinstance(content, str) and not content.strip()):
                return "empty_content"
            if isinstance(content, (list, tuple)) and not content:
                return "empty_content"
    return None


# ---------------------------------------------------------------------------------------------
# Text statistics (for the heuristic quality filters)
# ---------------------------------------------------------------------------------------------

# Words that nearly every English text of some length has (from the Gopher quality filters)
STOP_WORDS = ("the", "be", "to", "of", "and", "that", "have", "with")

_BULLETS = ("-", "*", "•", "‣", "◦", "⁃", "–", "—", "·")
_ELLIPSES = ("...", "…")
_SYMBOLS = ("#", "...", "…")
_HAS_ALPHA = re.compile(r"[^\W\d_]")

STATS = (
    "char_count",
    "word_count",
    "mean_word_length",
    "symbol_ratio",
    "bullet_lines",
    "ellipsis_lines",
    "alpha_words",
    "stop_words",
    "repeated_lines",
)


def text_stat(text: str, stat: str, stop_words: Sequence[str] = STOP_WORDS) -> float | None:
    """
    A statistic of a text, as used by the Gopher quality filters (Rae et al., 2021). Words are
    the text split on whitespace, and lines are the text split on newlines (ignoring blank lines).

    - `char_count`: the number of characters
    - `word_count`: the number of words
    - `mean_word_length`: the mean number of characters of the words (`None` without words)
    - `symbol_ratio`: the number of hash symbols and ellipses for each word
    - `bullet_lines`: the share of lines that start with a bullet point
    - `ellipsis_lines`: the share of lines that end with an ellipsis
    - `alpha_words`: the share of words with at least one letter
    - `stop_words`: the number of distinct stop words in the text (in lower case)
    - `repeated_lines`: the share of lines that are the same as an earlier line
    """
    if stat == "char_count":
        return float(len(text))
    tokens = text.split()
    if stat == "word_count":
        return float(len(tokens))
    if stat == "mean_word_length":
        return sum(len(t) for t in tokens) / len(tokens) if tokens else None
    if stat == "symbol_ratio":
        if not tokens:
            return None
        n_symbols = text.count("#") + text.count("...") + text.count("…")
        return n_symbols / len(tokens)
    if stat == "alpha_words":
        if not tokens:
            return None
        return sum(1 for t in tokens if _HAS_ALPHA.search(t)) / len(tokens)
    if stat == "stop_words":
        present = {t.casefold().strip(".,;:!?\"'()[]") for t in tokens}
        return float(sum(1 for w in dict.fromkeys(stop_words) if w.casefold() in present))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if stat == "bullet_lines":
        return sum(1 for line in lines if line.startswith(_BULLETS)) / len(lines)
    if stat == "ellipsis_lines":
        return sum(1 for line in lines if line.endswith(_ELLIPSES)) / len(lines)
    if stat == "repeated_lines":
        seen: set[str] = set()
        repeated = 0
        for line in lines:
            if line in seen:
                repeated += 1
            seen.add(line)
        return repeated / len(lines)
    raise ValueError(f"Unknown text statistic {stat!r}.")  # pragma: no cover


# ---------------------------------------------------------------------------------------------
# Personal information
# ---------------------------------------------------------------------------------------------

# Patterns of personal information in text, with only the regular expression features that
# every backend has (no lookarounds or backreferences), so they work on Polars, pandas, DuckDB,
# SQLite, and PySpark alike. They find likely matches: there's no checksum (e.g., of a card
# number), so a number of the right shape matches.
PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
    "phone": (
        r"(?:^|[^0-9])(?:\+[0-9]{1,3}[ .-]?)?(?:\([0-9]{3}\)|[0-9]{3})[ .-]?[0-9]{3}[ .-]"
        r"[0-9]{4}(?:[^0-9]|$)"
    ),
    "credit_card": (
        r"(?:^|[^0-9])(?:(?:4[0-9]{3}|5[1-5][0-9]{2}|2[2-7][0-9]{2}|6011|65[0-9]{2})"
        r"(?:[ -]?[0-9]{4}){3}|3[47][0-9]{2}[ -]?[0-9]{6}[ -]?[0-9]{5})(?:[^0-9]|$)"
    ),
    "ip_address": (
        r"(?:^|[^0-9.])(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(?:[^0-9.]|$)"
    ),
    "us_ssn": r"(?:^|[^0-9])[0-9]{3}-[0-9]{2}-[0-9]{4}(?:[^0-9]|$)",
    "iban": (
        r"(?:^|[^A-Za-z0-9])[A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?"
        r"(?:[^A-Za-z0-9]|$)"
    ),
}

PII_DESCRIPTIONS = {
    "email": "email addresses",
    "phone": "phone numbers",
    "credit_card": "credit card numbers",
    "ip_address": "IP addresses",
    "us_ssn": "US Social Security numbers",
    "iban": "IBANs",
}
