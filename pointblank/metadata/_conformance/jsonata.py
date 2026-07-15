"""Minimal JSONata expression evaluator for CDISC conformance rules.

Implements the subset of JSONata used in CDISC Library rule catalogs:

- Field access: `STUDYID`, `Dataset.Variable`
- Literals: numbers, strings, `true`, `false`, `null`
- Comparison: `=`, `!=`, `<`, `>`, `<=`, `>=`
- Arithmetic: `+`, `-`, `*`, `/`
- Boolean: `and`, `or`, `not(expr)`
- String functions: `$uppercase`, `$lowercase`, `$string`, `$length`,
  `$substring`, `$trim`
- Aggregate functions: `$count`, `$exists`, `$distinct`
- Type check: `$type`

Complex JSONata features (filter expressions, transforms, regex, lambda) raise `JSONataNotSupported`
so the caller can choose an appropriate fallback.

Typical usage:

    from pointblank.metadata._conformance.jsonata import evaluate_jsonata

    # Evaluate against a row dict
    result = evaluate_jsonata("$uppercase(DOMAIN) = DOMAIN", {"DOMAIN": "AE"})
    # -> True

    # Evaluate against a dataset summary dict
    result = evaluate_jsonata("$count(USUBJID) > 0", {"USUBJID": ["S1", "S2"]})
    # -> True
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["evaluate_jsonata", "JSONataNotSupported", "JSONataSyntaxError"]


class JSONataNotSupported(Exception):
    """Raised for JSONata constructs outside this evaluator's supported subset."""


class JSONataSyntaxError(Exception):
    """Raised when the expression cannot be parsed."""


# ── Tokenizer ─────────────────────────────────────────────────────────────────

_TOK = re.compile(
    r"\s*("
    r"\$[a-zA-Z_][a-zA-Z0-9_]*"  # $function / $variable
    r"|[a-zA-Z_][a-zA-Z0-9_]*"  # identifier / keyword
    r'|"(?:[^"\\]|\\.)*"'  # double-quoted string
    r"|'(?:[^'\\]|\\.)*'"  # single-quoted string
    r"|[0-9]+(?:\.[0-9]+)?"  # number (int or float)
    r"|<="  # two-char ops first
    r"|>="
    r"|!="
    r"|[=<>+\-*/().,\[\]]"  # single-char ops and punctuation
    r")\s*"
)


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(expr):
        m = _TOK.match(expr, pos)
        if not m:
            break
        tok = m.group(1)
        if tok:
            tokens.append(tok)
        pos = m.end()
    return tokens


# ── Parser ────────────────────────────────────────────────────────────────────


class _Parser:
    """Recursive-descent parser for the JSONata subset.

    Precedence (low -> high):
        or
        and
        comparison  (=, !=, <, >, <=, >=)
        additive    (+, -)
        multiplicative (*, /)
        unary (not, -)
        postfix / primary
    """

    def __init__(self, tokens: list[str]) -> None:
        self._tok = tokens
        self._pos = 0

    # ── Token stream helpers ─────────────────────────────────────────────────

    def _peek(self) -> str | None:
        return self._tok[self._pos] if self._pos < len(self._tok) else None

    def _consume(self) -> str:
        t = self._tok[self._pos]
        self._pos += 1
        return t

    def _expect(self, value: str) -> str:
        t = self._peek()
        if t != value:
            raise JSONataSyntaxError(f"Expected {value!r}, got {t!r}")
        return self._consume()

    # ── Grammar levels ────────────────────────────────────────────────────────

    def parse(self) -> tuple:
        node = self._or()
        if self._pos < len(self._tok):
            raise JSONataSyntaxError(f"Unexpected token: {self._peek()!r}")
        return node

    def _or(self) -> tuple:
        left = self._and()
        while self._peek() == "or":
            self._consume()
            right = self._and()
            left = ("or", left, right)
        return left

    def _and(self) -> tuple:
        left = self._comparison()
        while self._peek() == "and":
            self._consume()
            right = self._comparison()
            left = ("and", left, right)
        return left

    def _comparison(self) -> tuple:
        left = self._additive()
        op = self._peek()
        if op in ("=", "!=", "<", ">", "<=", ">="):
            self._consume()
            right = self._additive()
            return ("cmp", op, left, right)
        return left

    def _additive(self) -> tuple:
        left = self._multiplicative()
        while self._peek() in ("+", "-"):
            op = self._consume()
            right = self._multiplicative()
            left = ("binop", op, left, right)
        return left

    def _multiplicative(self) -> tuple:
        left = self._unary()
        while self._peek() in ("*", "/"):
            op = self._consume()
            right = self._unary()
            left = ("binop", op, left, right)
        return left

    def _unary(self) -> tuple:
        t = self._peek()
        if t == "-":
            self._consume()
            return ("neg", self._unary())
        if t and t.lower() == "not":
            self._consume()
            self._expect("(")
            inner = self._or()
            self._expect(")")
            return ("not", inner)
        return self._primary()

    def _primary(self) -> tuple:
        t = self._peek()
        if t is None:
            raise JSONataSyntaxError("Unexpected end of expression")

        # Grouped expression
        if t == "(":
            self._consume()
            inner = self._or()
            self._expect(")")
            return inner

        # String literal
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            self._consume()
            return ("lit", t[1:-1].replace('\\"', '"').replace("\\'", "'"))

        # Number literal
        if re.match(r"^[0-9]", t):
            self._consume()
            return ("lit", float(t) if "." in t else int(t))

        # Boolean / null literals
        if t.lower() == "true":
            self._consume()
            return ("lit", True)
        if t.lower() == "false":
            self._consume()
            return ("lit", False)
        if t.lower() == "null":
            self._consume()
            return ("lit", None)

        # $function call or $variable
        if t.startswith("$"):
            self._consume()
            name = t[1:]  # strip leading $
            if self._peek() == "(":
                return self._call(name)
            return ("var", t)  # bare $variable reference

        # Filter expression — not supported
        if t == "[":
            raise JSONataNotSupported("Filter expressions [...]  are not supported")

        # Plain identifier — may be a path (a.b.c)
        if re.match(r"^[a-zA-Z_]", t):
            self._consume()
            path = [t]
            while self._peek() == ".":
                self._consume()
                nxt = self._peek()
                if nxt and re.match(r"^[a-zA-Z_$]", nxt):
                    path.append(self._consume())
                else:
                    break
            return ("path", path)

        raise JSONataSyntaxError(f"Unexpected token: {t!r}")

    def _call(self, name: str) -> tuple:
        self._expect("(")
        args: list[tuple] = []
        while self._peek() != ")":
            args.append(self._or())
            if self._peek() == ",":
                self._consume()
        self._expect(")")
        return ("call", name, args)


# ── Evaluator ─────────────────────────────────────────────────────────────────


def _eval(node: tuple, ctx: dict[str, Any]) -> Any:
    kind = node[0]

    if kind == "lit":
        return node[1]

    if kind == "path":
        parts: list[str] = node[1]
        value = ctx
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        return value

    if kind == "var":
        return ctx.get(node[1])

    if kind == "neg":
        v = _eval(node[1], ctx)
        return -v if isinstance(v, (int, float)) else None

    if kind == "not":
        return not _eval(node[1], ctx)

    if kind == "or":
        return bool(_eval(node[1], ctx)) or bool(_eval(node[2], ctx))

    if kind == "and":
        return bool(_eval(node[1], ctx)) and bool(_eval(node[2], ctx))

    if kind == "cmp":
        _, op, lhs, rhs = node
        left = _eval(lhs, ctx)
        right = _eval(rhs, ctx)
        # JSONata uses = and != (not == and !==)
        if op == "=":
            return left == right
        if op == "!=":
            return left != right
        # For ordering operators, None/null is always less than any value
        if left is None or right is None:
            return False
        if op == "<":
            return left < right
        if op == ">":
            return left > right
        if op == "<=":
            return left <= right
        if op == ">=":
            return left >= right
        return False

    if kind == "binop":
        _, op, lhs, rhs = node
        left = _eval(lhs, ctx)
        right = _eval(rhs, ctx)
        if left is None or right is None:
            return None
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right if right != 0 else None
        return None

    if kind == "call":
        _, name, args = node
        evaled = [_eval(a, ctx) for a in args]
        return _call_function(name, evaled)

    raise JSONataSyntaxError(f"Unknown AST node: {kind!r}")


def _call_function(name: str, args: list[Any]) -> Any:
    arg0 = args[0] if args else None

    # ── String functions ─────────────────────────────────────────────────────
    if name == "uppercase":
        return str(arg0).upper() if arg0 is not None else None
    if name == "lowercase":
        return str(arg0).lower() if arg0 is not None else None
    if name == "string":
        return str(arg0) if arg0 is not None else ""
    if name == "length":
        if arg0 is None:
            return 0
        if isinstance(arg0, (list, tuple)):
            return len(arg0)
        return len(str(arg0))
    if name == "trim":
        return str(arg0).strip() if arg0 is not None else None
    if name == "substring":
        # $substring(str, start, length?)
        s = str(arg0) if arg0 is not None else ""
        start = int(args[1]) if len(args) > 1 and args[1] is not None else 0
        if start < 0:
            start = max(0, len(s) + start)
        if len(args) > 2 and args[2] is not None:
            length = int(args[2])
            return s[start : start + length]
        return s[start:]

    # ── Aggregate functions ──────────────────────────────────────────────────
    if name == "count":
        if arg0 is None:
            return 0
        if isinstance(arg0, (list, tuple)):
            return len(arg0)
        return 1  # scalar → count of 1

    if name == "exists":
        return arg0 is not None

    if name == "distinct":
        if arg0 is None:
            return []
        if isinstance(arg0, (list, tuple)):
            seen: list = []
            for item in arg0:
                if item not in seen:
                    seen.append(item)
            return seen
        return [arg0]

    if name == "sum":
        if isinstance(arg0, (list, tuple)):
            return sum(v for v in arg0 if isinstance(v, (int, float)))
        return arg0 if isinstance(arg0, (int, float)) else 0

    if name == "max":
        if isinstance(arg0, (list, tuple)):
            nums = [v for v in arg0 if isinstance(v, (int, float))]
            return max(nums) if nums else None
        return arg0

    if name == "min":
        if isinstance(arg0, (list, tuple)):
            nums = [v for v in arg0 if isinstance(v, (int, float))]
            return min(nums) if nums else None
        return arg0

    if name == "round":
        if arg0 is None:
            return None
        precision = int(args[1]) if len(args) > 1 and args[1] is not None else 0
        return round(float(arg0), precision)

    if name == "floor":
        return int(arg0) if arg0 is not None else None

    if name == "ceil":
        import math

        return math.ceil(arg0) if arg0 is not None else None

    if name == "abs":
        return abs(arg0) if arg0 is not None else None

    if name == "type":
        if arg0 is None:
            return "null"
        if isinstance(arg0, bool):
            return "boolean"
        if isinstance(arg0, (int, float)):
            return "number"
        if isinstance(arg0, str):
            return "string"
        if isinstance(arg0, (list, tuple)):
            return "array"
        if isinstance(arg0, dict):
            return "object"
        return "unknown"

    if name == "not":
        return not arg0

    raise JSONataNotSupported(
        f"JSONata function ${name}() is not supported by the native evaluator"
    )


# ── Public API ────────────────────────────────────────────────────────────────


def evaluate_jsonata(expr: str, context: dict[str, Any]) -> Any:
    """Evaluate a JSONata expression against *context*.

    Parameters
    ----------
    expr
        A JSONata expression string (e.g. `"$uppercase(DOMAIN) = DOMAIN"`).
    context
        A dict mapping variable names to their values (typically a row dict or a dataset-level
        summary dict).

    Returns
    -------
    Any
        The result of evaluating the expression. For conformance conditions this is typically a
        boolean.

    Raises
    ------
    JSONataNotSupported
        If the expression uses a JSONata feature outside the supported subset.
    JSONataSyntaxError
        If the expression cannot be parsed.
    """
    tokens = _tokenize(expr.strip())
    if not tokens:
        return None
    parser = _Parser(tokens)
    ast = parser.parse()
    return _eval(ast, context)
