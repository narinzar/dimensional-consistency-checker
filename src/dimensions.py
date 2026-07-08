"""Physical dimensions as vectors of the 7 SI base-dimension exponents.

The base dimensions are, in fixed order:

    mass (M), length (L), time (T), current (I),
    temperature (K), amount (N), luminous intensity (J)

A Dimension is an immutable 7-tuple of rational exponents. For example
force has dimension M^1 L^1 T^-2, i.e. (1, 1, -2, 0, 0, 0, 0).

This module has no dependency on the LLM. It provides the algebra used to
combine symbol dimensions into term dimensions and to compare terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import sympy as sp

# Fixed ordering of the seven SI base dimensions.
BASE_DIMENSIONS: Tuple[str, ...] = (
    "mass",
    "length",
    "time",
    "current",
    "temperature",
    "amount",
    "luminous",
)

# Short symbols, aligned with BASE_DIMENSIONS, used for pretty printing.
BASE_SYMBOLS: Tuple[str, ...] = ("M", "L", "T", "I", "K", "N", "J")

_INDEX = {name: i for i, name in enumerate(BASE_DIMENSIONS)}


def _to_fraction(value) -> Fraction:
    """Coerce ints, floats, strings, and Fractions to an exact Fraction."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        # limit_denominator keeps exponents like 0.5 or 1.5 exact.
        return Fraction(value).limit_denominator(10**6)
    # Handles strings such as "1", "-2", "1/2".
    return Fraction(str(value))


@dataclass(frozen=True)
class Dimension:
    """An immutable vector of the 7 SI base-dimension exponents."""

    exponents: Tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if len(self.exponents) != len(BASE_DIMENSIONS):
            raise ValueError(
                f"expected {len(BASE_DIMENSIONS)} exponents, "
                f"got {len(self.exponents)}"
            )

    # -- constructors -----------------------------------------------------
    @classmethod
    def dimensionless(cls) -> "Dimension":
        return cls(tuple(Fraction(0) for _ in BASE_DIMENSIONS))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "Dimension":
        """Build a Dimension from a dict of base-dimension -> exponent.

        Unknown keys raise a ValueError so a bad LLM response fails loudly
        rather than silently dropping a dimension.
        """
        exps = [Fraction(0) for _ in BASE_DIMENSIONS]
        for key, value in mapping.items():
            if key not in _INDEX:
                raise ValueError(f"unknown base dimension {key!r}")
            exps[_INDEX[key]] = _to_fraction(value)
        return cls(tuple(exps))

    # -- algebra ----------------------------------------------------------
    def multiply(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def divide(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def power(self, exponent) -> "Dimension":
        p = _to_fraction(exponent)
        return Dimension(tuple(a * p for a in self.exponents))

    # operator sugar
    def __mul__(self, other: "Dimension") -> "Dimension":
        return self.multiply(other)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return self.divide(other)

    def __pow__(self, exponent) -> "Dimension":
        return self.power(exponent)

    # -- comparison -------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self.exponents == other.exponents

    def __hash__(self) -> int:
        return hash(self.exponents)

    def is_dimensionless(self) -> bool:
        return all(e == 0 for e in self.exponents)

    # -- presentation -----------------------------------------------------
    def as_mapping(self) -> Dict[str, Fraction]:
        """Return only the non-zero base exponents as a dict."""
        return {
            BASE_DIMENSIONS[i]: e
            for i, e in enumerate(self.exponents)
            if e != 0
        }

    def __str__(self) -> str:
        parts = []
        for sym, e in zip(BASE_SYMBOLS, self.exponents):
            if e == 0:
                continue
            if e == 1:
                parts.append(sym)
            else:
                parts.append(f"{sym}^{_fmt_exp(e)}")
        return " ".join(parts) if parts else "dimensionless"


def _fmt_exp(e: Fraction) -> str:
    if e.denominator == 1:
        return str(e.numerator)
    return f"{e.numerator}/{e.denominator}"


# ---------------------------------------------------------------------------
# Computing term dimensions from symbol dimensions
# ---------------------------------------------------------------------------

def term_dimension(term: str, symbol_dims: Mapping[str, Dimension]) -> Dimension:
    """Compute the physical dimension of a single multiplicative term.

    ``term`` is a plain algebraic expression such as ``"m*a"``,
    ``"m*c**2"`` or ``"1/2*m*v**2"``. It is parsed with sympy and each free
    symbol is substituted by its dimension. Pure numeric factors (like the
    1/2) are dimensionless and drop out.

    Raises ValueError if a symbol used in the term has no known dimension, or
    if the term cannot be reduced to a single monomial in the base
    dimensions (for example if it contains an additive sub-expression).
    """
    # Bind every known symbol name to a plain sympy Symbol so that names which
    # collide with sympy built-ins (E for Euler's number, I for the imaginary
    # unit, etc.) are treated as physical symbols rather than constants.
    local_syms = {name: sp.Symbol(name) for name in symbol_dims}
    expr = sp.sympify(term, locals=local_syms, evaluate=True)

    # Expand so that, e.g., (a*b)**2 becomes a**2 * b**2 for extraction.
    expr = sp.expand_power_base(sp.powsimp(expr, force=True), force=True)

    missing = sorted(str(s) for s in expr.free_symbols if str(s) not in symbol_dims)
    if missing:
        raise ValueError(
            f"term {term!r} uses symbol(s) without a dimension: {missing}"
        )

    result = Dimension.dimensionless()

    # A well-formed term is a product of powers of symbols and numbers.
    for factor in _iter_factors(expr):
        base, exp = factor.as_base_exp()
        if base.is_number:
            # numeric factor -> dimensionless, ignore
            continue
        if not base.is_symbol:
            raise ValueError(
                f"term {term!r} is not a simple product of symbols "
                f"(offending factor: {factor})"
            )
        sym_dim = symbol_dims[str(base)]
        exp_value = _sympy_number_to_fraction(exp)
        result = result.multiply(sym_dim.power(exp_value))

    return result


def _iter_factors(expr: sp.Expr) -> Iterable[sp.Expr]:
    """Yield the multiplicative factors of a monomial expression."""
    if isinstance(expr, sp.Mul):
        for arg in expr.args:
            yield arg
    else:
        yield expr


def _sympy_number_to_fraction(value: sp.Expr) -> Fraction:
    if value.is_Integer:
        return Fraction(int(value))
    if value.is_Rational:
        return Fraction(int(value.p), int(value.q))
    if value.is_number:
        return Fraction(float(value)).limit_denominator(10**6)
    raise ValueError(f"non-numeric exponent {value!r}")


def side_term_dimensions(
    terms: Sequence[str], symbol_dims: Mapping[str, Dimension]
) -> Dict[str, Dimension]:
    """Map each additive term string to its computed Dimension."""
    return {term: term_dimension(term, symbol_dims) for term in terms}
