"""Suggest a fix when an equation is dimensionally inconsistent.

Strategy:

1. Compute the dimension of every additive term on both sides.
2. Find the majority (expected) dimension by counting how many terms share
   each dimension. The most common dimension is treated as the intended one.
3. Any term whose dimension differs from the expected one is an outlier and
   is flagged.
4. For each outlier, compute the ratio (expected / actual) and describe the
   change that would fix it. A ratio that is dimensionless means the term is
   already fine (e.g. a missing 1/2 factor never changes dimensions, so it is
   never flagged).

The suggester is deterministic and does not call the LLM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Mapping, Sequence

from .dimensions import (
    BASE_DIMENSIONS,
    Dimension,
    term_dimension,
)


@dataclass
class TermReport:
    term: str
    dimension: Dimension


@dataclass
class Suggestion:
    term: str
    actual: Dimension
    expected: Dimension
    message: str


@dataclass
class ConsistencyResult:
    consistent: bool
    expected_dimension: Dimension
    term_dimensions: List[TermReport] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)


def _describe_fix(actual: Dimension, expected: Dimension) -> str:
    """Describe the multiplicative change turning ``actual`` into ``expected``.

    The needed factor has dimension expected / actual. We render it as a
    product of base dimensions raised to powers and phrase it as "multiply by"
    or "divide by" for the common single-dimension cases.
    """
    needed = expected.divide(actual)
    if needed.is_dimensionless():
        # Same dimension already - only a numeric factor differs, which does
        # not affect dimensional consistency.
        return "term already has the expected dimension (only a numeric factor differs)"

    exps = needed.exponents
    # Human-friendly names for the base dimensions when phrasing a fix.
    factors_up = []
    factors_down = []
    for name, e in zip(BASE_DIMENSIONS, exps):
        if e == 0:
            continue
        magnitude = abs(e)
        piece = name if magnitude == 1 else f"{name}^{_fmt(magnitude)}"
        if e > 0:
            factors_up.append(piece)
        else:
            factors_down.append(piece)

    parts = []
    if factors_up:
        parts.append("multiply by a quantity with dimension " + " ".join(factors_up))
    if factors_down:
        parts.append("divide by a quantity with dimension " + " ".join(factors_down))
    return " and ".join(parts) + f" (missing factor dimension: {needed})"


def _fmt(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def check_and_suggest(
    lhs: Sequence[str],
    rhs: Sequence[str],
    symbol_dims: Mapping[str, Dimension],
) -> ConsistencyResult:
    """Check dimensional consistency and, if inconsistent, suggest fixes.

    ``lhs`` and ``rhs`` are lists of additive term strings; ``symbol_dims``
    maps each symbol to its Dimension.
    """
    all_terms = list(lhs) + list(rhs)
    reports = [
        TermReport(term=t, dimension=term_dimension(t, symbol_dims))
        for t in all_terms
    ]

    # Expected dimension = the most common one across all terms. Ties are
    # broken by first appearance for stability.
    counter: Counter = Counter(r.dimension for r in reports)
    # Counter.most_common preserves insertion order on ties in CPython 3.7+.
    expected = counter.most_common(1)[0][0]

    consistent = all(r.dimension == expected for r in reports)

    suggestions: List[Suggestion] = []
    if not consistent:
        for r in reports:
            if r.dimension != expected:
                suggestions.append(
                    Suggestion(
                        term=r.term,
                        actual=r.dimension,
                        expected=expected,
                        message=_describe_fix(r.dimension, expected),
                    )
                )

    return ConsistencyResult(
        consistent=consistent,
        expected_dimension=expected,
        term_dimensions=reports,
        suggestions=suggestions,
    )


def build_symbol_dims(
    symbols: Mapping[str, Mapping[str, object]]
) -> Dict[str, Dimension]:
    """Turn the LLM's symbol->exponent-mapping into Dimension objects."""
    return {name: Dimension.from_mapping(dim) for name, dim in symbols.items()}
