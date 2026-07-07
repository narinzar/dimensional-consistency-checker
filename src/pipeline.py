"""Orchestrate parse -> check -> suggest for an equation string."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .llm_parse import ParseError, ParsedEquation, parse_equation
from .suggest import ConsistencyResult, build_symbol_dims, check_and_suggest


@dataclass
class PipelineResult:
    equation: str
    parsed: ParsedEquation
    result: ConsistencyResult


def run_pipeline(equation: str, *, client=None) -> PipelineResult:
    """Parse ``equation`` with the LLM, then check consistency and suggest.

    Raises ParseError if the equation cannot be parsed or if a term references
    a symbol without a dimension.
    """
    parsed = parse_equation(equation, client=client)
    return _analyze(equation, parsed)


def analyze_parsed(equation: str, parsed: ParsedEquation) -> PipelineResult:
    """Run check + suggest on an already-parsed equation (no LLM call)."""
    return _analyze(equation, parsed)


def _analyze(equation: str, parsed: ParsedEquation) -> PipelineResult:
    try:
        symbol_dims = build_symbol_dims(parsed.symbols)
        result = check_and_suggest(parsed.lhs, parsed.rhs, symbol_dims)
    except ValueError as exc:
        # Turn dimension-algebra errors into a ParseError so callers only need
        # to handle one exception type.
        raise ParseError(str(exc)) from exc
    return PipelineResult(equation=equation, parsed=parsed, result=result)


def format_report(pr: PipelineResult) -> str:
    """Render a human-readable report of a PipelineResult."""
    lines = []
    lines.append(f"Equation: {pr.equation}")
    lhs = " + ".join(pr.parsed.lhs)
    rhs = " + ".join(pr.parsed.rhs)
    lines.append(f"Parsed as: {lhs} = {rhs}")
    lines.append("")
    lines.append("Per-term dimensions:")
    for report in pr.result.term_dimensions:
        lines.append(f"  {report.term:>20}  ->  {report.dimension}")
    lines.append("")
    if pr.result.consistent:
        lines.append(
            f"CONSISTENT: every term has dimension {pr.result.expected_dimension}"
        )
    else:
        lines.append(
            f"INCONSISTENT: expected dimension is {pr.result.expected_dimension}"
        )
        for s in pr.result.suggestions:
            lines.append(
                f"  outlier term '{s.term}' has dimension {s.actual}; "
                f"to fix, {s.message}"
            )
    return "\n".join(lines)
