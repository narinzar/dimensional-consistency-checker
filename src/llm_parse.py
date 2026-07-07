"""Parse a physics/engineering equation into structured JSON using Claude.

The model is asked to return strict JSON describing:

  - ``symbols``: a dict mapping each symbol name to a dict of base-dimension
    exponents (keys drawn from the seven SI base dimensions: mass, length,
    time, current, temperature, amount, luminous).
  - ``equation``: an object with ``lhs`` and ``rhs``, each a list of additive
    term strings (e.g. "m*a", "1/2*m*v**2").

Parse failures are handled gracefully: a ParseError is raised with a clear
message rather than letting a raw JSON or API error propagate.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv

# Model id is fixed by the project brief.
MODEL_ID = "claude-sonnet-5"

# The seven SI base dimensions the model is allowed to use.
ALLOWED_DIMENSIONS = (
    "mass",
    "length",
    "time",
    "current",
    "temperature",
    "amount",
    "luminous",
)

SYSTEM_PROMPT = (
    "You convert a single physics or engineering equation into strict JSON. "
    "You never add prose, explanation, or markdown fences. You output only a "
    "JSON object."
)

USER_TEMPLATE = """Convert the following equation into JSON.

Equation (plain text or LaTeX):
{equation}

Rules:
- Identify every symbol in the equation and assign it a physical dimension.
- A dimension is a JSON object of base-dimension exponents. The only allowed
  keys are: mass, length, time, current, temperature, amount, luminous.
  Omit keys whose exponent is zero. Exponents may be integers or simple
  fractions expressed as numbers (e.g. 0.5). A dimensionless quantity is the
  empty object {{}}.
- Split each side of the equation into a list of additive terms. Write each
  term as a plain Python-style algebraic expression using the symbol names,
  * for multiplication, / for division, and ** for powers. Pure numeric
  factors (like 1/2 or 2) may appear and are treated as dimensionless.
- Do not invent symbols that are not in the equation. Use the symbol names as
  they appear (map Greek letters to short ASCII names, e.g. rho, omega, mu).

Return exactly this shape:
{{
  "symbols": {{ "<name>": {{ "<base_dim>": <exponent>, ... }}, ... }},
  "equation": {{ "lhs": ["term", ...], "rhs": ["term", ...] }}
}}

Common reference dimensions you may use:
- velocity: {{"length": 1, "time": -1}}
- acceleration: {{"length": 1, "time": -2}}
- force: {{"mass": 1, "length": 1, "time": -2}}
- energy: {{"mass": 1, "length": 2, "time": -2}}
- mass: {{"mass": 1}}
- length/position: {{"length": 1}}
- time: {{"time": 1}}

Output only the JSON object."""


class ParseError(Exception):
    """Raised when the equation cannot be parsed into structured JSON."""


@dataclass
class ParsedEquation:
    """Structured result of parsing an equation."""

    symbols: Dict[str, Dict[str, float]]
    lhs: List[str]
    rhs: List[str]

    @property
    def terms(self) -> List[str]:
        return list(self.lhs) + list(self.rhs)


def _load_api_key() -> str:
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ParseError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your key."
        )
    return key


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model added them anyway."""
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return text


def _validate_structure(data: object) -> ParsedEquation:
    """Validate and normalize the model's JSON into a ParsedEquation."""
    if not isinstance(data, dict):
        raise ParseError("top-level JSON is not an object")

    symbols = data.get("symbols")
    equation = data.get("equation")
    if not isinstance(symbols, dict):
        raise ParseError("'symbols' is missing or not an object")
    if not isinstance(equation, dict):
        raise ParseError("'equation' is missing or not an object")

    lhs = equation.get("lhs")
    rhs = equation.get("rhs")
    if not isinstance(lhs, list) or not isinstance(rhs, list):
        raise ParseError("'equation.lhs' and 'equation.rhs' must be lists")
    if not lhs or not rhs:
        raise ParseError("each side of the equation needs at least one term")

    norm_symbols: Dict[str, Dict[str, float]] = {}
    for name, dim in symbols.items():
        if not isinstance(dim, dict):
            raise ParseError(f"dimension for symbol {name!r} is not an object")
        bad_keys = [k for k in dim if k not in ALLOWED_DIMENSIONS]
        if bad_keys:
            raise ParseError(
                f"symbol {name!r} uses unknown base dimension(s): {bad_keys}"
            )
        norm_symbols[str(name)] = {str(k): v for k, v in dim.items()}

    lhs_terms = [str(t) for t in lhs]
    rhs_terms = [str(t) for t in rhs]

    return ParsedEquation(symbols=norm_symbols, lhs=lhs_terms, rhs=rhs_terms)


def parse_response_text(text: str) -> ParsedEquation:
    """Parse the raw text returned by the model into a ParsedEquation.

    Split out from the network call so it can be unit-tested without hitting
    the API.
    """
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ParseError(f"model did not return valid JSON: {exc}") from exc
    return _validate_structure(data)


def parse_equation(equation: str, *, client=None) -> ParsedEquation:
    """Call Claude to parse ``equation`` into a ParsedEquation.

    ``client`` may be injected for testing; when omitted a real Anthropic
    client is constructed. Any API or JSON problem is surfaced as a
    ParseError.
    """
    if not equation or not equation.strip():
        raise ParseError("equation string is empty")

    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=_load_api_key())

    try:
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(equation=equation.strip()),
                }
            ],
        )
    except Exception as exc:  # network, auth, rate limit, etc.
        raise ParseError(f"LLM request failed: {exc}") from exc

    text = _extract_text(response)
    if not text:
        raise ParseError("model returned an empty response")
    return parse_response_text(text)


def _extract_text(response) -> str:
    """Concatenate the text blocks of a Claude Messages response."""
    parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()
