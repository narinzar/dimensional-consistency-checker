# Dimensional Consistency Checker

Takes a physics or engineering equation (plain text or LaTeX), uses Claude to
parse it into a structured form (symbols with their physical dimensions plus the
equation split into additive terms), then checks dimensional consistency
symbolically with sympy. When an equation is inconsistent, it flags the most
likely wrong term and suggests the change that would restore consistency.

## Problem

A dimensionally inconsistent equation is always wrong, but the error is easy to
miss when reading LaTeX or scratch notes. Checking by hand means tracking seven
base-dimension exponents (mass, length, time, current, temperature, amount,
luminous) through every additive term on both sides. The hard part is not the
arithmetic - it is turning free-form notation into a structured symbol table in
the first place, since a symbol like `c` could be a speed, a specific heat, or a
constant depending on context. This tool uses an LLM for the messy parsing step
and exact rational arithmetic for the part that must be correct.

## Approach

- An LLM (Claude, model `claude-sonnet-5`) converts an equation string into
  strict JSON: each symbol gets a dict of base-dimension exponents, and each
  side of the equation becomes a list of additive term strings.
- `Dimension` is a vector of the seven SI base-dimension exponents stored as
  exact `Fraction`s, with multiply/divide/power and value equality. No floating
  point error creeps into `1/2` powers or comparisons.
- sympy parses each additive term (e.g. `1/2*m*v**2`) into a monomial; numeric
  factors are dimensionless and drop out, and every symbol is replaced by its
  dimension to get the term's dimension.
- An equation is consistent when every term on both sides shares one dimension.
- When inconsistent, the suggester takes the majority dimension as the intended
  one, flags each outlier term, and computes the missing factor (expected /
  actual) to describe the fix. A missing `1/2` is dimensionless, so it is never
  flagged.
- LLM failures (bad JSON, missing fields, unknown base dimensions, network or
  auth errors) are caught and reported as a single `ParseError`.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then edit .env and add your ANTHROPIC_API_KEY
```

No GPU or torch is needed.

## How to run

Check a single equation (this calls the LLM to parse it):

```bash
python scripts/run_check.py --equation "F = m*a"
python scripts/run_check.py --equation "E = m*c^2"
python scripts/run_check.py --equation "x = v"
```

The script prints the parsed terms, each term's dimension, and either
`CONSISTENT` or `INCONSISTENT` with a suggested fix. Its exit code is `0` for a
consistent equation and `1` for an inconsistent one, so it can gate a script.

Run the tests (these never call the real API - the LLM is mocked):

```bash
pytest -q
```

## Results

Measured on a Windows workstation, CPU only (no GPU, no torch). The symbolic
checker was exercised for real; the live LLM parse path was not, because no
`ANTHROPIC_API_KEY` was configured in this environment (see the note below).

**Tests.** `pytest -q` passes **14/14** tests in about 0.25 s with no network
calls. These cover the exact-`Fraction` dimension algebra (multiply, divide,
power, dimensionless equality), `term_dimension` on real terms including the
`1/2*m*v**2` numeric-factor case, the consistency check and suggester on
hand-built symbol tables, and the LLM parse path through a mocked Anthropic
client (valid JSON, an inconsistent equation, malformed JSON, and an unknown
base dimension both surfaced as a single `ParseError`).

**Symbolic checker, run directly on structured inputs** (no API key needed,
via `src.pipeline.analyze_parsed`):

`F = m*a` (Newton's second law) is reported CONSISTENT, every term `M L T^-2`:

```
Per-term dimensions:
                     F  ->  M L T^-2
                   m*a  ->  M L T^-2
CONSISTENT: every term has dimension M L T^-2
```

`E = m*c**2` is reported CONSISTENT, every term `M L^2 T^-2`:

```
Per-term dimensions:
                     E  ->  M L^2 T^-2
                m*c**2  ->  M L^2 T^-2
CONSISTENT: every term has dimension M L^2 T^-2
```

`x = v` (position set equal to velocity) is reported INCONSISTENT. The velocity
term is flagged as the outlier and the suggester computes the missing factor
(expected / actual = `T`), i.e. multiply by a time to restore a length:

```
Per-term dimensions:
                     x  ->  L
                     v  ->  L T^-1
INCONSISTENT: expected dimension is L
  outlier term 'v' has dimension L T^-1; to fix, multiply by a quantity with
  dimension time (missing factor dimension: T)
```

A missing dimensionless factor is never flagged: with the same symbol table,
`E = m*v**2` and `E = 1/2*m*v**2` are both CONSISTENT, because the `1/2` drops
out as dimensionless.

Reproduce:

```bash
pytest -q
python -c "from src.llm_parse import ParsedEquation; from src.pipeline import analyze_parsed, format_report; \
print(format_report(analyze_parsed('x = v', ParsedEquation(symbols={'x':{'length':1},'v':{'length':1,'time':-1}}, lhs=['x'], rhs=['v']))))"
```

## Note on the LLM parse step

The text/LaTeX -> structured-JSON parse is done by Claude and needs
`ANTHROPIC_API_KEY`. That key was **not** configured in the environment used
for these results, so the end-to-end `scripts/run_check.py --equation "..."`
path (text in, verdict out) was not exercised here; without a key it exits
cleanly with `error: ANTHROPIC_API_KEY is not set`. Everything downstream of the
parse - the dimension algebra, the sympy term reduction, the consistency check,
and the fix suggester - is deterministic and was run for real on the structured
inputs shown above. No LLM output is reported or invented in this README.

## What I'd do next at larger scale

Batch many equations in one LLM call and cache parses by equation hash so a
document full of equations is checked in a handful of requests. Add a unit-aware
mode that also tracks numeric SI prefixes (so `km` vs `m` mismatches surface),
and let the suggester rank candidate fixes when several terms could each be the
outlier.
