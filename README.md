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

Numbers below are produced by running the commands above; this repo ships the
code, run it to populate them.

Reproduction:

```bash
pytest -q
python scripts/run_check.py --equation "F = m*a"
python scripts/run_check.py --equation "x = v"
```

Expected behavior:

- `pytest -q` passes with no API calls. The dimension algebra, the
  consistency check, and the suggester are exercised on hand-built inputs, and
  the LLM parse path is exercised through a mocked client.
- A dimensionally correct equation (`F = m*a`, `E = m*c^2`) reports
  `CONSISTENT` and lists every term with the same dimension. `F = m*a` should
  show each term as `M L T^-2`; `E = m*c^2` should show each term as
  `M L^2 T^-2`.
- A wrong equation such as `x = v` (position equals velocity) reports
  `INCONSISTENT`. The velocity term is flagged as the outlier and the
  suggestion notes that multiplying by a time (dimension `T`) would restore
  consistency, since velocity times time is a length.
- A missing dimensionless factor is never flagged: `E = m*v**2` and
  `E = 1/2*m*v**2` are both reported `CONSISTENT` because the `1/2` does not
  change any dimension.

Worked example (`x = v`):

```
Equation: x = v
Parsed as: x = v

Per-term dimensions:
                     x  ->  L
                     v  ->  L T^-1

INCONSISTENT: expected dimension is L
  outlier term 'v' has dimension L T^-1; to fix, multiply by a quantity with
  dimension time (missing factor dimension: T)
```

The exact parsed dimensions depend on the LLM's reading of each symbol; the
consistency verdict and the suggestion are computed deterministically from that
parse.

## What I'd do next at larger scale

Batch many equations in one LLM call and cache parses by equation hash so a
document full of equations is checked in a handful of requests. Add a unit-aware
mode that also tracks numeric SI prefixes (so `km` vs `m` mismatches surface),
and let the suggester rank candidate fixes when several terms could each be the
outlier.
