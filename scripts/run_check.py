"""CLI: check the dimensional consistency of an equation.

Usage:
    python scripts/run_check.py --equation "F = m*a"
    python scripts/run_check.py --equation "E = m*c^2"

Requires ANTHROPIC_API_KEY in .env (the LLM parses the equation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script: add the repo root to sys.path so `src` imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_parse import ParseError  # noqa: E402
from src.pipeline import format_report, run_pipeline  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the dimensional consistency of a physics/engineering equation."
    )
    parser.add_argument(
        "--equation",
        required=True,
        help='the equation to check, e.g. "F = m*a" or LaTeX "E = m c^2"',
    )
    args = parser.parse_args(argv)

    try:
        pr = run_pipeline(args.equation)
    except ParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_report(pr))
    # Exit code 0 when consistent, 1 when inconsistent - useful in scripts.
    return 0 if pr.result.consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
