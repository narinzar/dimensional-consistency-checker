"""Tests for the suggester's outlier detection, plus a mocked-LLM parse test."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dimensions import Dimension  # noqa: E402
from src.suggest import build_symbol_dims, check_and_suggest  # noqa: E402
from src.llm_parse import ParsedEquation, ParseError, parse_equation  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402


class SuggesterTests(unittest.TestCase):
    def test_flags_the_outlier_term(self):
        # v = a + m  where a=acceleration and m=mass are wrong; the majority
        # dimension comes from... here we build a clearer case:
        # p = m*v + m   (momentum + a mass). Two terms are momentum, one is
        # mass, so mass is the outlier.
        symbols = {
            "p": {"mass": 1, "length": 1, "time": -1},
            "m": {"mass": 1},
            "v": {"length": 1, "time": -1},
        }
        dims = build_symbol_dims(symbols)
        res = check_and_suggest(lhs=["p"], rhs=["m*v", "m"], symbol_dims=dims)

        self.assertFalse(res.consistent)
        momentum = Dimension.from_mapping({"mass": 1, "length": 1, "time": -1})
        self.assertEqual(res.expected_dimension, momentum)

        flagged_terms = [s.term for s in res.suggestions]
        self.assertEqual(flagged_terms, ["m"])
        # The lone mass term needs a velocity to become momentum.
        self.assertIn("length", res.suggestions[0].message)
        self.assertIn("time", res.suggestions[0].message)

    def test_consistent_equation_has_no_suggestions(self):
        symbols = {
            "F": {"mass": 1, "length": 1, "time": -2},
            "m": {"mass": 1},
            "a": {"length": 1, "time": -2},
        }
        dims = build_symbol_dims(symbols)
        res = check_and_suggest(lhs=["F"], rhs=["m*a"], symbol_dims=dims)
        self.assertTrue(res.consistent)
        self.assertEqual(res.suggestions, [])


class MockedLLMParseTests(unittest.TestCase):
    """Ensure the pipeline works end-to-end without hitting the real API."""

    def _fake_client(self, json_text):
        """Build a mock Anthropic client returning ``json_text`` as one block."""
        text_block = mock.Mock()
        text_block.type = "text"
        text_block.text = json_text
        response = mock.Mock()
        response.content = [text_block]
        client = mock.Mock()
        client.messages.create.return_value = response
        return client

    def test_parse_equation_with_mocked_client(self):
        json_text = (
            '{"symbols": {"F": {"mass": 1, "length": 1, "time": -2}, '
            '"m": {"mass": 1}, "a": {"length": 1, "time": -2}}, '
            '"equation": {"lhs": ["F"], "rhs": ["m*a"]}}'
        )
        client = self._fake_client(json_text)
        parsed = parse_equation("F = m*a", client=client)
        self.assertIsInstance(parsed, ParsedEquation)
        self.assertEqual(parsed.lhs, ["F"])
        self.assertEqual(parsed.rhs, ["m*a"])
        client.messages.create.assert_called_once()

    def test_pipeline_with_mocked_client_detects_inconsistency(self):
        json_text = (
            '{"symbols": {"x": {"length": 1}, "v": {"length": 1, "time": -1}}, '
            '"equation": {"lhs": ["x"], "rhs": ["v"]}}'
        )
        client = self._fake_client(json_text)
        pr = run_pipeline("x = v", client=client)
        self.assertFalse(pr.result.consistent)
        self.assertEqual(len(pr.result.suggestions), 1)
        self.assertEqual(pr.result.suggestions[0].term, "v")

    def test_bad_json_raises_parse_error(self):
        client = self._fake_client("this is not json")
        with self.assertRaises(ParseError):
            parse_equation("F = m*a", client=client)

    def test_unknown_dimension_raises_parse_error(self):
        json_text = (
            '{"symbols": {"F": {"charge": 1}}, '
            '"equation": {"lhs": ["F"], "rhs": ["F"]}}'
        )
        client = self._fake_client(json_text)
        with self.assertRaises(ParseError):
            parse_equation("F = F", client=client)


if __name__ == "__main__":
    unittest.main()
