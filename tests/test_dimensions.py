"""Direct, no-LLM tests using hand-built structured inputs."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dimensions import Dimension, term_dimension  # noqa: E402
from src.pipeline import analyze_parsed  # noqa: E402
from src.llm_parse import ParsedEquation  # noqa: E402

# Reference dimensions used across tests.
MASS = Dimension.from_mapping({"mass": 1})
LENGTH = Dimension.from_mapping({"length": 1})
TIME = Dimension.from_mapping({"time": 1})
VELOCITY = Dimension.from_mapping({"length": 1, "time": -1})
ACCEL = Dimension.from_mapping({"length": 1, "time": -2})
FORCE = Dimension.from_mapping({"mass": 1, "length": 1, "time": -2})
ENERGY = Dimension.from_mapping({"mass": 1, "length": 2, "time": -2})


class DimensionAlgebraTests(unittest.TestCase):
    def test_multiply_and_divide(self):
        # velocity * time = length
        self.assertEqual(VELOCITY.multiply(TIME), LENGTH)
        # length / time = velocity
        self.assertEqual(LENGTH.divide(TIME), VELOCITY)

    def test_power(self):
        # velocity squared = length^2 time^-2
        self.assertEqual(
            VELOCITY.power(2),
            Dimension.from_mapping({"length": 2, "time": -2}),
        )

    def test_dimensionless_equality(self):
        self.assertTrue(Dimension.dimensionless().is_dimensionless())
        # velocity / velocity is dimensionless
        self.assertTrue(VELOCITY.divide(VELOCITY).is_dimensionless())

    def test_term_dimension_numeric_factor_is_dimensionless(self):
        # 1/2 * m * v**2 has energy dimension; the 1/2 drops out.
        dims = {"m": MASS, "v": VELOCITY}
        self.assertEqual(term_dimension("1/2*m*v**2", dims), ENERGY)
        self.assertEqual(term_dimension("m*v**2", dims), ENERGY)


class ConsistencyTests(unittest.TestCase):
    def _analyze(self, symbols, lhs, rhs):
        parsed = ParsedEquation(symbols=symbols, lhs=lhs, rhs=rhs)
        return analyze_parsed("test", parsed).result

    def test_force_equals_mass_times_acceleration_is_consistent(self):
        # F = m*a
        res = self._analyze(
            symbols={
                "F": {"mass": 1, "length": 1, "time": -2},
                "m": {"mass": 1},
                "a": {"length": 1, "time": -2},
            },
            lhs=["F"],
            rhs=["m*a"],
        )
        self.assertTrue(res.consistent)
        self.assertEqual(res.expected_dimension, FORCE)
        self.assertEqual(res.suggestions, [])

    def test_energy_equals_mc_squared_is_consistent(self):
        # E = m*c**2
        res = self._analyze(
            symbols={
                "E": {"mass": 1, "length": 2, "time": -2},
                "m": {"mass": 1},
                "c": {"length": 1, "time": -1},
            },
            lhs=["E"],
            rhs=["m*c**2"],
        )
        self.assertTrue(res.consistent)
        self.assertEqual(res.expected_dimension, ENERGY)

    def test_position_equals_velocity_is_inconsistent(self):
        # x = v  (position equals velocity) -> inconsistent
        res = self._analyze(
            symbols={
                "x": {"length": 1},
                "v": {"length": 1, "time": -1},
            },
            lhs=["x"],
            rhs=["v"],
        )
        self.assertFalse(res.consistent)
        # Exactly one outlier term should be flagged.
        self.assertEqual(len(res.suggestions), 1)
        flagged = res.suggestions[0]
        # The velocity term is the outlier (expected dimension is length,
        # since x=length appears first and is the majority tie-break).
        self.assertEqual(flagged.term, "v")
        self.assertEqual(flagged.actual, VELOCITY)
        self.assertEqual(flagged.expected, LENGTH)
        # The fix is to multiply by a time (velocity * time = length).
        self.assertIn("time", flagged.message)

    def test_missing_half_factor_stays_consistent(self):
        # E = m*v**2 vs E = 1/2*m*v**2 are both dimensionally consistent;
        # the missing 1/2 is dimensionless so it is never flagged.
        res = self._analyze(
            symbols={
                "E": {"mass": 1, "length": 2, "time": -2},
                "m": {"mass": 1},
                "v": {"length": 1, "time": -1},
            },
            lhs=["E"],
            rhs=["m*v**2"],
        )
        self.assertTrue(res.consistent)
        self.assertEqual(res.suggestions, [])


if __name__ == "__main__":
    unittest.main()
