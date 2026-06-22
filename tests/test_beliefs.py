import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clarification_guided_interaction import initialize_robot_beliefs  # noqa: E402


class BeliefInitializationTests(unittest.TestCase):
    def test_initialize_robot_beliefs_returns_uniform_distribution(self):
        beliefs = initialize_robot_beliefs([{}, {}, {}, {}])

        self.assertEqual(beliefs.shape, (4,))
        self.assertAlmostEqual(float(beliefs.sum()), 1.0)
        np.testing.assert_allclose(beliefs, np.full(4, 0.25))


if __name__ == "__main__":
    unittest.main()
