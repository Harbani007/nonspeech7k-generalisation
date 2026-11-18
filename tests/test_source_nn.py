import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dl_metrics import source_nn_rate  # noqa: E402


class SourceNNRateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = np.repeat([0, 1], 60)
        self.g = np.repeat(np.arange(24), 5)  # 12 sources per class, 5 clips each
        self.rng = rng

    def test_source_encoding_embedding_gives_rate_one(self):  # positive control
        E = np.eye(24)[self.g] + 0.01 * self.rng.standard_normal((120, 24))
        r = source_nn_rate(E, self.y, self.g)
        self.assertEqual(r["rate"], 1.0)

    def test_random_embedding_is_near_chance(self):  # negative control
        rates = [source_nn_rate(self.rng.standard_normal((120, 32)), self.y, self.g)["rate"] for _ in range(30)]
        chance = source_nn_rate(self.rng.standard_normal((120, 32)), self.y, self.g)["chance"]
        self.assertAlmostEqual(chance, 4 / 59)
        self.assertLess(abs(np.mean(rates) - chance), 0.03)

    def test_self_is_excluded_even_for_identical_copies(self):
        E = self.rng.standard_normal((120, 8))
        r = source_nn_rate(E, self.y, np.arange(120))  # every clip its own source: rate must be 0
        self.assertEqual(r["rate"], 0.0)
        self.assertEqual(r["n_eligible"], 0)

    def test_neighbours_only_within_class(self):
        # make each clip's closest vector a clip of the OTHER class from the same group id; must be ignored
        y = np.array([0, 0, 1, 1]); g = np.array([0, 1, 0, 1])
        E = np.array([[1, 0], [0, 1], [1, 0.01], [0.01, 1]], float)
        self.assertEqual(source_nn_rate(E, y, g)["rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
