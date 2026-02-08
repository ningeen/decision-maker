import unittest

from mcda.methods.ahp import AHPMethod


class TestAHPMethod(unittest.TestCase):
    def test_ahp_compute_scores_ranks_best(self) -> None:
        method = AHPMethod()
        weights = [0.5, 0.5]
        option_scores = {
            "Option A": [0.0, 0.0],
            "Option B": [10.0, 0.0],
        }

        ranked = method.compute_scores(weights, option_scores)
        self.assertEqual(ranked[0][0], "Option B")
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_ahp_compute_weights_basic(self) -> None:
        method = AHPMethod()
        criteria = ["Cost", "Quality"]
        pairwise = [
            [1.0, 3.0],
            [1.0 / 3.0, 1.0],
        ]

        result = method.compute_weights(criteria, pairwise)
        self.assertEqual(len(result.weights), 2)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=3)
        self.assertAlmostEqual(result.weights[0], 0.75, places=2)
        self.assertAlmostEqual(result.weights[1], 0.25, places=2)
        self.assertIsNotNone(result.consistency_ratio)
