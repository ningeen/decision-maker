import unittest

from mcda.core import MethodContext
from mcda.methods.bwm import BWMMethod


class TestBWMMethod(unittest.TestCase):
    def test_bwm_compute_weights_consistent(self) -> None:
        method = BWMMethod()
        criteria = ["Speed", "Cost", "Risk"]
        context = MethodContext(
            best_index=0,
            worst_index=2,
            best_to_others=[1.0, 2.0, 6.0],
            others_to_worst=[6.0, 3.0, 1.0],
        )

        result = method.compute_weights(criteria, pairwise=[], context=context)
        self.assertEqual(len(result.weights), 3)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=3)
        self.assertAlmostEqual(result.weights[0], 0.6, places=2)
        self.assertAlmostEqual(result.weights[1], 0.3, places=2)
        self.assertAlmostEqual(result.weights[2], 0.1, places=2)

    def test_bwm_compute_scores_ranks_best(self) -> None:
        method = BWMMethod()
        weights = [0.6, 0.3, 0.1]
        option_scores = {
            "Option A": [0.0, 0.0, 0.0],
            "Option B": [10.0, 0.0, 0.0],
        }

        ranked = method.compute_scores(weights, option_scores)
        self.assertEqual(ranked[0][0], "Option B")
        self.assertGreater(ranked[0][1], ranked[1][1])
