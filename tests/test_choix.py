import importlib.util
import unittest

from mcda.methods.choix import ChoixMethod


@unittest.skipUnless(importlib.util.find_spec("choix"), "choix not installed")
class TestChoixMethod(unittest.TestCase):
    def test_compute_weights(self) -> None:
        method = ChoixMethod()
        criteria = ["Cost", "Quality"]
        pairwise = [
            [1.0, 2.0],
            [0.5, 1.0],
        ]

        result = method.compute_weights(criteria, pairwise)
        self.assertEqual(len(result.weights), 2)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=3)
        self.assertGreater(result.weights[0], result.weights[1])
