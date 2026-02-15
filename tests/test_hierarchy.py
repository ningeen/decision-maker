import unittest

from mcda.hierarchy import aggregate_option_scores


class TestHierarchyAggregation(unittest.TestCase):
    def test_aggregate_option_scores_with_subcriteria(self) -> None:
        options = ["Option A", "Option B"]
        criteria = ["Cost", "Quality"]
        scores = [
            [5.0, 9.0],
            [8.0, 6.0],
        ]
        subcriteria = [["Upfront", "Ongoing"], []]
        sub_scores = [
            [
                [2.0, 8.0],
                [6.0, 4.0],
            ],
            [],
        ]
        sub_weights = [[0.25, 0.75], []]

        result = aggregate_option_scores(
            options, criteria, scores, subcriteria, sub_scores, sub_weights
        )
        self.assertAlmostEqual(result["Option A"][0], 6.5, places=3)
        self.assertAlmostEqual(result["Option A"][1], 9.0, places=3)
        self.assertAlmostEqual(result["Option B"][0], 4.5, places=3)
        self.assertAlmostEqual(result["Option B"][1], 6.0, places=3)

    def test_aggregate_option_scores_defaults_weights(self) -> None:
        options = ["Option A"]
        criteria = ["Cost"]
        scores = [[7.0]]
        subcriteria = [["Capex", "Opex"]]
        sub_scores = [[[10.0, 0.0]]]
        sub_weights = [[]]

        result = aggregate_option_scores(
            options, criteria, scores, subcriteria, sub_scores, sub_weights
        )
        self.assertAlmostEqual(result["Option A"][0], 5.0, places=3)
