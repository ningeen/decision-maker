from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from mcda.core import MCDAMethod, MethodResult


class AHPMethod(MCDAMethod):
    id = "ahp"
    name = "Analytic Hierarchy Process"

    def compute_weights(self, criteria: List[str], pairwise: List[List[float]]) -> MethodResult:
        if len(criteria) < 2:
            return MethodResult(weights=[1.0] * len(criteria), consistency_ratio=None)

        try:
            from pyDecision.algorithm import ahp_method
        except ModuleNotFoundError:
            from pydecision.algorithm import ahp_method

        matrix = np.array(pairwise, dtype=float)
        weights, rc = ahp_method(matrix, wd="geometric")
        return MethodResult(weights=weights.tolist(), consistency_ratio=float(rc))

    def compute_scores(
        self,
        weights: List[float],
        option_scores: Dict[str, List[float]],
    ) -> List[Tuple[str, float]]:
        if not option_scores:
            return []

        options = list(option_scores.keys())
        scores_matrix = np.array([option_scores[option] for option in options], dtype=float)
        normalized = self._normalize(scores_matrix)
        weights_array = np.array(weights, dtype=float)
        weighted_scores = normalized.dot(weights_array)
        results = list(zip(options, weighted_scores.tolist()))
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        mins = scores.min(axis=0)
        maxs = scores.max(axis=0)
        ranges = maxs - mins
        normalized = np.zeros_like(scores, dtype=float)
        for idx in range(scores.shape[1]):
            if ranges[idx] == 0:
                normalized[:, idx] = 1.0
            else:
                normalized[:, idx] = (scores[:, idx] - mins[idx]) / ranges[idx]
        return normalized
