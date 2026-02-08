from __future__ import annotations

from typing import List

import numpy as np

from mcda.core import MethodResult
from mcda.methods.ahp import AHPMethod


class ChoixMethod(AHPMethod):
    id = "choix"
    name = "Choix (Bradley-Terry)"

    _ratio_scale = [2.0, 3.0, 5.0, 7.0, 9.0]
    _strength_map = {2.0: 1, 3.0: 2, 5.0: 3, 7.0: 4, 9.0: 5}

    def compute_weights(self, criteria: List[str], pairwise: List[List[float]]) -> MethodResult:
        n_items = len(criteria)
        if n_items == 0:
            return MethodResult(weights=[], consistency_ratio=None)
        if n_items == 1:
            return MethodResult(weights=[1.0], consistency_ratio=None)

        comp_mat = np.zeros((n_items, n_items), dtype=float)
        for i in range(n_items):
            for j in range(i + 1, n_items):
                ratio = float(pairwise[i][j]) if pairwise else 1.0
                if ratio <= 0:
                    continue
                if abs(ratio - 1.0) < 1e-9:
                    comp_mat[i, j] += 1.0
                    comp_mat[j, i] += 1.0
                else:
                    strength = self._ratio_to_strength(ratio)
                    if ratio > 1.0:
                        comp_mat[i, j] += strength
                    else:
                        comp_mat[j, i] += strength

        if comp_mat.sum() == 0:
            return MethodResult(weights=[1.0 / n_items] * n_items, consistency_ratio=None)

        from choix import lsr_pairwise_dense

        params = lsr_pairwise_dense(comp_mat, alpha=1e-4)
        weights = np.exp(params)
        total = weights.sum()
        if total <= 0:
            weights = np.ones(n_items) / n_items
        else:
            weights = weights / total
        return MethodResult(weights=weights.tolist(), consistency_ratio=None)

    @classmethod
    def _ratio_to_strength(cls, ratio: float) -> int:
        value = ratio if ratio >= 1.0 else 1.0 / ratio
        closest = min(cls._ratio_scale, key=lambda candidate: abs(candidate - value))
        return cls._strength_map[closest]
