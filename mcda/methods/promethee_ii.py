from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from mcda.core import MCDAMethod, MethodContext, MethodResult


class PrometheeIIMethod(MCDAMethod):
    id = "promethee_ii"
    name = "PROMETHEE II"

    def compute_weights(
        self,
        criteria: List[str],
        pairwise: List[List[float]],
        context: MethodContext | None = None,
    ) -> MethodResult:
        n_items = len(criteria)
        if n_items == 0:
            return MethodResult(weights=[], consistency_ratio=None)

        raw = list(context.weights_raw) if context and context.weights_raw else [1.0] * n_items
        weights = self._normalize_weights(raw, n_items)
        return MethodResult(weights=weights, consistency_ratio=None)

    def compute_scores(
        self,
        weights: List[float],
        option_scores: Dict[str, List[float]],
        context: MethodContext | None = None,
    ) -> List[Tuple[str, float]]:
        if not option_scores:
            return []

        options = list(option_scores.keys())
        dataset = np.array([option_scores[option] for option in options], dtype=float)
        n_criteria = dataset.shape[1]

        directions = list(context.directions) if context else []
        for idx in range(n_criteria):
            if idx < len(directions) and directions[idx] == "min":
                dataset[:, idx] = -dataset[:, idx]

        if len(weights) != n_criteria:
            raw = list(context.weights_raw) if context and context.weights_raw else [1.0] * n_criteria
            weights = self._normalize_weights(raw, n_criteria)

        functions = self._pad_list(
            list(context.preference_functions) if context else [], n_criteria, "t1"
        )
        q = self._pad_list(list(context.q) if context else [], n_criteria, 0.0)
        p = self._pad_list(list(context.p) if context else [], n_criteria, 0.0)
        s = self._pad_list(list(context.s) if context else [], n_criteria, 0.0)

        try:
            from pyDecision.algorithm import promethee_ii
        except ModuleNotFoundError:
            from pydecision.algorithm import promethee_ii

        flow = promethee_ii(
            dataset,
            weights,
            q,
            s,
            p,
            functions,
            sort=True,
            topn=0,
            graph=False,
            verbose=False,
        )

        results: List[Tuple[str, float]] = []
        for row in flow:
            index = int(row[0]) - 1
            if 0 <= index < len(options):
                results.append((options[index], float(row[1])))
        return results

    @staticmethod
    def _pad_list(values: List[float | str], size: int, default: float | str) -> List[float | str]:
        padded = list(values[:size])
        while len(padded) < size:
            padded.append(default)
        return padded

    @staticmethod
    def _normalize_weights(values: List[float], size: int) -> List[float]:
        weights = [max(0.0, float(value)) for value in values[:size]]
        while len(weights) < size:
            weights.append(1.0)
        total = sum(weights)
        if total <= 0:
            return [1.0 / size] * size
        return [value / total for value in weights]
