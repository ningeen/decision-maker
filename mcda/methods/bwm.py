from __future__ import annotations

from typing import Dict, List, Tuple
import math

import numpy as np

from mcda.core import MCDAMethod, MethodContext, MethodResult


class BWMMethod(MCDAMethod):
    id = "bwm"
    name = "Best-Worst Method"

    def compute_weights(
        self,
        criteria: List[str],
        pairwise: List[List[float]],
        context: MethodContext | None = None,
    ) -> MethodResult:
        n_items = len(criteria)
        if n_items == 0:
            return MethodResult(weights=[], consistency_ratio=None)
        if n_items == 1:
            return MethodResult(weights=[1.0], consistency_ratio=0.0)

        best_index, worst_index = self._resolve_indices(context, n_items)
        best_to_others = self._pad_list(
            list(context.best_to_others) if context else [], n_items, 1.0
        )
        others_to_worst = self._pad_list(
            list(context.others_to_worst) if context else [], n_items, 1.0
        )

        best_to_others = [max(1.0, float(value)) for value in best_to_others]
        others_to_worst = [max(1.0, float(value)) for value in others_to_worst]
        best_to_others[best_index] = 1.0
        others_to_worst[worst_index] = 1.0

        weights = [1.0 / n_items] * n_items
        weights = self._optimize_weights(
            weights,
            best_index,
            worst_index,
            best_to_others,
            others_to_worst,
        )
        xi = self._max_deviation(
            weights,
            best_index,
            worst_index,
            best_to_others,
            others_to_worst,
        )
        return MethodResult(weights=weights, consistency_ratio=xi)

    def compute_scores(
        self,
        weights: List[float],
        option_scores: Dict[str, List[float]],
        context: MethodContext | None = None,
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

    @staticmethod
    def _pad_list(values: List[float], size: int, default: float) -> List[float]:
        padded = list(values[:size])
        while len(padded) < size:
            padded.append(default)
        return padded

    @staticmethod
    def _resolve_indices(context: MethodContext | None, size: int) -> tuple[int, int]:
        best_index = 0
        worst_index = size - 1 if size > 1 else 0
        if context:
            if context.best_index is not None and 0 <= context.best_index < size:
                best_index = int(context.best_index)
            if context.worst_index is not None and 0 <= context.worst_index < size:
                worst_index = int(context.worst_index)
        if size > 1 and best_index == worst_index:
            worst_index = (best_index + 1) % size
        return best_index, worst_index

    def _optimize_weights(
        self,
        weights: List[float],
        best_index: int,
        worst_index: int,
        best_to_others: List[float],
        others_to_worst: List[float],
    ) -> List[float]:
        step_base = 0.2
        for iteration in range(1, 5001):
            max_abs, grad = self._max_residual_and_grad(
                weights,
                best_index,
                worst_index,
                best_to_others,
                others_to_worst,
            )
            if max_abs < 1e-8:
                break
            step = step_base / math.sqrt(iteration)
            weights = [w - step * g for w, g in zip(weights, grad)]
            weights = self._project_simplex(weights)
        return weights

    @staticmethod
    def _max_residual_and_grad(
        weights: List[float],
        best_index: int,
        worst_index: int,
        best_to_others: List[float],
        others_to_worst: List[float],
    ) -> tuple[float, List[float]]:
        size = len(weights)
        max_abs = -1.0
        grad = [0.0] * size

        for idx in range(size):
            residual = weights[best_index] - best_to_others[idx] * weights[idx]
            abs_residual = abs(residual)
            if abs_residual > max_abs + 1e-12:
                max_abs = abs_residual
                grad = [0.0] * size
                sign = 1.0 if residual >= 0 else -1.0
                grad[best_index] += sign
                grad[idx] -= sign * best_to_others[idx]

        for idx in range(size):
            residual = weights[idx] - others_to_worst[idx] * weights[worst_index]
            abs_residual = abs(residual)
            if abs_residual > max_abs + 1e-12:
                max_abs = abs_residual
                grad = [0.0] * size
                sign = 1.0 if residual >= 0 else -1.0
                grad[idx] += sign
                grad[worst_index] -= sign * others_to_worst[idx]

        return max_abs, grad

    @staticmethod
    def _max_deviation(
        weights: List[float],
        best_index: int,
        worst_index: int,
        best_to_others: List[float],
        others_to_worst: List[float],
    ) -> float:
        max_abs = 0.0
        for idx in range(len(weights)):
            residual = weights[best_index] - best_to_others[idx] * weights[idx]
            max_abs = max(max_abs, abs(residual))
            residual = weights[idx] - others_to_worst[idx] * weights[worst_index]
            max_abs = max(max_abs, abs(residual))
        return max_abs

    @staticmethod
    def _project_simplex(values: List[float]) -> List[float]:
        size = len(values)
        if size == 0:
            return []
        ordered = sorted(values, reverse=True)
        cumulative = 0.0
        rho = -1
        theta = 0.0
        for idx, value in enumerate(ordered, start=1):
            cumulative += value
            t = (cumulative - 1.0) / idx
            if value - t > 0:
                rho = idx
                theta = t
        if rho == -1:
            return [1.0 / size] * size
        return [max(value - theta, 0.0) for value in values]
