from __future__ import annotations

from typing import Dict, List


def aggregate_option_scores(
    options: List[str],
    criteria: List[str],
    scores: List[List[float]],
    subcriteria: List[List[str]],
    sub_scores: List[List[List[float]]],
    sub_weights: List[List[float]],
) -> Dict[str, List[float]]:
    if not options or not criteria:
        return {}

    option_scores: Dict[str, List[float]] = {}
    for option_index, option in enumerate(options):
        scores_row: List[float] = []
        for criterion_index in range(len(criteria)):
            sub_items = subcriteria[criterion_index] if criterion_index < len(subcriteria) else []
            if sub_items:
                weights = sub_weights[criterion_index] if criterion_index < len(sub_weights) else []
                if len(weights) != len(sub_items):
                    weights = [1.0 / len(sub_items)] * len(sub_items)
                sub_row = []
                if criterion_index < len(sub_scores) and option_index < len(sub_scores[criterion_index]):
                    sub_row = list(sub_scores[criterion_index][option_index])
                if len(sub_row) < len(weights):
                    sub_row = sub_row + [0.0] * (len(weights) - len(sub_row))
                scores_row.append(sum(w * s for w, s in zip(weights, sub_row)))
            else:
                value = 0.0
                if option_index < len(scores) and criterion_index < len(scores[option_index]):
                    value = float(scores[option_index][criterion_index])
                scores_row.append(value)
        option_scores[option] = scores_row
    return option_scores
