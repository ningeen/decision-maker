from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class MethodResult:
    weights: List[float]
    consistency_ratio: float | None = None


class MCDAMethod(ABC):
    id: str
    name: str

    @abstractmethod
    def compute_weights(self, criteria: List[str], pairwise: List[List[float]]) -> MethodResult:
        raise NotImplementedError

    @abstractmethod
    def compute_scores(
        self,
        weights: List[float],
        option_scores: Dict[str, List[float]],
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError
