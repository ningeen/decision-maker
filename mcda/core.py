from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class MethodResult:
    weights: List[float]
    consistency_ratio: float | None = None


@dataclass
class MethodContext:
    criteria: List[str] = field(default_factory=list)
    directions: List[str] = field(default_factory=list)
    preference_functions: List[str] = field(default_factory=list)
    q: List[float] = field(default_factory=list)
    p: List[float] = field(default_factory=list)
    s: List[float] = field(default_factory=list)
    weights_raw: List[float] = field(default_factory=list)


class MCDAMethod(ABC):
    id: str
    name: str

    @abstractmethod
    def compute_weights(
        self,
        criteria: List[str],
        pairwise: List[List[float]],
        context: MethodContext | None = None,
    ) -> MethodResult:
        raise NotImplementedError

    @abstractmethod
    def compute_scores(
        self,
        weights: List[float],
        option_scores: Dict[str, List[float]],
        context: MethodContext | None = None,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError
