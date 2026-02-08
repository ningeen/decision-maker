from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Result:
    option: str
    score: float


@dataclass
class Project:
    name: str
    method_id: str = "ahp"
    criteria: List[str] = field(default_factory=list)
    pairwise: List[List[float]] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    consistency_ratio: Optional[float] = None
    options: List[str] = field(default_factory=list)
    scores: List[List[float]] = field(default_factory=list)
    results: List[Result] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "method_id": self.method_id,
            "criteria": list(self.criteria),
            "pairwise": [list(row) for row in self.pairwise],
            "weights": list(self.weights),
            "consistency_ratio": self.consistency_ratio,
            "options": list(self.options),
            "scores": [list(row) for row in self.scores],
            "results": [result.__dict__ for result in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        results = [Result(**item) for item in data.get("results", [])]
        return cls(
            name=data.get("name", "Untitled"),
            method_id=data.get("method_id", "ahp"),
            criteria=list(data.get("criteria", [])),
            pairwise=[list(row) for row in data.get("pairwise", [])],
            weights=list(data.get("weights", [])),
            consistency_ratio=data.get("consistency_ratio"),
            options=list(data.get("options", [])),
            scores=[list(row) for row in data.get("scores", [])],
            results=results,
        )
