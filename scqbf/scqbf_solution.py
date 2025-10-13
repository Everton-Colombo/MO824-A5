from dataclasses import dataclass
from typing import List, Set

@dataclass
class ScQbfSolution:
    elements: List[int]
    _objfun_val: float = None

    def __str__(self):
        return f"ScQbfSolution(_objfun_val={self._objfun_val:.2f}, elements={len(self.elements)})"