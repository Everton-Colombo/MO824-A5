from dataclasses import dataclass, field
from typing import List, Set, Optional

@dataclass
class ScQbfSolution:
    elements: List[int]
    _objfun_val: Optional[float] = field(default=None, init=True, repr=False)
    _cached_elements_hash: Optional[int] = field(default=None, init=False, repr=False)

    @property
    def objfun_val(self) -> Optional[float]:
        """Returns cached objective function value if cache is valid, None otherwise."""
        current_hash = hash(tuple(sorted(self.elements)))
        if self._cached_elements_hash == current_hash:
            return self._objfun_val
        return None
    
    @objfun_val.setter
    def objfun_val(self, value: Optional[float]):
        """Sets the objective function value and updates the cache hash."""
        self._objfun_val = value
        if value is not None:
            self._cached_elements_hash = hash(tuple(sorted(self.elements)))
        else:
            self._cached_elements_hash = None

    def __str__(self):
        obj_val = self.objfun_val
        if obj_val is not None:
            return f"ScQbfSolution(_objfun_val={obj_val:.2f}, elements={len(self.elements)})"
        return f"ScQbfSolution(_objfun_val=None, elements={len(self.elements)})"