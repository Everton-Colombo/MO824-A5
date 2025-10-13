from dataclasses import dataclass
from typing import List, Set

@dataclass
class ScQbfInstance:
    """
    Container for a MAX-SC-QBF instance.

    Attributes
    ----------
    n : int
        Number of variables/subsets.
    subsets : List[Set[int]]
        List of subsets; each subset is a set of covered elements (1-based).
    A : List[List[float]]
        n x n coefficient matrix (upper-triangular part provided by the file).
    """
    n: int
    subsets: List[Set[int]]
    A: List[List[float]]

    @classmethod
    def from_file(cls, filename: str) -> 'ScQbfInstance':
        """
        Reads a MAX-SC-QBF instance from a file and returns a ScQbfInstance.
        
        Parameters
        ----------
        filename : str
            Path to the instance file.
            
        Returns
        -------
        ScQbfInstance
            The loaded instance.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]

        # Read n
        n: int = int(lines[0])

        # Read sizes of each subset
        subset_sizes: List[int] = list(map(int, lines[1].split()))
        if len(subset_sizes) != n:
            raise ValueError(f"Expected {n} subset sizes, got {len(subset_sizes)}.")

        # Read subsets
        subsets: List[Set[int]] = []
        idx = 2
        for size in subset_sizes:
            elements = set(map(int, lines[idx].split()))
            if len(elements) != size:
                raise ValueError(f"Expected {size} elements in subset, got {len(elements)}.")
            subsets.append(elements)
            idx += 1

        # Read upper triangular matrix
        A: List[List[float]] = [[0.0] * n for _ in range(n)]
        row = 0
        while idx < len(lines) and row < n:
            values = list(map(float, lines[idx].split()))
            for col, val in enumerate(values, start=row):
                if col >= n:
                    break
                A[row][col] = val
            idx += 1
            row += 1

        return cls(n=n, subsets=subsets, A=A)



