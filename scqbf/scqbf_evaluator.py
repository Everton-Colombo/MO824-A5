from .scqbf_instance import *
from .scqbf_solution import *

class ScQbfEvaluator:
    
    def __init__(self, problem_instance: ScQbfInstance):
        self.problem_instance = problem_instance
        
    def evaluate_objfun(self, solution: ScQbfSolution) -> float:
        if self.problem_instance is None:
            raise ValueError("Problem instance is not initialized")
        
        if solution._objfun_val != None:
            return solution._objfun_val
        
        A = self.problem_instance.A
        total = 0.0

        # Calculate QBF value directly from solution indices
        for i in solution.elements:
            for j in solution.elements:
                total += A[i][j]

        solution._objfun_val = total
        return total

    def _evaluate_element_contribution(self, elem: int, solution: ScQbfSolution) -> float:
        """
        Calculate the contribution of an element given a list of other elements it interacts with.
        This includes both the diagonal term and interactions with other elements.
        """
        A = self.problem_instance.A
        total = 0.0
        
        # Add interactions with other elements
        for j in solution.elements:
            if j != elem:  # Avoid self-interaction in the loop
                total += A[elem][j] + A[j][elem]
        
        # Add diagonal element contribution
        total += A[elem][elem]
        
        return total

    def evaluate_insertion_delta(self, elem: int, solution: ScQbfSolution) -> float:
        if self.problem_instance is None:
            raise ValueError("Problem instance is not initialized")

        if elem in solution.elements:
            return 0.0
        
        return self._evaluate_element_contribution(elem, solution)

    def evaluate_removal_delta(self, elem: int, solution: ScQbfSolution) -> float:
        if self.problem_instance is None:
            raise ValueError("Problem instance is not initialized")

        if elem not in solution.elements:
            return 0.0
        
        return -self._evaluate_element_contribution(elem, solution)

    
    def evaluate_exchange_delta(self, elem_in: int, elem_out: int, solution: ScQbfSolution) -> float:
        if self.problem_instance is None:
            raise ValueError("Problem instance is not initialized")

        if elem_in == elem_out:
            return 0.0
        
        if elem_in in solution.elements:
            return self.evaluate_removal_delta(elem_out, solution)
        
        if elem_out not in solution.elements:
            return self.evaluate_insertion_delta(elem_in, solution)
        
        A = self.problem_instance.A
        total = 0.0

        total += self._evaluate_element_contribution(elem_in, solution)
        total -= self._evaluate_element_contribution(elem_out, solution)

        # Subtract interaction between elem_in and elem_out
        total -= (A[elem_in][elem_out] + A[elem_out][elem_in])
        
        return total
    
    def evaluate_coverage(self, solution: ScQbfSolution) -> float:
        if self.problem_instance is None:
            raise ValueError("Problem instance is not initialized")

        # Find the maximum element across all subsets to determine domain size
        domain_size = self.problem_instance.n
        
        covered = [False] * (domain_size)
        covered_count = 0

        for idx in solution.elements:
            for elem in self.problem_instance.subsets[idx]:
                if not covered[elem-1]:
                    covered[elem-1] = True
                    covered_count += 1
        
        return covered_count / domain_size

    def evaluate_insertion_delta_coverage(self, elem: int, solution: ScQbfSolution) -> float:
        if self.problem_instance is None:
            raise ValueError("Problem instance is not initialized")

        if elem in solution.elements:
            return 0.0

        # Calculate coverage delta more efficiently
        domain_size = self.problem_instance.n
        covered = [False] * domain_size
        
        # Mark elements already covered by current solution
        for idx in solution.elements:
            for element in self.problem_instance.subsets[idx]:
                covered[element-1] = True
        
        # Count new elements that would be covered by adding elem
        new_covered_count = 0
        for element in self.problem_instance.subsets[elem]:
            if not covered[element-1]:
                new_covered_count += 1
        
        return new_covered_count / domain_size

    def is_solution_feasible(self, solution: ScQbfSolution):
        return self.evaluate_coverage(solution) == 1.0