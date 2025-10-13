import math
import random
import time
from typing import Literal, Optional
from dataclasses import dataclass
import warnings

from ..scqbf_instance import *
from ..scqbf_evaluator import *
from ..scqbf_solution import *
from .abc_solver import SCQBF_Solver, TerminationCriteria, DebugOptions


@dataclass
class GraspStrategy:
    """
    Configuration dataclass for the GRASP algorithm.
    
    Attributes
    ----------
    construction_method : Literal['traditional', 'random_plus_greedy', 'sampled_greedy']
        The construction method to use in the constructive phase.
        - 'traditional': Uses alpha parameter for RCL construction
        - 'random_plus_greedy': Uses both alpha and p parameters
        - 'sampled_greedy': Uses only p parameter
    local_search_method : Literal['best_improve', 'first_improve']
        The local search strategy to use.
    alpha : float, optional
        Alpha parameter for traditional and random_plus_greedy construction.
        Controls the greediness/randomness tradeoff in RCL construction.
        Required for: 'traditional', 'random_plus_greedy'
        Range: [0, 1]
    p : float, optional
        Percentage parameter for random_plus_greedy and sampled_greedy.
        - For random_plus_greedy: fraction of elements to select randomly before greedy phase.
        - For sampled_greedy: fraction of candidates to sample for the RCL.
        Required for: 'random_plus_greedy', 'sampled_greedy'
        Range: (0, 1]
    """
    construction_method: Literal['traditional', 'random_plus_greedy', 'sampled_greedy'] = 'traditional'
    local_search_method: Literal['best_improve', 'first_improve'] = 'best_improve'
    alpha: Optional[float] = None
    p: Optional[float] = None
    
    def __post_init__(self):
        if self.construction_method == 'traditional':
            if self.alpha is None:
                self.alpha = 0.5
            if not (0 <= self.alpha <= 1):
                raise ValueError("Alpha must be in the range [0, 1].")
            if self.p is not None:
                warnings.warn("Parameter 'p' is not used with 'traditional' construction method and will be ignored.")
        
        elif self.construction_method == 'random_plus_greedy':
            if self.alpha is None:
                self.alpha = 0.5
            if self.p is None:
                self.p = 0.2
            
            if not (0 <= self.alpha <= 1):
                raise ValueError("Alpha must be in the range [0, 1].")
            if not (0 < self.p <= 1):
                raise ValueError("P must be in the range (0, 1].")
        
        elif self.construction_method == 'sampled_greedy':
            if self.p is None:
                self.p = 0.1
            if not (0 < self.p <= 1):
                raise ValueError("P must be in the range (0, 1].")
            if self.alpha is not None:
                warnings.warn("Parameter 'alpha' is not used with 'sampled_greedy' construction method and will be ignored.")
        
        else:
            raise ValueError(f"Unknown construction method: {self.construction_method}")


class ScQbfGrasp(SCQBF_Solver):
    
    def __init__(self, instance: ScQbfInstance, strategy: GraspStrategy = GraspStrategy(),
                 termination_criteria: TerminationCriteria = TerminationCriteria(),
                 debug_options: DebugOptions = DebugOptions()):
        
        super().__init__(instance, termination_criteria, debug_options)
        
        self.strategy = strategy
        
        if debug_options.log_history:
            self.history: list[float] = []

    def solve(self) -> ScQbfSolution:
        self._reset_execution_state()
        
        while not self._check_termination():
            self._perform_debug_actions()
            
            constructed_sol = self._constructive_heuristic()
            
            if self.debug_options.verbose:
                print(f"Constructed solution (iteration {self._iters}): {constructed_sol.elements}")

            if not self.evaluator.is_solution_feasible(constructed_sol):
                if self.debug_options.verbose:
                    print("Constructed solution is not feasible, fixing...")
                constructed_sol = self._fix_solution(constructed_sol)
            
            self._current_solution = self._local_search(constructed_sol)
            
            self._update_execution_state()
        
        self.execution_time = time.time() - self._start_time
        return self.best_solution
    
    def _perform_debug_actions(self):
        if self.debug_options.verbose:
            best_val = f'{self.evaluator.evaluate_objfun(self.best_solution):.2f}' if self.best_solution else 'N/A'
            current_val = f'{self.evaluator.evaluate_objfun(self._current_solution):.2f}' if self._current_solution else 'N/A'
            print(f"Iteration {self._iters}: Best ObjFun = {best_val}, Current ObjFun = {current_val}")

        if self.debug_options.log_history:
            self.history.append(self.evaluator.evaluate_objfun(self.best_solution) if self.best_solution else 0)
    
    def _fix_solution(self, sol: ScQbfSolution) -> ScQbfSolution:
        """
        This function is called when the constructed solution is not feasible.
        It'll add the most covering elements until the solution is feasible.
        """
        while not self.evaluator.is_solution_feasible(sol):
            cl = [i for i in range(self.instance.n) if i not in sol.elements]
            best_cand = None
            best_coverage = -1
            
            for cand in cl:
                coverage = self.evaluator.evaluate_insertion_delta_coverage(cand, sol)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_cand = cand
            
            if best_cand is not None:
                sol.elements.append(best_cand)
            else:
                break
        
        if not self.evaluator.is_solution_feasible(sol):
            raise ValueError("Could not fix the solution to be feasible")
        
        return sol

    def _constructive_heuristic(self) -> ScQbfSolution:
        if self.strategy.construction_method == "traditional":
            return self._constructive_heuristic_traditional(self.strategy.alpha)
        elif self.strategy.construction_method == "random_plus_greedy":
            return self._constructive_heuristic_random_plus_greedy(self.strategy.alpha, self.strategy.p)
        elif self.strategy.construction_method == "sampled_greedy":
            return self._constructive_heuristic_sampled_greedy(self.strategy.p)
        else:
            raise ValueError(f"Unknown construction method: {self.strategy.construction_method}")

    def _constructive_heuristic_traditional(self, alpha: float) -> ScQbfSolution:
        constructed_sol = ScQbfSolution([])
        cl = [i for i in range(self.instance.n)] # makeCl

        while not self.evaluator.is_solution_feasible(constructed_sol): # Constructive Stop Criteria
            rcl = []
            min_delta = math.inf
            max_delta = -math.inf
            cl = [i for i in cl if i not in constructed_sol.elements] # update_cl
            
            for candidate_element in cl:
                delta_objfun = self.evaluator.evaluate_insertion_delta(candidate_element, constructed_sol)
                if delta_objfun < min_delta:
                    min_delta = delta_objfun
                if delta_objfun > max_delta:
                    max_delta = delta_objfun
            
            for candidate_element in cl:
                delta_objfun = self.evaluator.evaluate_insertion_delta(candidate_element, constructed_sol)
                if delta_objfun >= (min_delta + alpha * (max_delta - min_delta)):
                    ## ONLY add to rcl if coverage increases
                    if self.evaluator.evaluate_insertion_delta_coverage(candidate_element, constructed_sol) > 0:
                        rcl.append(candidate_element)

            if rcl:
                chosen_element = random.choice(rcl)
                constructed_sol.elements.append(chosen_element)
            else:
                break

        return constructed_sol

    def _constructive_heuristic_random_plus_greedy(self, alpha: float, p: float):
        constructed_sol = ScQbfSolution([])
        cl = [i for i in range(self.instance.n)] # make_cl

        # Select first p elements at random
        for _ in range(int(p * self.instance.n)):
            cl = [i for i in cl if i not in constructed_sol.elements] # update_cl
            if cl:
                constructed_sol.elements.append(random.choice(cl))
        
        # Continue with a purely greedy approach
        while not self.evaluator.is_solution_feasible(constructed_sol): # Constructive Stop Criteria
            cl = [i for i in cl if i not in constructed_sol.elements] # update_cl
            
            best_delta = float("-inf")
            best_cand_in = None
            
            for candidate_element in cl:
                delta_objfun = self.evaluator.evaluate_insertion_delta(candidate_element, constructed_sol)
                if delta_objfun > best_delta and self.evaluator.evaluate_insertion_delta_coverage(candidate_element, constructed_sol) > 0:
                    best_cand_in = candidate_element
                    best_delta = delta_objfun
            
            if best_delta > 0 and best_cand_in is not None:
                constructed_sol.elements.append(best_cand_in)
            else:
                break

        return constructed_sol
    
    def _constructive_heuristic_sampled_greedy(self, p: float) -> ScQbfSolution:
        constructed_sol = ScQbfSolution([])
        cl = [i for i in range(self.instance.n)] # makeCl

        while not self.evaluator.is_solution_feasible(constructed_sol): # Constructive Stop Criteria
            cl = [i for i in cl if i not in constructed_sol.elements] # update_cl
            
            rcl = random.sample(cl, min(len(cl), max(1, math.floor(p * self.instance.n))))
            best_delta = float("-inf")
            best_cand_in = None
            for candidate_element in rcl:
                delta = self.evaluator.evaluate_insertion_delta(candidate_element, constructed_sol)
                if delta > best_delta:
                    best_delta = delta
                    best_cand_in = candidate_element
            
            if best_delta > 0 and best_cand_in is not None:
                constructed_sol.elements.append(best_cand_in)
            else:
                break
        
        return constructed_sol

    ####################

    def _local_search(self, starting_point: ScQbfSolution) -> ScQbfSolution:
        if self.strategy.local_search_method == "best_improve":
            return self._local_search_best_improve(starting_point)
        elif self.strategy.local_search_method == "first_improve":
            return self._local_search_first_improve(starting_point)
        else:
            raise ValueError(f"Unknown local search method: {self.strategy.local_search_method}")

    def _local_search_best_improve(self, starting_point: ScQbfSolution) -> ScQbfSolution:
        sol = ScQbfSolution(starting_point.elements.copy())
        
        _search_iterations = 0
        
        while True:
            _search_iterations += 1
            
            best_delta = float("-inf")
            best_cand_in = None
            best_cand_out = None

            cl = [i for i in range(self.instance.n) if i not in sol.elements]

            # Evaluate insertions
            for cand_in in cl:
                delta = self.evaluator.evaluate_insertion_delta(cand_in, sol)
                if delta > best_delta:
                    best_delta = delta
                    best_cand_in = cand_in
                    best_cand_out = None

            # Evaluate removals
            for cand_out in sol.elements:
                delta = self.evaluator.evaluate_removal_delta(cand_out, sol)
                if delta > best_delta:
                    # Check if removing this element would break feasibility
                    temp_sol = ScQbfSolution(sol.elements.copy())
                    temp_sol.elements.remove(cand_out)
                    if self.evaluator.is_solution_feasible(temp_sol):
                        best_delta = delta
                        best_cand_in = None
                        best_cand_out = cand_out

            # Evaluate exchanges
            for cand_in in cl:
                for cand_out in sol.elements:
                    delta = self.evaluator.evaluate_exchange_delta(cand_in, cand_out, sol)
                    if delta > best_delta:
                        # Check if this exchange would break feasibility
                        temp_sol = ScQbfSolution(sol.elements.copy())
                        temp_sol.elements.remove(cand_out)
                        temp_sol.elements.append(cand_in)
                        if self.evaluator.is_solution_feasible(temp_sol):
                            best_delta = delta
                            best_cand_in = cand_in
                            best_cand_out = cand_out

            # Apply the best move if it improves the solution
            if best_delta > 0:  # Positive delta means improvement for maximization
                if self.debug_options.verbose:
                    print(f"[local_search]: Improvement found! Delta: {best_delta}, in {best_cand_in}, out {best_cand_out}")
                
                if best_cand_in is not None:
                    sol.elements.append(best_cand_in)
                if best_cand_out is not None:
                    sol.elements.remove(best_cand_out)

                self.evaluator.evaluate_objfun(sol)
            else:
                if self.debug_options.verbose:
                    print(f"[local_search]: No improvement found after ({_search_iterations}) iterations!")
                break  # No improving move found
        
        return sol

    def _local_search_first_improve(self, starting_point: ScQbfSolution) -> ScQbfSolution:
        sol = ScQbfSolution(starting_point.elements.copy())
        _search_iterations = 0
        
        while True:
            _search_iterations += 1
            improvement_found = False
            
            cl = [i for i in range(self.instance.n) if i not in sol.elements]

            # Evaluate insertions
            for cand_in in cl:
                delta = self.evaluator.evaluate_insertion_delta(cand_in, sol)
                if delta > 0:
                    if self.debug_options.verbose:
                        print(f"[local_search]: First improvement found (insertion)! Delta: {delta}, in {cand_in}")
                    sol.elements.append(cand_in)
                    self.evaluator.evaluate_objfun(sol)
                    improvement_found = True
                    break
            
            if improvement_found:
                continue
            
            # Evaluate removals
            for cand_out in sol.elements:
                delta = self.evaluator.evaluate_removal_delta(cand_out, sol)
                if delta > 0: 
                    # Check if removing this element would break feasibility
                    temp_sol = ScQbfSolution(sol.elements.copy())
                    temp_sol.elements.remove(cand_out)
                    if self.evaluator.is_solution_feasible(temp_sol):
                        if self.debug_options.verbose:
                            print(f"[local_search]: First improvement found (removal)! Delta: {delta}, out {cand_out}")
                        sol.elements.remove(cand_out)
                        self.evaluator.evaluate_objfun(sol)
                        improvement_found = True
                        break
            
            if improvement_found:
                continue

            # Evaluate exchanges
            for cand_in in cl:
                for cand_out in sol.elements:
                    delta = self.evaluator.evaluate_exchange_delta(cand_in, cand_out, sol)
                    if delta > 0:
                        # Check if this exchange would break feasibility
                        temp_sol = ScQbfSolution(sol.elements.copy())
                        temp_sol.elements.remove(cand_out)
                        temp_sol.elements.append(cand_in)
                        if self.evaluator.is_solution_feasible(temp_sol):
                            if self.debug_options.verbose:
                                print(f"[local_search]: First improvement found (exchange)! Delta: {delta}, in {cand_in}, out {cand_out}")
                            sol.elements.remove(cand_out)
                            sol.elements.append(cand_in)
                            self.evaluator.evaluate_objfun(sol)
                            improvement_found = True
                            break
                
                if improvement_found:
                    break

            if not improvement_found:
                if self.debug_options.verbose:
                    print(f"[local_search]: No improvement found after ({_search_iterations}) iterations!")
                break
        
        return sol
