import math
import numpy as np
from ..scqbf_instance import *
from ..scqbf_evaluator import *
from ..scqbf_solution import *
from .abc_solver import SCQBF_Solver, TerminationCriteria, DebugOptions
import random
import time
from collections import deque
from typing import Literal
from dataclasses import dataclass

PLACE_HOLDER = -1

class RestartIntensificationComponent():
    def __init__(self, instance: ScQbfInstance = None, restart_patience: int = 100, max_fixed_elements: int = 3):
        self._instance = instance
        
        self.recency_memory: List[int] = None
        self.restart_patience = restart_patience
        self.max_fixed_elements = max_fixed_elements
    
    def set_instance(self, instance: ScQbfInstance):
        self._instance = instance
        self.recency_memory = [0] * instance.n

    def update_recency_memory(self, best_solution: ScQbfSolution):
        elements_in_solution = set(best_solution.elements)
        elements_not_in_solution = set(range(self._instance.n)) - elements_in_solution
        
        for element in elements_in_solution:
            self.recency_memory[element] += 1
                
        for element in elements_not_in_solution:
            self.recency_memory[element] = 0
        
    def get_attractive_elements(self) -> List[int]:
        # return a list of the most recurring elements (up to max_fixed_elements) that dont have a zero value in recency_memory
        
        sorted_elements = sorted(range(self._instance.n), key=lambda x: self.recency_memory[x], reverse=True)
        return [element for element in sorted_elements if self.recency_memory[element] > 0][:self.max_fixed_elements]


@dataclass
class TSStrategy():
    """
    Configuration data class for the Tabu Search algorithm.
    """
    search_strategy: Literal['first', 'best'] = 'first'
    probabilistic_ts: bool = False
    probabilistic_param: float = 0.8
    ibr_component: RestartIntensificationComponent = None
    
    def __post_init__(self):
        if self.probabilistic_ts and not (0 < self.probabilistic_param < 1):
            raise ValueError("Probabilistic parameter must be in the range (0, 1) when probabilistic TS is enabled.")


class ScQbfTS(SCQBF_Solver):
    
    def __init__(self, instance: ScQbfInstance, tenure: int = 7, strategy: TSStrategy = TSStrategy(), 
                 termination_criteria: TerminationCriteria = TerminationCriteria(),
                 debug_options: DebugOptions = DebugOptions()):
        
        # Initialize parent class
        super().__init__(instance, termination_criteria, debug_options)
        
        # TS-specific properties
        self.strategy = strategy
        self.tabu_list = deque([PLACE_HOLDER] * tenure * 2, maxlen=tenure * 2)
        self._fixed_elements: List[int] = []
        
        # Initialize IBR component if present
        if self.strategy.ibr_component is not None:
            self.strategy.ibr_component.set_instance(instance)
        
        # For debug history tracking
        if debug_options.log_history:
            self.history: List[tuple] = []

    def solve(self) -> ScQbfSolution:
        """Main method to solve the problem using Tabu Search."""
        self._reset_execution_state()
        
        # Initialize with constructive heuristic
        self.best_solution = self._constructive_heuristic()
        self._current_solution = self.best_solution
        
        while not self._check_termination():
            self._perform_debug_actions()
            
            # Check for restart with intensification
            if (self.strategy.ibr_component is not None and 
                (self._iters_wo_improvement + 1) % self.strategy.ibr_component.restart_patience == 0):
                self._fixed_elements = self.strategy.ibr_component.get_attractive_elements()
                self._current_solution = ScQbfSolution(self.best_solution.elements.copy())
                
                if self.debug_options.verbose:
                    print(f"Restarting with intensification at iteration {self._iters}. Fixed elements: {self._fixed_elements}.")
            
            # Perform neighborhood move
            self._current_solution = self._neighborhood_move(self._current_solution)
            
            # Update execution state (handles best solution tracking)
            self._update_execution_state()
            
            # Update IBR memory if enabled
            if self.strategy.ibr_component is not None:
                self.strategy.ibr_component.update_recency_memory(self.best_solution)
        
        self.execution_time = time.time() - self._start_time
        return self.best_solution
    
    def _perform_debug_actions(self):
        """Perform debug actions, such as logging or printing debug information."""
        if self.debug_options.verbose:
            best_val = f'{self.evaluator.evaluate_objfun(self.best_solution):.2f}' if self.best_solution else 'N/A'
            current_val = f'{self.evaluator.evaluate_objfun(self._current_solution):.2f}' if self._current_solution else 'N/A'
            print(f"Iteration {self._iters}: Best ObjFun = {best_val}, Current ObjFun = {current_val}")

        if self.debug_options.log_history:
            self.history.append((
                self._iters, 
                self.evaluator.evaluate_objfun(self.best_solution) if self.best_solution else 0,
                self.evaluator.evaluate_objfun(self._current_solution) if self._current_solution else 0
            ))
    
    def _constructive_heuristic(self) -> ScQbfSolution:
        """
        Very simple constructive heuristic. Adds elements that add coverage until solution is feasible.
        """
        constructed_sol = ScQbfSolution([])
        cl = [i for i in range(self.instance.n)]
        random.shuffle(cl)
        
        while not self.evaluator.is_solution_feasible(constructed_sol):
            rcl = [i for i in cl if i not in constructed_sol.elements 
                      and self.evaluator.evaluate_insertion_delta_coverage(i, constructed_sol) > 0]

            element = random.choice(rcl)
            rcl.remove(element)
            constructed_sol.elements.append(element)
            
            cl = rcl

        if not self.evaluator.is_solution_feasible(constructed_sol):
            raise ValueError("Constructive heuristic failed to produce a feasible solution")
        
        return constructed_sol

    def _neighborhood_move(self, solution: ScQbfSolution) -> ScQbfSolution:
        if self.strategy.search_strategy == 'first':
            return self._neighborhood_move_first_improving(solution)
        elif self.strategy.search_strategy == 'best':
            return self._neighborhood_move_best_improving(solution)
        else:
            raise ValueError(f"Unknown search strategy: {self.strategy.search_strategy}")

    def _neighborhood_move_best_improving(self, solution: ScQbfSolution) -> ScQbfSolution:
        best_delta = float('-inf')
        best_cand_in = None
        best_cand_out = None
        
        cl = [i for i in range(self.instance.n) if i not in solution.elements]
        random.shuffle(cl)
        if self.strategy.probabilistic_ts:
            cl = random.sample(cl, min(len(cl), max(1, int(len(cl) * self.strategy.probabilistic_param))))

        current_objfun_val = self.evaluator.evaluate_objfun(solution)
        best_objfun_val = self.evaluator.evaluate_objfun(self.best_solution)
        
        # Evaluate insertions
        for cand_in in cl:
            delta = self.evaluator.evaluate_insertion_delta(cand_in, solution) 
            
            aspiration_criterion = current_objfun_val + delta > best_objfun_val
            if cand_in not in self.tabu_list or aspiration_criterion:
                if delta > best_delta:
                    best_delta = delta
                    best_cand_in = cand_in
                    best_cand_out = None
        
        # Evaluate removals
        for cand_out in solution.elements:
            delta = self.evaluator.evaluate_removal_delta(cand_out, solution)  
            
            aspiration_criterion = current_objfun_val + delta > best_objfun_val
            if ((cand_out not in self.tabu_list and cand_out not in self._fixed_elements) or 
                aspiration_criterion):
                if delta > best_delta:
                    # Check if removing this element would break feasibility
                    temp_sol = ScQbfSolution(solution.elements.copy())
                    temp_sol.elements.remove(cand_out)
                    if self.evaluator.is_solution_feasible(temp_sol):
                        best_delta = delta
                        best_cand_in = None
                        best_cand_out = cand_out
        
        # Evaluate exchanges
        for cand_in in cl:
            for cand_out in solution.elements:
                delta = self.evaluator.evaluate_exchange_delta(cand_in, cand_out, solution)  
                
                aspiration_criterion = current_objfun_val + delta > best_objfun_val
                if ((cand_in not in self.tabu_list and cand_out not in self.tabu_list and 
                     cand_out not in self._fixed_elements) or aspiration_criterion):
                    if delta > best_delta:
                        # Check if removing this element would break feasibility
                        temp_sol = ScQbfSolution(solution.elements.copy())
                        temp_sol.elements.append(cand_in)
                        temp_sol.elements.remove(cand_out)
                        if self.evaluator.is_solution_feasible(temp_sol):
                            best_delta = delta
                            best_cand_in = cand_in
                            best_cand_out = cand_out
        
        new_solution = ScQbfSolution(solution.elements.copy())
        
        # Implement the best move and update tabu list
        if best_cand_out is not None:
            new_solution.elements.remove(best_cand_out)
            self.tabu_list.append(best_cand_out)
        else:
            self.tabu_list.append(PLACE_HOLDER)
        
        if best_cand_in is not None:
            new_solution.elements.append(best_cand_in)
            self.tabu_list.append(best_cand_in)
        else:
            self.tabu_list.append(PLACE_HOLDER)
        
        return new_solution

    def _neighborhood_move_first_improving(self, solution: ScQbfSolution) -> ScQbfSolution:
        selected_cand_in = None
        selected_cand_out = None
        
        cl = [i for i in range(self.instance.n) if i not in solution.elements]
        random.shuffle(cl)
        if self.strategy.probabilistic_ts:
            cl = random.sample(cl, min(len(cl), max(1, int(len(cl) * self.strategy.probabilistic_param))))
        
        current_objfun_val = self.evaluator.evaluate_objfun(solution)
        best_objfun_val = self.evaluator.evaluate_objfun(self.best_solution)
        
        improvement_found = False
        
        def evaluate_insertions():
            nonlocal selected_cand_in, selected_cand_out, improvement_found
            
            for cand_in in cl:
                delta = self.evaluator.evaluate_insertion_delta(cand_in, solution) 
                
                aspiration_criterion = current_objfun_val + delta > best_objfun_val
                if cand_in not in self.tabu_list or aspiration_criterion:
                    if delta > 0:
                        selected_cand_in = cand_in
                        selected_cand_out = None
                        improvement_found = True
                        return
        
        def evaluate_removals():
            nonlocal selected_cand_in, selected_cand_out, improvement_found
            
            for cand_out in solution.elements:
                delta = self.evaluator.evaluate_removal_delta(cand_out, solution)  
                
                aspiration_criterion = current_objfun_val + delta > best_objfun_val
                if ((cand_out not in self.tabu_list and cand_out not in self._fixed_elements) or 
                    aspiration_criterion):
                    if delta > 0:
                        # Check if removing this element would break feasibility
                        temp_sol = ScQbfSolution(solution.elements.copy())
                        temp_sol.elements.remove(cand_out)
                        if self.evaluator.is_solution_feasible(temp_sol):
                            selected_cand_in = None
                            selected_cand_out = cand_out
                            improvement_found = True
                            return
        
        def evaluate_exchanges():
            nonlocal selected_cand_in, selected_cand_out, improvement_found
            
            for cand_in in cl:
                for cand_out in solution.elements:
                    delta = self.evaluator.evaluate_exchange_delta(cand_in, cand_out, solution)  
                    
                    aspiration_criterion = current_objfun_val + delta > best_objfun_val
                    if ((cand_in not in self.tabu_list and cand_out not in self.tabu_list and 
                         cand_out not in self._fixed_elements) or aspiration_criterion):
                        if delta > 0:
                            # Check if removing this element would break feasibility
                            temp_sol = ScQbfSolution(solution.elements.copy())
                            temp_sol.elements.append(cand_in)
                            temp_sol.elements.remove(cand_out)
                            if self.evaluator.is_solution_feasible(temp_sol):
                                selected_cand_in = cand_in
                                selected_cand_out = cand_out
                                improvement_found = True
                                return
        
        actions = [evaluate_insertions, evaluate_removals, evaluate_exchanges]
        random.shuffle(actions)
        for next_action in actions:
            if not improvement_found:
                next_action()
            else:
                if self.debug_options.verbose:
                    print(f"First improving move found with action {next_action.__name__}")
                break
        
        new_solution = ScQbfSolution(solution.elements.copy())
        
        # Implement the best move and update tabu list
        if selected_cand_out is not None:
            new_solution.elements.remove(selected_cand_out)
            self.tabu_list.append(selected_cand_out)
        else:
            self.tabu_list.append(PLACE_HOLDER)
        
        if selected_cand_in is not None:
            new_solution.elements.append(selected_cand_in)
            self.tabu_list.append(selected_cand_in)
        else:
            self.tabu_list.append(PLACE_HOLDER)
        
        return new_solution

