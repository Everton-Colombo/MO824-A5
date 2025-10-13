import math
from ..scqbf_instance import *
from ..scqbf_evaluator import *
import random
import time

class ScQbfGrasp:
    
    def __init__(self, instance: ScQbfInstance, max_iterations,
                 config: dict = {
                    "construction_method": "traditional",  # traditional | random_plus_greedy | sampled_greedy
                    "construction_args": (),
                    "local_search_method": "best_improve"  # best_improve | first_improve
                 }, time_limit_secs: float = None,
                 patience: int = None,
                 debug: bool = False):
        
        
        self.instance = instance
        self.max_iterations = max_iterations
        self.config = config
        self.time_limit_secs = time_limit_secs
        self.patience = patience
        self.debug = debug
        self.solve_time = 0
        self.iterations = 0
        self.evaluator = ScQbfEvaluator(instance)


    def solve(self) -> ScQbfSolution:
        if self.instance is None:
            raise ValueError("Problem instance is not initialized")
        
        best_sol = ScQbfSolution([])
        start_time = time.perf_counter()
        self.iterations = 0
        current_patience = self.patience
        while ((self.iterations < self.max_iterations) if self.max_iterations is not None else True):
            self.iterations += 1
            constructed_sol = self._constructive_heuristic()
            if self.debug:
                print(f"Constructed solution (iteration {self.iterations}): {constructed_sol.elements}")

            if not self.evaluator.is_solution_valid(constructed_sol):
                if self.debug:
                    print("Constructed solution is not feasible, fixing...")
                constructed_sol = self._fix_solution(constructed_sol)
            
            sol = self._local_search(constructed_sol)
            
            if (self.evaluator.evaluate_objfun(sol) > self.evaluator.evaluate_objfun(best_sol)):
                best_sol = sol
                current_patience = self.patience
            else:
                if current_patience is not None:
                    current_patience -= 1
                    if current_patience <= 0:
                        print(f"Patience exhausted, no improvement to the objective solution in {self.patience} iterations, stopping GRASP.")
                        break
            
            self.solve_time = time.perf_counter() - start_time
            if self.time_limit_secs is not None and self.solve_time >= self.time_limit_secs:
                print(f"Time limit of {self.time_limit_secs} seconds reached, stopping GRASP.")
                break
            
        return best_sol
    
    def _fix_solution(self, sol: ScQbfSolution) -> ScQbfSolution:
        """
        This function is called when the constructed solution is not feasible.
        It'll add the most covering elements until the solution is feasible.
        """
        while not self.evaluator.is_solution_valid(sol):
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
        
        if not self.evaluator.is_solution_valid(sol):
            raise ValueError("Could not fix the solution to be feasible")
        
        return sol

    def _constructive_heuristic(self) -> ScQbfSolution:
        if self.config["construction_method"] == "traditional":
            alpha = self.config["construction_args"][0] if len(self.config.get("construction_args", [])) > 0 else 0.5
            return self._constructive_heuristic_traditional(alpha)

        elif self.config["construction_method"] == "random_plus_greedy":
            alpha, p = self.config["construction_args"] if len(self.config.get("construction_args", [])) > 0 else (0.5, 0.2)
            return self._constructive_heuristic_random_plus_greedy(alpha, p)
        
        elif self.config["construction_method"] == "sampled_greedy":
            p = self.config["construction_args"][0] if len(self.config.get("construction_args", [])) > 0 else 0.1
            return self._constructive_heuristic_sampled_greedy(p)
        
        else:
            return self._constructive_heuristic_traditional()

    def _constructive_heuristic_traditional(self, alpha: float) -> ScQbfSolution:
        constructed_sol = ScQbfSolution([])
        cl = [i for i in range(self.instance.n)] # makeCl

        while not self.evaluator.is_solution_valid(constructed_sol): # Constructive Stop Criteria
            # traditional constructive heuristic
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
            
            # This is where we define the RCL.
            for candidate_element in cl:
                delta_objfun = self.evaluator.evaluate_insertion_delta(candidate_element, constructed_sol)
                if delta_objfun >= (min_delta + alpha * (max_delta - min_delta)):

                    ## ONLY add to rcl if coverage increases
                    if self.evaluator.evaluate_insertion_delta_coverage(candidate_element, constructed_sol) > 0:
                        rcl.append(candidate_element)

            # Randomly select an element from the RCL to add to the solution
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
            constructed_sol.elements.append(random.choice(cl))
        
        # Continue with a purely greedy approach
        while not self.evaluator.is_solution_valid(constructed_sol): # Constructive Stop Criteria
            cl = [i for i in cl if i not in constructed_sol.elements] # update_cl
            
            best_delta = float("-inf")
            best_cand_in = -1
            
            for candidate_element in cl:
                # Only consider candidates that improve coverage and objective function
                delta_objfun = self.evaluator.evaluate_insertion_delta(candidate_element, constructed_sol)
                if delta_objfun > best_delta and self.evaluator.evaluate_insertion_delta_coverage(candidate_element, constructed_sol) > 0:
                    best_cand_in = candidate_element
                    best_delta = delta_objfun
            
            if best_delta > 0:
                constructed_sol.elements.append(best_cand_in)
            else:
                break

        return constructed_sol
    
    def _constructive_heuristic_sampled_greedy(self, p: int):
        constructed_sol = ScQbfSolution([])
        cl = [i for i in range(self.instance.n)] # makeCl

        while not self.evaluator.is_solution_valid(constructed_sol): # Constructive Stop Criteria
            cl = [i for i in cl if i not in constructed_sol.elements] # update_cl
            
            rcl = random.sample(cl, min(len(cl), math.floor(p * self.instance.n)))
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
        if self.config.get("local_search_method", False) == "best_improve":
            return self._local_search_best_improve(starting_point)
        elif self.config.get("local_search_method", False) == "first_improve":
            return self._local_search_first_improve(starting_point)


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
                    if self.evaluator.is_solution_valid(temp_sol):
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
                        if self.evaluator.is_solution_valid(temp_sol):
                            best_delta = delta
                            best_cand_in = cand_in
                            best_cand_out = cand_out

            # Apply the best move if it improves the solution
            if best_delta > 0:  # Positive delta means improvement for maximization
                if self.debug:
                    print(f"[local_search]: Improvement found! Delta: {best_delta}, in {best_cand_in}, out {best_cand_out}")
                if best_cand_in is not None:
                    sol.elements.append(best_cand_in)
                if best_cand_out is not None:
                    sol.elements.remove(best_cand_out)

                self.evaluator.evaluate_objfun(sol)
            else:
                if self.debug:
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

            # Insertions
            for cand_in in cl:
                delta = self.evaluator.evaluate_insertion_delta(cand_in, sol)
                if delta > 0:
                    if self.debug:
                        print(f"[local_search]: First improvement found (insertion)! Delta: {delta}, in {cand_in}")
                    sol.elements.append(cand_in)
                    self.evaluator.evaluate_objfun(sol)
                    improvement_found = True
                    break
            
            if improvement_found:
                continue
            
            # Removals
            for cand_out in sol.elements:
                delta = self.evaluator.evaluate_removal_delta(cand_out, sol)
                if delta > 0: 
                    # Check if removing this element would break feasibility
                    temp_sol = ScQbfSolution(sol.elements.copy())
                    temp_sol.elements.remove(cand_out)
                    if self.evaluator.is_solution_valid(temp_sol):
                        if self.debug:
                            print(f"[local_search]: First improvement found (removal)! Delta: {delta}, out {cand_out}")
                        sol.elements.remove(cand_out)
                        self.evaluator.evaluate_objfun(sol)
                        improvement_found = True
            
            if improvement_found:
                continue

            # Exchanges
            for cand_in in cl:
                for cand_out in sol.elements:
                    delta = self.evaluator.evaluate_exchange_delta(cand_in, cand_out, sol)
                    if delta > 0:
                        # Check if this exchange would break feasibility
                        temp_sol = ScQbfSolution(sol.elements.copy())
                        temp_sol.elements.remove(cand_out)
                        temp_sol.elements.append(cand_in)
                        if self.evaluator.is_solution_valid(temp_sol):
                            if self.debug:
                                print(f"[local_search]: First improvement found (exchange)! Delta: {delta}, in {cand_in}, out {cand_out}")
                            sol.elements.remove(cand_out)
                            sol.elements.append(cand_in)
                            self.evaluator.evaluate_objfun(sol)
                            improvement_found = True
                            break
                
                if improvement_found:
                    break

            if improvement_found:
                continue
            if self.debug:
                print(f"[local_search]: No improvement found after ({_search_iterations}) iterations!")
            break
        
        return sol
