import time
import warnings
from dataclasses import dataclass
from typing import Optional

import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, RangeSet
from pyomo.opt import SolverFactory

from ..scqbf_instance import *
from ..scqbf_evaluator import *
from ..scqbf_solution import *
from .abc_solver import SCQBF_Solver, TerminationCriteria, DebugOptions


@dataclass
class GurobiConfig:
    """
    Configuration dataclass for the Gurobi solver.

    Attributes
    ----------
    mip_gap : float, optional
        Relative MIP gap tolerance for the solver.
    tee : bool
        Whether to display solver output.
    """
    mip_gap: Optional[float] = None
    tee: bool = False


class ScQbfGurobi(SCQBF_Solver):

    def __init__(self, instance: ScQbfInstance, gurobi_config: GurobiConfig = GurobiConfig(),
                 termination_criteria: TerminationCriteria = TerminationCriteria(),
                 debug_options: DebugOptions = DebugOptions()):
        
        super().__init__(instance, termination_criteria, debug_options)

        self.config = gurobi_config
        self.model: Optional[pyo.Model] = None
        self.solver_results: Optional[dict] = None
        
        # Gurobi-specific results
        self.primal_bound: Optional[float] = None
        self.dual_bound: Optional[float] = None
        self.relative_gap: Optional[float] = None
        self.absolute_gap: Optional[float] = None
        self.solver_runtime: Optional[float] = None
        
        # Validate termination criteria
        self._validate_termination_criteria()
    
    def _validate_termination_criteria(self):
        """Validate that termination criteria are compatible with Gurobi solver."""
        if self.termination_criteria.max_iterations is not None:
            warnings.warn(
                "Gurobi is an exact solver and does not use iterations like metaheuristics. "
                "'max_iterations' will be ignored.",
                UserWarning
            )
        
        if self.termination_criteria.max_no_improvement is not None:
            warnings.warn(
                "Gurobi is an exact solver and does not track improvement iterations. "
                "'max_no_improvement' will be ignored.",
                UserWarning
            )
        
        # target_value and max_time_secs are valid for Gurobi
        # target_value maps to BestObjStop, max_time_secs is the time limit
    
    def solve(self) -> ScQbfSolution:
        """Solve the problem using Gurobi."""
        self._reset_execution_state()
        
        # Create the Gurobi model
        if self.debug_options.verbose:
            print("Creating Gurobi model...")
        self.model = self._create_model()
        
        # Solve the model
        if self.debug_options.verbose:
            print("Solving Gurobi model...")
        self._solve_model()
        
        # Extract solution
        self.best_solution = self._extract_solution()
        self._current_solution = self.best_solution
        
        self.execution_time = time.time() - self._start_time
        
        # Set appropriate stop reason
        if self.solver_runtime is not None and self.termination_criteria.max_time_secs is not None:
            if self.solver_runtime >= self.termination_criteria.max_time_secs - 0.1:  # Small tolerance for timing
                self.stop_reason = "max_time_secs"
        
        if self.termination_criteria.target_value is not None and self.primal_bound is not None:
            if self.primal_bound >= self.termination_criteria.target_value:
                self.stop_reason = "target_value"
        
        if self.debug_options.verbose:
            self._perform_debug_actions()
        
        return self.best_solution
    
    def _perform_debug_actions(self):
        """Perform debug actions, such as logging or printing debug information."""
        if self.debug_options.verbose:
            print(f"Gurobi Solution:")
            print(f"  Objective: {self.evaluator.evaluate_objfun(self.best_solution):.2f}")
            print(f"  Selected elements: {self.best_solution.elements}")
            print(f"  Primal bound: {self.primal_bound}")
            print(f"  Dual bound: {self.dual_bound}")
            if self.relative_gap is not None:
                print(f"  Relative gap: {self.relative_gap:.4%}")
            if self.absolute_gap is not None:
                print(f"  Absolute gap: {self.absolute_gap:.2f}")
            print(f"  Solver runtime: {self.solver_runtime:.2f}s")
            print(f"  Total execution time: {self.execution_time:.2f}s")
            if self.stop_reason:
                print(f"  Stop reason: {self.stop_reason}")
    
    def _create_model(self) -> pyo.Model:
        """Create the Pyomo model from the instance."""
        n = self.instance.n
        sets = self.instance.subsets
        A = self.instance.A
        
        model = ConcreteModel()

        ## Vars:
        model.I = RangeSet(n)
        model.x = Var(model.I, domain=pyo.Binary)

        model.IJs = pyo.Set(dimen=2, initialize=[(i, j) for i in model.I for j in model.I if i <= j]) # Only take elements on the diagonal and above
        model.y = Var(model.IJs, domain=pyo.Binary)

        ## Objective:
        def objective_rule(model):
            return sum(A[i-1][j-1] * model.y[i, j] for (i, j) in model.IJs)

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        ## Constraints:

        # Linearization contraints (definition of y_{i, j})
        model.y_constraints = pyo.ConstraintList()
        for (i, j) in model.IJs:
            model.y_constraints.add(expr=model.y[i, j] <= model.x[i])                   # 1
            model.y_constraints.add(expr=model.y[i, j] <= model.x[j])                   # 2
            model.y_constraints.add(expr=model.y[i, j] >= model.x[i] + model.x[j] - 1)  # 3

        # Set-cover contraints:
        model.set_cover_contraints = pyo.ConstraintList()
        universe = set.union(*[s for s in sets if s])
        
        for element in universe:
            # Get all indexes from the sets that contain element:
            containing_sets_idx: List[int] = []
            for i, curr_set in enumerate(sets):
                if element in curr_set:
                    containing_sets_idx.append(i)
            
            model.set_cover_contraints.add(expr=sum(model.x[i+1] for i in containing_sets_idx) >= 1)
        
        return model
    
    def _solve_model(self):
        """Solve the model and store results."""
        opt = SolverFactory('gurobi_persistent') # Model persists in memory after solve, intead of being discarded. Allows for accessing properties after solve.
        opt.set_instance(self.model)
        
        # Configure solver options
        solver_options = {}
        
        # Use time limit from termination_criteria if available
        if self.termination_criteria.max_time_secs is not None:
            solver_options['TimeLimit'] = self.termination_criteria.max_time_secs
        
        # Set BestObjStop (target value) if specified
        if self.termination_criteria.target_value is not None:
            solver_options['BestObjStop'] = self.termination_criteria.target_value
        
        if self.config.mip_gap is not None:
            solver_options['MIPGap'] = self.config.mip_gap
        
        model_results = opt.solve(
            tee=self.config.tee,
            options=solver_options
        )
        
        solcount = opt.get_model_attr('SolCount')
        self.solver_runtime = opt.get_model_attr('Runtime')  
        self.dual_bound = opt.get_model_attr('ObjBound')                         # dual (best bound)
        self.primal_bound = opt.get_model_attr('ObjVal') if solcount > 0 else None  # primal (incumbent)
        self.relative_gap = opt.get_model_attr('MIPGap') if solcount > 0 else None

        # Calculate absolute gap
        self.absolute_gap = abs(self.primal_bound - self.dual_bound) if self.primal_bound is not None else None
        
        # Store full results
        self.solver_results = {
            "objective": pyo.value(self.model.obj) if solcount > 0 else None,
            "variables": [pyo.value(self.model.x[i]) for i in self.model.I] if solcount > 0 else None,
            "primal": self.primal_bound,
            "dual": self.dual_bound,
            "relative_gap": self.relative_gap,
            "absolute_gap": self.absolute_gap,
            "solver_runtime_sec": self.solver_runtime,
            "model_results": model_results,
        }
    
    def _extract_solution(self) -> ScQbfSolution:
        """Extract the solution from the solved model."""
        variables = [pyo.value(self.model.x[i]) for i in self.model.I]
        
        # Extract indices where x[i] == 1 (selected sets, 0-based)
        selected_elements = [i - 1 for i, v in enumerate(variables, start=1) if v == 1]
        
        solution = ScQbfSolution(selected_elements)
        self.evaluator.evaluate_objfun(solution)
        
        return solution