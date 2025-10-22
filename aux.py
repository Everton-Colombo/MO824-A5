from scqbf import *
from scqbf.solvers import *

def process_single_run(args):
    """Helper function to process a single run of a heuristic"""
    heuristic_class, instance, tc, seed = args
    random.seed(seed)
    heuristic = heuristic_class(instance, termination_criteria=tc)
    heuristic.solve()
    return {
        'time': heuristic.execution_time,
        'objfun_val': heuristic.best_solution.objfun_val
    }
    
