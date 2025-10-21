from scqbf import *
from scqbf.solvers import *

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed

pickle_dir = Path("pickles/tttplots")
pickle_dir.mkdir(parents=True, exist_ok=True)

def process_instance(path, i, j):
    """Process a single instance and save results to pickle."""
    pickle_path = pickle_dir / f"gen{i}_instance{j}.pkl"
    
    if pickle_path.exists():
        return f"Skipping gen{i}/instance{j} - results already exist"
    
    print(f"Processing gen{i}/instance{j}...")
    instance = ScQbfInstance.from_file(path)
    
    # TODO: change `max_time_secs` to 30*60
    termination_criteria = TerminationCriteria(max_time_secs=5)  # 30 minutes per instance
    # gurobi_solver: ScQbfGurobi = ScQbfGurobi(instance, termination_criteria=termination_criteria)
    results = {}

    heuristics: list[SCQBF_Solver] = [
        ScQbfGrasp(instance, termination_criteria=termination_criteria),
        ScQbfTS(instance, termination_criteria=termination_criteria),
        ScQbfGA(instance, termination_criteria=termination_criteria),
    ]

    for solver in heuristics:
        result = solver.solve()
        results[solver.__class__.__name__] = {
            "result": result,
            "time": solver.execution_time,
        }

    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    return f"Saved results for gen{i}/instance{j}"
    

def ecdf_success(times: np.ndarray):
    x = np.sort(times)
    if len(x) == 0:
        return x, np.array([])
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            if hasattr(cur, k):
                cur = getattr(cur, k)
            else:
                return default
    return cur
