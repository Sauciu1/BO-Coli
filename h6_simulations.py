from  src import ax_helper
from src.ax_helper import SequentialRuns
from src.toy_functions import Hartmann6D
import numpy as np
from ax import Client, RangeParameterConfig
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement, qMaxValueEntropy
import pickle
import time


param_range = [
    RangeParameterConfig(name=f"x{i+1}", parameter_type="float", bounds=(0.0, 1.0))
    for i in range(6)
]

metric_name = "response"

dim_names = [rp.name for rp in param_range]


test_fn = Hartmann6D().eval_at



def _single_run(params, n_runs = 30):
    tr, noise = params

    def noise_fn(x):
        return x + np.random.normal(0, noise)
    
    local_tester = SequentialRuns(test_fn, param_range, dim_names, metric_name)
    runs = local_tester.run(
        SingleTaskGP,
        n_runs=n_runs,
        technical_repeats=tr,
        batch_size=1,
        noise_fn=noise_fn,
        plot_each=False,
    )
    df_local = ax_helper.get_y_data(runs, dim_names, test_fn)

    return df_local

save_dir = "data/bayes_sim/"

import multiprocessing as mp
import os

print(__name__)
mode = 'multicore'
if __name__ == "__main__":
    n_runs = 20
    mp.set_start_method("spawn", force=True)  # Windows-safe

    t0 = time.perf_counter()
    print("Starting batch Bayesian optimization tests...")

    param_grid = [(tr, float(n)) for tr in range(1, 9) for n in np.linspace(0, 2.2, 12)]
    n_workers = min(len(param_grid), os.cpu_count() or 1)


    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        results = pool.starmap(_single_run, [(el, n_runs) for el in param_grid])


    r_n_dict = dict(zip(param_grid, results))

    with open(save_dir + 'multicore_trial.pkl', 'wb') as f:
        pickle.dump(r_n_dict, f)

    elapsed = time.perf_counter() - t0
    print(f"Completed {len(param_grid)} configs in {elapsed:.2f}s with {n_workers} workers")
    r_n_dict

    # Runtime for sequential runs : 112s
    # Same code with 8 workers: 40s