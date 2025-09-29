from  src import ax_helper, model_generation
from src.ax_helper import SequentialRuns
from src.toy_functions import Hartmann6D
import numpy as np
from ax import Client, RangeParameterConfig
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement, qMaxValueEntropy
import pickle
import time
import multiprocessing as mp
import os
from src.model_generation import GammaNoiseSGP, HeteroNoiseSGP, HeteroWhiteSGP

param_range = [
    RangeParameterConfig(name=f"x{i+1}", parameter_type="float", bounds=(0.0, 1.0))
    for i in range(6)
]

metric_name = "response"

dim_names = [rp.name for rp in param_range]


test_fn = Hartmann6D().eval_at



def _single_run(params, n_runs = 30):
    tr, noise = params

    def noise_fn(x, y):
        return y + np.random.normal(0, noise)
    
    local_tester = SequentialRuns(test_fn, param_range, dim_names, metric_name)
    runs = local_tester.run(
        HeteroWhiteSGP,
        n_runs=n_runs,
        technical_repeats=tr,
        batch_size=1,
        noise_fn=noise_fn,
        plot_each=False,
    )
    df_local = ax_helper.get_y_data(runs, dim_names, test_fn)

    return df_local




print(__name__)
mode = 'multicore'



def run_grid(save_path):
    n_runs = 100
    mp.set_start_method("spawn", force=True)  # Windows-safe

    t0 = time.perf_counter()
    print("Starting batch Bayesian optimization tests...")

    param_grid = [(tr, n, f"again_{again}") for tr in range(1, 10, 2) for n in np.linspace(0, 1.1, 6) for again in range(1, 7)]
    n_workers = 64

    # Create results dict and ensure directory exists
    r_n_dict = {}
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Callback function to save after each completion
    def save_result(result, param_key):
        r_n_dict[param_key] = result
        with open(save_path, 'wb') as f:
            pickle.dump(r_n_dict, f)
        print(f"Saved result for {param_key} - Total completed: {len(r_n_dict)}/{len(param_grid)}")

    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        # Submit all jobs and collect AsyncResult objects
        jobs = []
        for param in param_grid:
            job = pool.apply_async(_single_run, args=(param[0:2], n_runs))
            jobs.append((job, param))
        
        # Collect results as they complete
        for job, param_key in jobs:
            result = job.get()  # This blocks until the specific job completes
            save_result(result, param_key)

    t1 = time.perf_counter()
    print(f"All tasks completed in {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    save_dir = "data/bayes_sim/"
    save_PATH = save_dir + 'HeteroWhite_09_29.pkl'

    run_grid(save_PATH)
    # Runtime for sequential runs : 112s
    # Same code with 8 workers: 40s