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


def _single_run(params):

    def noise_fn(x, y):
        return y + np.random.normal(0, params["noise"])
    
    local_tester = SequentialRuns(Hartmann6D().eval_at, param_range, dim_names, metric_name)

    runs = local_tester.run(
        HeteroWhiteSGP,
        n_runs=params["cycles"],
        technical_repeats=params["technical_repeats"],
        batch_size=params["batches"],
        noise_fn=noise_fn,
        plot_each=False,
    )
    df_local = runs.get_batch_observations()

    return df_local




print(__name__)
mode = 'multicore'

def run_grid(save_path):
    #param_grid = [(tr, n, f"again_{again}") for tr in range(1, 10, 2) for n in np.linspace(0, 1.1, 6) for again in range(1, 7)]

    param_grid = [
        {"technical_repeats":technical_repeats, "noise":noise, "cycles":cycles, "batches":batch}
        for technical_repeats in [1, 2, 4, 8]
        for noise in [3.4* x for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]
        for batch in [1]
        for cycles in [80]
        for rerun in range(10)
    ]
    

    t0 = time.perf_counter()
    print("Starting batch Bayesian optimization tests...")

    
    n_workers = 64

    r_n_dict = {}

    def save_result(result, param_key):
        r_n_dict[param_key] = result
        with open(save_path, 'wb') as f:
            pickle.dump(r_n_dict, f)
        print(f"Saved result for {param_key} - Total completed: {len(r_n_dict)}/{len(param_grid)}")

    mp.set_start_method("spawn", force=True)  # Windows-safe

    def dict_key(d: dict) -> str:
        # Create a stable, hashable, human-readable key
        return "|".join(f"{k}={d[k]}" for k in sorted(d))

    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        jobs = []
        for param in param_grid:
            key = dict_key(param)
            job = pool.apply_async(_single_run, args=[param])
            jobs.append((job, key, param))

        for job, param_key, original_param in jobs:
            result = job.get()
            # Optionally store original param dict alongside result
            save_result(result, param_key)

    t1 = time.perf_counter()
    print(f"All tasks completed in {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    save_dir = "data/bayes_sim/"
    save_PATH = save_dir + 'HeteroWhite_sequential_09_31.pkl'

    run_grid(save_PATH)
    # Runtime for sequential runs : 112s
    # Same code with 8 workers: 40s