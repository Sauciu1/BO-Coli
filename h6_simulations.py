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



def run_grid(save_path ):
    n_runs = 100
    mp.set_start_method("spawn", force=True)  # Windows-safe

    t0 = time.perf_counter()
    print("Starting batch Bayesian optimization tests...")

    #param_grid = [(tr, float(n)) for tr in range(1, 9, 2) for n in np.linspace(0, 1.1, 6)]
    param_grid = [(tr, n, f"again_{again}") for tr in range(1, 10, 2) for n in np.linspace(0, 2.2, 11) for again in range(1, 10)]
    n_workers = min(len(param_grid), int(os.cpu_count()*0.8) or 1)


    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        results = pool.starmap(_single_run, [(el[0:2], n_runs) for el in param_grid])

    r_n_dict = dict(zip(param_grid, results))

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    with open(save_path, 'wb') as f:
        pickle.dump(r_n_dict, f)

    elapsed = time.perf_counter() - t0
    print(f"Completed {len(param_grid)} configs in {elapsed:.2f}s with {n_workers} workers")
    r_n_dict


if __name__ == "__main__":
    save_dir = "data/bayes_sim/"
    save_PATH = save_dir + 'PBS_test_vs_t_repeats_09_28.pkl'

    run_grid(save_PATH)
    # Runtime for sequential runs : 112s
    # Same code with 8 workers: 40s