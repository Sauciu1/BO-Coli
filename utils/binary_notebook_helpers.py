from . import NNarySearch, distribution_functions
import pandas as pd
def run_simulations(center_space, power_space, split_space) -> pd.DataFrame:
    """Run simulations for different parameter combinations."""
    results = []
    for log_center in center_space:
        linspace, logistic = distribution_functions.logistic_tensor(float(log_center), 1e-3, 0, 1e6)
        for splits in split_space:
            for power in power_space:
                power = round(power, 1)
                search= NNarySearch(splits, split_power=power)

                search.run_search(logistic)

                results.append({
                    "log_center": log_center,
                    "splits": splits,
                    "power": power,
                    "value": search.iterations,
                })
    df = pd.DataFrame(results)
    pivot = df.pivot_table(
        index="splits",
        columns="power",
        values="value",
        aggfunc="mean"
    )

    return pivot