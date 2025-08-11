import pandas as pd
from IPython.display import display
from typing import Literal
import seaborn as sns
import numpy as np


class NNarySearch:
    """
    Performs an n-nary (e.g. 2 is binary) search on the given data to localize it within a normalized range.
    Returns the indexed region of interest
    """

    def __init__(
        self, n: int = 2, bounds: float = (0.1, 0.90), split_power: float = 1
    ) -> float:
        self.n = n
        self.bounds = bounds
        self.history = dict()
        self.data = None
        self.split_power = float(split_power)
        self.iterations = 0

        self.split_function = self._logistic_split_data

    def _preprocess_data(self, data):
        """Indexes and normalizes data"""
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        self.history[0] = ["initial", data.index[0], data.index[-1]]
        self.data = (data - data.min()) / (data.max() - data.min())
        return self.data

    def _logistic_split_data(self, data):
        reg_sizes = self.split_power**np.arange(self.n)
        reg_sizes = reg_sizes/reg_sizes.min()
        reg_sizes = reg_sizes*(len(data)//sum(reg_sizes))
        remainder = len(data)% sum(reg_sizes)

        reg_sizes[-1]=reg_sizes[-1]+remainder 

        start = 0
        splits = list()
        for size in reg_sizes.astype(int):

            splits.append(data[start : start + size])
            start += size
        return splits

    def _linear_split_data(self, data):
        """
        Split a NumPy array into exactly `self.n` contiguous parts, preserving the original index.
        """

        sizes = [
            (int(len(data) / self.n)) + (1 if i < len(data) % self.n else 0)
            for i in range(self.n)
        ]
        start = 0
        splits = list()
        for size in sizes:
            splits.append(data[start : start + size])
            start += size
        return splits

    def _check_region(self, data):
        """Evaluates region type"""
        min_bound, max_bound = data.min(), data.max()

        if min_bound > self.bounds[0] and max_bound < self.bounds[1]:
            return "target_match"
        elif min_bound < self.bounds[0] and max_bound > self.bounds[1]:
            return "inflection"

        elif max_bound < self.bounds[0]:
            return "stable_low"
        elif min_bound > self.bounds[1]:
            return "stable_high"

        return "boundary"

    def display_history(self):
        """Shows history of actions in ipython notebook"""
        for key, val in self.history.items():
            print("Iteration:", key)
            df = pd.DataFrame(val).T
            display(df)

    def _add_history(self, regions: list[pd.Series], reg_types: list[str]):
        """Extends region analysis history with current split"""
        res = []
        for reg, typ in zip(regions, reg_types):
            res.append(
                {
                    "region": typ,
                    "start": reg.index[0],
                    "end": reg.index[-1],
                    "min": reg.min(),
                    "max": reg.max(),
                }
            )
        self.history[max(self.history.keys()) + 1] = res

    def run_search(self, data, plot = False):
        data = self._preprocess_data(data)

        while True:
            if plot:
                sns.lineplot(data)
            regions = self.split_function(data)
            region_types = [self._check_region(region) for region in regions]

            self._add_history(regions, region_types)

            target_reg = [
                regions[i]
                for i, reg in enumerate(region_types)
                if reg in ["boundary", "inflection"]
            ]
            data = pd.concat(target_reg)


            if "target_match" in region_types or len(target_reg) == self.n:
                break
        
        self.iterations = max(self.history.keys())
        return data


if __name__ == "__main__":
    from distribution_functions import logistic_tensor
    from matplotlib import pyplot as plt
    import seaborn as sns
    linspace, logistic = logistic_tensor(3e5, 1e-3, 0, 1e6)


if __name__ =="__main__":


    grid = pd.DataFrame()
    for splits in range(2, 6):
        for power in np.linspace(0.2, 2, 10):
            search = NNarySearch(splits, split_power=float(power))
            search.run_search(logistic)
            print(search.iterations)



if __name__ == "__main__a":



    nary = NNarySearch(n=3, bounds=(0.1, 0.9), split_power=0.2)
    res = nary.run_search(logistic, True)

    sns.lineplot(res)

    plt.show()

