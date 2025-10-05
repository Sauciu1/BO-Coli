import torch
from IPython.display import display
from typing import Literal, List, Tuple
import pandas as pd
import src.distribution_functions
from src.ax_helper import dtype


class NNarySearch:
    """
    Performs an n-nary (e.g. 2 is binary) search on the given data to localize it within a normalized range.
    Returns the indexed region of interest
    """

    def __init__(
        self, n: int = 2, bounds: Tuple[float, float] = (0.1, 0.90), split_power: float = 1
    ) -> None:
        self.n = n
        self.bounds = bounds
        self.history = dict()
        self.data = None
        self.split_power = float(split_power)
        self.iterations = 0
        self.indices = None

    def _preprocess_data(self, data):
        """Indexes and normalizes data"""
        if isinstance(data, pd.Series):
            self.indices = data.index
            data = torch.tensor(data.values, dtype=dtype)
        elif not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=dtype)
            self.indices = torch.arange(len(data))
        else:
            self.indices = torch.arange(len(data))

        self.history[0] = ["initial", self.indices[0].item(), self.indices[-1].item()]

        min_val = data.min()
        max_val = data.max()
        self.data = (data - min_val) / (max_val - min_val)
        return self.data

    def _split_bounds(self, data, start=0, end=None) -> List[Tuple[int, int]]:
        """Splits data by returning start/end bounds as tuples, according to the function init parameters."""
        if end is None:
            end = len(data)
            
        # Safety check to ensure valid bounds
        if start >= end:
            return [(start, min(start+1, len(data)))]
        
        # Calculate how to split the range
        range_size = end - start
        
        # Create power-based region sizes
        reg_sizes = torch.tensor([self.split_power**i for i in range(self.n)], dtype=dtype)
        
        # Normalize region sizes to fit within range
        total_units = reg_sizes.sum().item()
        unit_size = range_size / total_units
        reg_sizes = reg_sizes * unit_size
        
        # Convert to integers while preserving total size
        int_sizes = reg_sizes.floor().to(torch.int64)
        remaining = range_size - int_sizes.sum().item()
        
        # Distribute remaining elements to maintain total size
        if remaining > 0:
            fracs = reg_sizes - int_sizes.float()
            _, indices = torch.sort(fracs, descending=True)
            for i in range(min(remaining, len(indices))):
                int_sizes[indices[i]] += 1
        
        # Create bounds
        bounds = []
        curr = start
        for size in int_sizes:
            size_val = size.item()
            if size_val <= 0:  # Ensure minimum region size
                size_val = 1
                
            next_pos = curr + size_val
            if next_pos > end:  # Ensure we don't exceed the end boundary
                next_pos = end
                
            bounds.append((curr, next_pos))
            curr = next_pos
            
            if curr >= end:  # Stop if we've reached the end
                break
        
        return bounds


    def _check_region_type(
        self, start, end
    ) -> Literal[
        "target_match", "inflection", "stable_low", "stable_high", "boundary"
    ]:
        """Evaluates region type"""
        region_data = self.data[start:end]
        min_bound, max_bound = region_data.min(), region_data.max()

        if min_bound > self.bounds[0] and max_bound < self.bounds[1]:
            return "target_match"
        elif min_bound < self.bounds[0] and max_bound > self.bounds[1]:
            return "inflection"
        elif max_bound < self.bounds[0]:
            return "stable_low"
        elif min_bound > self.bounds[1]:
            return "stable_high"
        return "boundary"

    def display_history(self) -> None:
        """Shows history of actions in ipython notebook"""
        for key, val in self.history.items():
            print("Iteration:", key)
            df = pd.DataFrame(val).T
            display(df)

    def _add_history(self, regions: List[Tuple[int, int]], reg_types: List[str]) -> None:
        """Extends region analysis history with current split"""
        res = []
        for (start, end), typ in zip(regions, reg_types):
            region_data = self.data[start:end]
            res.append(
                {
                    "region": typ,
                    "start": start,
                    "end": end - 1,
                    "min": region_data.min().item(),
                    "max": region_data.max().item(),
                }
            )
        self.history[max(self.history.keys()) + 1] = res

    def run_search(self, data, plot=False) -> torch.Tensor:
        """Runs the n-nary search on the provided data."""
        data = self._preprocess_data(data)
        start, end = 0, len(data)

        while True:
            # Safety check
            if end - start <= 1:
                break
            
            regions = self._split_bounds(data, start, end)
            
            # Use the region bounds from the split, not the original start/end
            region_types = [self._check_region_type(s, e) for s, e in regions]

   
            self._add_history(regions, region_types)


            target_bounds = [
                region
                for region, typ in zip(regions, region_types)
                if typ in ["boundary", "inflection"]
            ]
            
            if not target_bounds:
                break
                
            new_start = target_bounds[0][0]
            new_end = target_bounds[-1][1]
            
            # Make sure we're making progress and have valid bounds
            if new_end <= new_start or (new_start == start and new_end == end):
                break
                
            start, end = new_start, new_end
            
            if "target_match" in region_types or len(target_bounds) == self.n:
                break

        self.iterations = max(self.history.keys())
        return self.data[start:end], start, end
