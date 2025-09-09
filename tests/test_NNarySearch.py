import pytest
from torch.utils import data
from src.NNarySearch import NNarySearch
from src.distribution_functions import logistic_tensor




def test_NNarySearch_imported():
    search = NNarySearch()
    assert search is not None


class TestNNarySearch_success:
    def setup_method(self):
        self.bounds = [0.1, 0.9]
        self.search = NNarySearch(n=6, bounds=self.bounds, split_power=0.3)
        linspace, logistic = logistic_tensor(100.0, 1e-3, 0, 1e6)
        self.data, self.start, self.end = self.search.run_search(logistic)


    def test_search_initialized(self):
        """Initialise the search object."""
        assert self.search is not None

    def test_search_ran(self):
        assert self.search.iterations > 0

    
    def test_region_exists(self):
        """The search object should be able to run the search."""       
        assert self.data is not None
        assert len(self.data) > 0

    def test_inflexion_within_bounds(self):
        """The search should respect the bounds."""
        assert self.data.min() <= self.bounds[0]
        assert self.data.max() >= self.bounds[1]
        
