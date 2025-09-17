
from venv import create
import torch
from torch import Tensor
import pandas as pd
import numpy as np


def sinc(x: Tensor) -> Tensor:
    return torch.sinc(x / torch.pi)

def linear(x: Tensor, slope: float = 1.0, intercept: float = 0.0) -> Tensor:
    return slope * x + intercept

def quadratic_from_roots(x: Tensor, r1: float = 0.0, r2: float = 0.0) -> Tensor:
    return (x - r1) * (x - r2)

def cubic_from_roots(x: Tensor, r1: float = 0.0, r2: float = 0.0, r3: float = 0.0) -> Tensor:
    return (x - r1) * (x - r2) * (x - r3)

def polynomial_from_roots(x: Tensor, roots: list[float]) -> Tensor:
    result = torch.ones_like(x)
    for r in roots:
        result *= (x - r)
    return result

def logarithmic(x: Tensor, a: float = 1.0, b: float = 1.0) -> Tensor:
    return a * torch.log(b * x + 1)


arrays = {
        "sinc": lambda x: sinc(x) * 10,
        "linear": lambda x: linear(x, 0.2, 1),
        "quadratic": lambda x: polynomial_from_roots(x, [4, 16])*-0.1,
        "cubic": lambda x: polynomial_from_roots(x, [2, 9, 18])*0.02,
        "logarithmic": lambda x: logarithmic(x, 1, 1)*2,
        "sqrt": lambda x: torch.sqrt(x)
    }

def six_curve_sum(coord) -> Tensor:
    valuation = [func(coord) for func, coord in zip(arrays.values(), coord)]
    return sum(valuation)





class ResponseFunction:
    """Handles valuation for generated response functions"""
    def __init__(self, function: callable, n_dim: int):
        if not isinstance(n_dim, int):
            raise TypeError("n_dim must be an integer")
        elif n_dim <= 0:
            raise ValueError("n_dim must be a positive integer")
        self.n_dim = n_dim


        if not callable(function):
            raise TypeError("function must be callable")

        self.function = function
        self.mesh = None
        self.y = None
        

    def evaluate(self, coord: Tensor) -> Tensor:
        """Evaluates the surface at given coordinates."""

            
        if isinstance(coord, pd.Series):
            coord = torch.tensor(coord.iloc[:].astype(float).values, dtype=torch.float64)
        elif isinstance(coord, np.ndarray):
            coord = torch.tensor(coord, dtype=torch.float64)
        elif not isinstance(coord, Tensor):
            coord = torch.tensor(coord, dtype=torch.float64)

        if coord.shape[0] != self.n_dim:
            raise ValueError(f"Coordinate first dimension {coord.shape[0]} does not match expected {self.n_dim}")



        return self.function(coord)
    
    def get_mesh(self) -> Tensor:
        return self.mesh

    def get_y(self) -> Tensor:
        return self.y
    
    def get_all(self) -> tuple[Tensor, Tensor]:
        return self.mesh, self.y

    def create_mesh(self, domain: Tensor) -> Tensor:
        self.mesh = torch.meshgrid(*domain)
        self.y = self.evaluate(self.mesh)
        return self.y




if __name__ == "__main__":
    domain = [torch.linspace(0, 20, 100), torch.linspace(0, 20, 100), torch.linspace(0, 20, 100),
              torch.linspace(0, 20, 100), torch.linspace(0, 20, 100), torch.linspace(0, 20, 100)]
    test_func = ResponseFunction(six_curve_sum, n_dim=6)

    t = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    print(t)
    print(t.shape)

    print(test_func.evaluate(t))
