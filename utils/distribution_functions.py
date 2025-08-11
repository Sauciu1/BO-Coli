import torch

def logistic_tensor(x_0: float, k: float, start: float, end: float, steps: int = 100000) -> (torch.Tensor, torch.Tensor):
    """Generates a logistic function tensor.
    x_0: The x-value of the sigmoid's midpoint.
    k: The logistic growth rate.
    start: The start of the range.
    end: The end of the range.
    steps: The number of steps in the range.
    """
    linspace = torch.linspace(start, end, steps)

    logistic = 1 / (1 + torch.exp(-k *(linspace - x_0) ))
    return linspace, logistic

