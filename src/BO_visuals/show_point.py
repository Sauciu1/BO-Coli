from urllib import response
from ax import Client
import pandas as pd
import torch
from torch import Tensor
import math



def get_trial_data(client: Client) -> pd.DataFrame:
    """Get trial data from an Ax client as a pandas DataFrame.
    Variable names as columns and response as 'response' column.
    """

    results = pd.DataFrame()
    for trial in client._experiment.trials.values():
        df = trial.arm.parameters
        df["response"] = float(trial.fetch_data().df["mean"].values[0])

        df = pd.DataFrame([df])
        results = pd.concat([results, df], ignore_index=True)

    return results



def get_train_Xy(trial_df:pd.DataFrame, response_col = 'response') -> tuple[Tensor, Tensor]:
    """Get training data from trial DataFrame."""
    if response_col not in trial_df.columns:
        raise KeyError(f"Response column '{response_col}' not found in DataFrame.")


    non_response_cols = [col for col in trial_df.columns if col != response_col]

    train_X = torch.tensor(trial_df[non_response_cols].values, dtype=torch.float32)
    train_Y = torch.tensor(trial_df[response_col].values, dtype=torch.float32).unsqueeze(-1)
    return train_X, train_Y

def get_plane_gaussian_for_point(gp, coordinates: Tensor, fixed_dim:int, grid: Tensor) -> tuple:
    """returns mean and std of GP prediction along a grid in fixed_dim, holding other dims at coordinates"""


    zeros = torch.zeros_like(grid)
    test_X = torch.stack([ zeros+coordinates[i] if i!=fixed_dim else grid for i in range(len(coordinates))], dim=-1)

    # Predict GP mean and variance
    gp.eval()
    with torch.no_grad():
        posterior = gp.posterior(test_X)
        mean = posterior.mean.squeeze().numpy()
        std = posterior.variance.sqrt().squeeze().numpy()
    
    return mean, std



def grid_dims(results_df, response_col=True) -> tuple[int, int]:
    """Get grid dimensions for plotting based on number of results."""
    n = results_df.shape[1] - 1*response_col  # Exclude response column
    length = math.ceil(n**0.5)
    return length, math.ceil(n / length)