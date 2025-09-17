from ax import Client

from typing import Literal
from ax.core import trial
import pandas as pd
from torch import Tensor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64


def safe_get(df, key, index, axis=0, default=torch.nan):
    try:
        return df.loc[index, key]
    except KeyError:
        return default


def get_guess_coords(client:Client, output_format: Literal["df", "tensor"]="df") -> Tensor:
    """Get the coordinates of all the guesses made so far in the optimization."""
    full_df = pd.DataFrame()
    for index, trial in client._experiment.trials.items():

        df = pd.DataFrame({'trial_name': trial.arm.name,**trial.arm.parameters}, index=[index])

        full_df = pd.concat([full_df, df], ignore_index=True)
    if output_format.lower() == "df":
        return full_df
    elif output_format.lower() == "tensor":
        
        return torch.as_tensor(
            full_df[[dim for dim in full_df.columns if dim != "name"]].values,
            dtype=dtype,
            device=device,
        )
    


def get_obs_from_client(client: Client, response_col:str) -> pd.DataFrame:
    """Get trial data from an Ax client as a pandas DataFrame.
    Variable names as columns and response as 'response' column, return missing response as NaN.
    """
    obs = get_guess_coords(client, output_format="df")
    results = client._experiment.fetch_data().df
    obs[response_col] = obs.index.map(lambda index: safe_get(results, 'mean', index))



    return obs


def get_train_Xy(
    trial_df: pd.DataFrame, dim_cols:list[str], response_col="response"
) -> tuple[Tensor, Tensor]:
    """Get training data from trial DataFrame."""



    if response_col not in trial_df.columns:
        raise KeyError(f"Response column '{response_col}' not found in DataFrame.")
    trial_df = trial_df.dropna(subset=[response_col])
 

    train_X = torch.tensor(trial_df[dim_cols].values, dtype=dtype, device=device).T
    train_Y = torch.tensor(trial_df[response_col].values, dtype=dtype, device=device).unsqueeze(-1)
    return train_X, train_Y