# Created by Povilas (GitHub: Sauciu1) on 2025-09-15
# last updated on 2025-09-15 by Povilas



from urllib import response
from webbrowser import get
from ax import Client
import pandas as pd
import torch
from torch import Tensor
import math
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


def get_obs_from_client(client: Client) -> pd.DataFrame:
    """Get trial data from an Ax client as a pandas DataFrame.
    Variable names as columns and response as 'response' column.
    """

    obs = pd.DataFrame()
    for trial in client._experiment.trials.values():
        df = trial.arm.parameters
        df["response"] = float(trial.fetch_data().df["mean"].values[0])

        df = pd.DataFrame([df])
        obs = pd.concat([obs, df], ignore_index=True)

    return obs



def get_train_Xy(trial_df:pd.DataFrame, response_col = 'response') -> tuple[Tensor, Tensor]:
    """Get training data from trial DataFrame."""
    if response_col not in trial_df.columns:
        raise KeyError(f"Response column '{response_col}' not found in DataFrame.")

    non_response_cols = [col for col in trial_df.columns if col != response_col]

    train_X = torch.tensor(trial_df[non_response_cols].values, dtype=torch.float32)
    train_Y = torch.tensor(trial_df[response_col].values, dtype=torch.float32).unsqueeze(-1)
    return train_X, train_Y



def mock_data() -> pd.DataFrame:
    """Generate mock data for testing."""
    np.random.seed(42)
    X = np.random.uniform(0, 10, size=(20, 3))

    # Define functions
    def logistic(x): return 1 / (1 + np.exp(-x))
    def linear(x): return 2 * x
    def exponential(x): return np.exp(0.3 * x)

    # Calculate response as sum of logistic, linear, and exponential for each dimension
    response = (
        logistic(X[:, 0]) +
        linear(X[:, 1]) +
        exponential(X[:, 2])
    )

    # Create DataFrame
    mock_df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    mock_df['response'] = response

    return mock_df




def mock_client(df: pd.DataFrame) -> Client:
    from ax import Client, RangeParameterConfig
    # Generate mock data
 
    param_names = [col for col in df.columns if col != "response"]

    client = Client()
    
    client.configure_experiment(
        name="mock_experiment",
        parameters=[
         RangeParameterConfig(
            name=name,
            bounds=(0, 10),
            parameter_type="float",
        ) for name in param_names
        ],
 
    )

    client.configure_optimization(objective="response")

    # Add trials from mock data
    for _, row in df.iterrows():
        params = {name: float(row[name]) for name in param_names}
        trial_index = client.attach_trial(params)
        client.complete_trial(trial_index, raw_data={"response": (float(row["response"]), 0.1)})

    return client

def subplot_dims(n) -> tuple[int, int]:
        """Get grid dimensions for plotting based on number of results."""
        length = math.ceil(n**0.5)
        return length, math.ceil(n / length)


class GPVisualiser:
    def __init__(self, gp: callable, client: Client) -> None:
        obs = get_obs_from_client(client)
        self.gp = gp(*get_train_Xy(obs))

        self.obs_X = obs
        self.obs_y = self.obs_X.pop("response")
        self.subplot_dims = subplot_dims(self.obs_X.shape[1])


    def plot(self, coordinates: list[float] | Tensor, linspaces = None):
        
        coordinates = (
            coordinates.numpy() if isinstance(coordinates, Tensor) else coordinates
        )

        linspaces = self.create_linespace(100) if linspaces is None else linspaces


    def create_linespace(self, num_points: int = 100) -> list[Tensor]:
        linspaces = []
        for i in range(self.obs_X.shape[1]): 
            grid = torch.linspace(
                self.obs_X.iloc[:, i].min()*0.95,
                self.obs_X.iloc[:, i].max()*1.05,
                num_points
            )
            linspaces.append(grid)
        return linspaces


    def get_plane_gaussian_for_point(self, coordinates: Tensor, fixed_dim:int, grid: Tensor) -> tuple:
        """returns mean and std of GP prediction along a grid in fixed_dim, holding other dims at coordinates"""

        zeros = torch.zeros_like(grid)
        test_X = torch.stack([ zeros+coordinates[i] if i!=fixed_dim else grid for i in range(len(coordinates))], dim=-1)

        self.gp.eval()
        with torch.no_grad():
            posterior = self.gp.posterior(test_X)
            mean = posterior.mean.squeeze().numpy()
            std = posterior.variance.sqrt().squeeze().numpy()
        
        return mean, std

    @staticmethod
    def get_euclidean_distance(point1: list[float] | Tensor, point2: list[float] | Tensor) -> float:
        """Calculate Euclidean distance between two points."""
        point1 = torch.tensor(point1, dtype=torch.float32)
        point2 = torch.tensor(point2, dtype=torch.float32)
        return torch.dist(point1, point2).item()
    
    
    



class GPVisualiserMatplotlib(GPVisualiser):

    @staticmethod
    def plt_gp_along_dimension(
        grid: Tensor, mean: Tensor, std: Tensor, ax: plt.Axes
    ):
        """Plot GP mean and confidence intervals along a single dimension."""

        ax.plot(grid.numpy(), mean, label="GP mean")
        ax.fill_between(
            grid.numpy(), mean - 2 * std, mean + 2 * std, alpha=0.3, label="95% CI"
        )

    def plot_gp_all_dims(self, coordinates: list[float] | Tensor, linspace = None):

        fig, axs = plt.subplots(*self.subplot_dims, figsize=(12, 8))
        axs = axs.flatten()

        linspace = self.create_linespace(100) if linspace is None else linspace

        for i in range(self.obs_X.shape[1]):
            ax: plt.Axes = axs[i]



            grid =torch.tensor(linspace[i], dtype=torch.float32)

            mean, std = self.get_plane_gaussian_for_point(
                 coordinates, fixed_dim=i, grid=grid
            )


            rounded_coords = [f"{x:.3g}" for x in coordinates]


            self.plt_gp_along_dimension(grid, mean, std, ax)

            dim = self.obs_X.columns[i]
            self.plot_observations_along_dimension(
                ax, dim, coordinates, point_size=50
            )

            
            plt.suptitle(f"GP Along Each Dimension for point {rounded_coords}", fontsize=16)
            ax.set_xlabel(dim)
            ax.set_ylabel("Response")
            ax.set_title(f"GP along {dim}")

            y_min, y_max = ax.get_ylim()

            ax.vlines(
                x=coordinates[i],
                ymin=y_min,
                ymax=y_max,
                color="k",
                linestyles="--",
                label="Current Dim Value",
                alpha=0.7,
            )


            ax.legend().set_visible(False)

            # Move legend to the right outside the entire figure
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
            plt.tight_layout(rect=[0, 0, 0.98, 1])

    def plot_observations_along_dimension(self,
        ax: plt.Axes,
        dim_name: str,
        coordinates,
        point_size,
    ):
        """Plot observed data points along a single dimension."""


        distances = [
            GPVisualiser.get_euclidean_distance(coordinates, row)
            for row in self.obs_X.to_numpy()
        ]
        point_size = [
            (1 - d / max(distances)) * 100 + 10 for d in distances
        ]

        sns.scatterplot(
            x=self.obs_X[dim_name],
            y=self.obs_y,
            s=point_size,
            ax=ax,
            legend=False,
            edgecolor="k",
            alpha=0.7,
        )
        ax.set_title(f"Observations along {dim_name}")




if __name__ == "__main__":
    # Generate 20 datapoints in 3 dimensions
    from botorch.models import SingleTaskGP
    import matplotlib.pyplot as plt



    from GPVisualiser import mock_data, get_train_Xy, mock_client
    #df = 
    df = mock_data()
    client = mock_client(df)

    gp = SingleTaskGP(*get_train_Xy(df))
    plotter = GPVisualiserMatplotlib(SingleTaskGP, client)
    plotter.plot_gp_all_dims(coordinates=[5.0, 5.0, 5.0])

    plt.show()
    None