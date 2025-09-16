# Created by Povilas (GitHub: Sauciu1) on 2025-09-15
# last updated on 2025-09-15 by Povilas

from cProfile import label
from urllib import response
from venv import create
from webbrowser import get
from ax import Client
import pandas as pd
import torch
from torch import Tensor
import math
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Default dtype for the visualiser tensors - keep high precision by default
dtype = torch.float64


def _to_tensor(x, dtype_: torch.dtype | None = None, device_: torch.device | None = None):
    """Convert input to a torch.Tensor safely and place on the chosen device.

    If `x` is already a Tensor, return a detached clone with the requested dtype
    and device to avoid the `torch.tensor(tensor)` copy-construction warning.
    Otherwise, construct a new tensor from the input.
    """
    if dtype_ is None:
        dtype_ = dtype
    if device_ is None:
        device_ = device

    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(dtype=dtype_, device=device_)
    return torch.tensor(x, dtype=dtype_, device=device_)


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


def get_train_Xy(
    trial_df: pd.DataFrame, response_col="response"
) -> tuple[Tensor, Tensor]:
    """Get training data from trial DataFrame."""
    if response_col not in trial_df.columns:
        raise KeyError(f"Response column '{response_col}' not found in DataFrame.")

    non_response_cols = [col for col in trial_df.columns if col != response_col]

    # Build tensors on the configured device/dtype
    train_X = torch.tensor(trial_df[non_response_cols].values, dtype=dtype, device=device)
    train_Y = torch.tensor(trial_df[response_col].values, dtype=dtype, device=device).unsqueeze(-1)
    return train_X, train_Y


def mock_data() -> pd.DataFrame:
    """Generate mock data for testing."""
    np.random.seed(42)
    X = np.random.uniform(0, 10, size=(20, 3))

    # Define functions
    def logistic(x):
        return 1 / (1 + np.exp(-x))

    def linear(x):
        return 2 * x

    def exponential(x):
        return np.exp(0.3 * x)

    # Calculate response as sum of logistic, linear, and exponential for each dimension
    response = logistic(X[:, 0]) + linear(X[:, 1]) + exponential(X[:, 2])

    # Create DataFrame
    mock_df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    mock_df["response"] = response

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
            )
            for name in param_names
        ],
    )

    client.configure_optimization(objective="response")

    for _, row in df.iterrows():
        params = {name: float(row[name]) for name in param_names}
        trial_index = client.attach_trial(params)
        client.complete_trial(
            trial_index, raw_data={"response": (float(row["response"]), 0.1)}
        )

    return client


def subplot_dims(n) -> tuple[int, int]:
    """Get grid dimensions for plotting based on number of results."""
    length = math.ceil(n**0.5)
    return length, math.ceil(n / length)


class GPVisualiser:
    def __init__(self, gp: callable, obs: pd.DataFrame, response_col = "response") -> None:
        self.gp = gp(*get_train_Xy(obs))

        self.response_col = response_col

        self.obs_X = obs.copy()
        self.obs_y = self.obs_X.pop(response_col)
        self.subplot_dims = subplot_dims(self.obs_X.shape[1])
        self.fig = None

    def _create_linspace(self, num_points: int = 100) -> list[Tensor]:
        linspaces = []
        for i in range(self.obs_X.shape[1]):
            grid = torch.linspace(
                self.obs_X.iloc[:, i].min() * 0.95,
                self.obs_X.iloc[:, i].max() * 1.05,
                num_points,
                dtype=dtype,
                device=device,
            )
            linspaces.append(grid)
        return linspaces

    def _get_plane_gaussian_for_point(
        self, coordinates: Tensor, fixed_dim: int, grid: Tensor
    ) -> tuple:
        """returns mean and std of GP prediction along a grid in fixed_dim, holding other dims at coordinates"""

        zeros = torch.zeros_like(grid)
        test_X = torch.stack(
            [
                zeros + coordinates[i] if i != fixed_dim else grid
                for i in range(len(coordinates))
            ],
            dim=-1,
        )

        self.gp.eval()
        with torch.no_grad():
            posterior = self.gp.posterior(test_X)
            mean = posterior.mean.squeeze().numpy()
            std = posterior.variance.sqrt().squeeze().numpy()

        return mean, std

    @staticmethod
    def _get_euclidean_distance(
        point1: list[float] | Tensor, point2: list[float] | Tensor
    ) -> float:
        """Calculate Euclidean distance between two points."""
        point1 = _to_tensor(point1)
        point2 = _to_tensor(point2)
        return torch.dist(point1, point2).item()
    
    def plot_all(self, coordinates: list[float] | Tensor | pd.Series, linspace=None) -> None:
        """Handle plotting all dimensions. One subplot per dimension.
        Requires definition of:
        * self.create_subplot,
        * self.plot_gp,
        * self._plot_observations,
        * self._vlines,
        * self._add_subplot_elements
        """

        # Normalize coordinates into a torch.Tensor (avoid torch.tensor(tensor))
        if isinstance(coordinates, pd.Series):
            coordinates = coordinates[
                [col for col in coordinates.index if col != self.response_col]
            ].values
            coordinates = _to_tensor(coordinates)
        elif isinstance(coordinates, np.ndarray):
            coordinates = _to_tensor(coordinates)
        elif isinstance(coordinates, torch.Tensor):
            coordinates = _to_tensor(coordinates)

        self.fig, axs = self._create_subplots()

        linspace = self._create_linspace(100) if linspace is None else linspace

        for i in range(self.obs_X.shape[1]):
            ax: plt.Axes = axs[i]


            grid = _to_tensor(linspace[i])

            mean, std = self._get_plane_gaussian_for_point(
                coordinates, fixed_dim=i, grid=grid
            )

            self._plot_gp(grid, mean, std, ax, coordinates)
            dim = self.obs_X.columns[i]

            self._vlines(ax, coordinates[i])

            self._plot_observations(ax, dim, coordinates)

            

        rounded_coords = [f"{x:.3g}" for x in coordinates]
        self._add_subplot_elements(rounded_coords)

        return self.fig, axs


    def _create_subplots(self):
        raise NotImplementedError
    

    def _create_subplot_elements(self, fig, ax, dim, rounded_coords):
        raise NotImplementedError
    def _plot_gp(self, grid: Tensor, mean: Tensor, std: Tensor, ax, coordinates):
        raise NotImplementedError
    def _plot_observations(
        self,
        ax,
        dim_name: str,
        coordinates,
    ) -> None:
        
        raise NotImplementedError
    def _vlines(self, ax, coordinates):
        raise NotImplementedError
    def _add_subplot_elements(self, ax, rounded_coords):
        raise NotImplementedError



class GPVisualiserMatplotlib(GPVisualiser):
    import matplotlib.pyplot as plt

    @staticmethod
    def _plot_gp(grid: Tensor, mean: Tensor, std: Tensor, ax: plt.Axes, coordinates):
        """Plot GP mean and confidence intervals along a single dimension."""

        ax.plot(grid.numpy(), mean, label="GP mean")
        ax.fill_between(
            grid.numpy(), mean - 2 * std, mean + 2 * std, alpha=0.3, label="95% CI"
        )


    def _vlines(self, ax , coordinates):
        bounds = ax.get_ylim()
        ax.vlines(
            x=coordinates,
            ymin=bounds[0],
            ymax=bounds[1],
            color="k",
            linestyles="--",
            label="Current Dim Value",
            alpha=0.7,
            )
        
    def _create_subplots(self):
        fig, axs = plt.subplots(*self.subplot_dims, figsize=(12, 8))
        axs = axs.flatten()
        return fig, axs



    def _add_subplot_elements(self, rounded_coords):
        # Use the Figure's suptitle so we don't rely on pyplot state
        self.fig.suptitle(f"GP Along Each Dimension for point {rounded_coords}", fontsize=16)

       # self.fig.legend().set_visible(True)

        # Remove duplicate legend entries
        handles, labels = self.fig.axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        self.fig.legend(unique.values(), unique.keys(), loc='center right', bbox_to_anchor=(1.0, 0.5))
        

        self.fig.tight_layout(rect=[0, 0, 0.86, 0.95])


        


    def _plot_observations(
        self,
        ax: plt.Axes,
        dim_name: str,
        coordinates,
    ) -> None:
        """Plot observed data points along a single dimension."""

        distances = [
            GPVisualiser._get_euclidean_distance(coordinates, row)
            for row in self.obs_X.to_numpy()
        ]
        point_size = [(1 - d / max(distances)) * 100 + 10 for d in distances]

        sns.scatterplot(
            x=self.obs_X[dim_name],
            y=self.obs_y,
            s=point_size,
            ax=ax,
            label="Observations",
            edgecolor="k",
            alpha=0.7,
        )


        ax.set_title(f"GP along {dim_name}")
        ax.legend().set_visible(False)


if __name__ == "__main__":
    # Generate 20 datapoints in 3 dimensions
    from botorch.models import SingleTaskGP
    

    from GPVisualiser import mock_data, get_train_Xy

    df = mock_data()


    gp = SingleTaskGP(*get_train_Xy(df))
    plotter = GPVisualiserMatplotlib(SingleTaskGP, df)
    plotter.plot_all(coordinates=[8, 8, 8])

    plt.show()
    None
