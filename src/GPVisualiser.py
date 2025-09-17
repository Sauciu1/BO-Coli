# Created by Povilas (GitHub: Sauciu1) on 2025-09-15
# last updated on 2025-09-15 by Povilas


from ax import Client

import pandas as pd
import torch
from torch import Tensor
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from .ax_helper import get_train_Xy, get_obs_from_client


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Default dtype for the visualiser tensors - keep high precision by default
dtype = torch.float64


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
    length = math.floor(n**0.5)
    return length, math.ceil(n / length)


class GPVisualiser:
    def __init__(
        self,
        gp: callable,
        obs: pd.DataFrame,
        dim_cols: list[str],
        response_col="response",
    ) -> None:

        if not callable(gp):
            raise TypeError(
                "gp must be a callable that returns a trained GP model when given training data"
            )
        if not isinstance(obs, pd.DataFrame):
            raise TypeError("obs must be a pandas DataFrame")
        if not isinstance(dim_cols, list) or not all(
            isinstance(col, str) for col in dim_cols
        ):
            raise TypeError("dim_cols must be a list of strings")
        if not isinstance(response_col, str):
            raise TypeError("response_col must be a string")

        self.response_col = response_col

        mask_na = obs[response_col].isna()
        self.predict_X = obs.loc[mask_na, dim_cols]
        self.predict_y = obs.loc[mask_na, response_col]

        self.obs_X = obs.loc[~mask_na, dim_cols]
        self.obs_y = obs.loc[~mask_na, response_col]



        train_X = torch.tensor(self.obs_X.values, dtype=dtype, device=device)
        train_Y = torch.tensor(self.obs_y.values, dtype=dtype, device=device).unsqueeze(
            -1
        )

        self.gp = gp(train_X, train_Y)
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
    
    def _eval_gp(self, test_X:Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the GP model at given test points."""
        if not isinstance(test_X, Tensor):
            test_X = torch.tensor(self.predict_X.values, dtype=dtype, device=device)
        
        with torch.no_grad():
            posterior = self.gp.posterior(test_X)
            mean = posterior.mean.squeeze()
            std = posterior.variance.sqrt().squeeze()
        return mean, std

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

        mean, std = self._eval_gp(test_X)

        return mean, std

    @staticmethod
    def _get_euclidean_distance(
        point1: list[float] | Tensor, point2: list[float] | Tensor
    ) -> float:
        """Calculate Euclidean distance between two points."""
        if not isinstance(point1, Tensor):
            point1 = torch.tensor(point1)
        if not isinstance(point2, Tensor):
            point2 = torch.tensor(point2)
        return torch.dist(point1, point2).item()

    def plot_all(
        self, coordinates: list[float] | Tensor | pd.Series, linspace=None
    ) -> None:
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
            coordinates = torch.tensor(coordinates)
        elif isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates)
        elif isinstance(coordinates, torch.Tensor):
            coordinates = torch.tensor(coordinates)

        self.fig, axs = self._create_subplots()

        linspace = self._create_linspace(100) if linspace is None else linspace

        for i in range(self.obs_X.shape[1]):
            ax: plt.Axes = axs[i]

            if not isinstance(linspace[i], Tensor):
                grid = torch.tensor(
                    linspace[i], dtype=dtype, device=device
                )
            else:
                grid = linspace[i].detach().clone()


            mean, std = self._get_plane_gaussian_for_point(
                coordinates, fixed_dim=i, grid=grid
            )

            self._plot_gp(grid, mean, std, ax, coordinates)
            dim = self.obs_X.columns[i]

            self._vlines(ax, coordinates[i])

            self._plot_observations(ax, dim, coordinates)

            self._add_expected_improvement(ax, dim, coordinates)

        rounded_coords = [f"{x:.3g}" for x in coordinates]
        self._add_subplot_elements(rounded_coords)

        return self.fig, axs
    



    def _get_size(self, obs:pd.DataFrame, coordinates:Tensor) -> list[float]:
        distances = [
            GPVisualiser._get_euclidean_distance(coordinates, row_coord)
            for row_coord in obs.values
        ]
        return torch.tensor([(1 - d / max(distances)) for d in distances], dtype=dtype)
    

    def _add_expected_improvement(self, ax:plt.Axes, dim:str, coordinates:Tensor):
        
        mean, std = self._eval_gp(self.predict_X)
        
        dim_x = self.predict_X.loc[:, dim]

        sizes = self._get_size(self.predict_X, coordinates)

        self._plot_expected_improvement(ax, dim_x, mean, std, sizes)

    def _plot_expected_improvement():
        raise NotImplementedError




class GPVisualiserMatplotlib(GPVisualiser):
    def _plot_expected_improvement(self, ax, x, mean, std, sizes):
        if len(x) == 0:
            return
        elif len(x)==1:
            x, mean, std, sizes = [x], [mean], [std], [sizes]


        # Plot each predicted point as a separate error bar
        for xi, mi, si, sz in zip(x, mean, std, sizes):
                ax.errorbar(
                    xi,
                    mi,
                    yerr=2 * si,
                    fmt='',
                    color='red',
                    linestyle='--',
                    alpha=0.8,
                    linewidth=sz * 4 + 1,
                    label='Predicted (selected point)',
                )

    @staticmethod
    def _plot_gp(grid: Tensor, mean: Tensor, std: Tensor, ax: plt.Axes, coordinates):
        """Plot GP mean and confidence intervals along a single dimension."""

        ax.plot(grid.numpy(), mean, label="GP mean")
        ax.fill_between(
            grid.numpy(), mean - 2 * std, mean + 2 * std, alpha=0.3, label="95% CI"
        )


    def _vlines(self, ax, coordinates):
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
        fig, axs = plt.subplots(*self.subplot_dims, figsize=(12, 6))
        axs = axs.flatten()
        return fig, axs

    def _add_subplot_elements(self, rounded_coords):
        # Use the Figure's suptitle so we don't rely on pyplot state
        self.fig.suptitle(
            f"GP Along Each Dimension for point {rounded_coords}", fontsize=16
        )

        handles, labels = self.fig.axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        self.fig.legend(
            unique.values(),
            unique.keys(),
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
        )

        self.fig.tight_layout(rect=[0, 0, 0.86, 0.95])

    def _plot_observations(
        self,
        ax: plt.Axes,
        dim_name: str,
        coordinates,
    ) -> None:
        """Plot observed data points along a single dimension."""

        point_size = self._get_size(self.obs_X, coordinates)

        sns.scatterplot(
            x=self.obs_X[dim_name],
            y=self.obs_y,
            s=point_size*100+1,
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

    dim_cols = ["x1", "x2", "x3"]
    df = mock_data()
    # Add a few datapoints without response to mock data
    for i in range(3):
        new_row = {col: np.random.uniform(0, 10) for col in dim_cols}
        new_row["response"] = np.nan
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    

    gp = SingleTaskGP(*get_train_Xy(df, dim_cols))
    plotter = GPVisualiserMatplotlib(SingleTaskGP, df, dim_cols)
    plotter.plot_all(coordinates=[8, 8, 8])

    plt.show()
    None
