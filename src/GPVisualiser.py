# Created by Povilas (GitHub: Sauciu1) on 2025-09-15
# last updated on 2025-09-15 by Povilas


from ax import Client

from ax.plot import render
from ipykernel.pickleutil import istype
import pandas as pd
import torch
from torch import Tensor
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ax import RangeParameterConfig
from sklearn.preprocessing import FunctionTransformer
import src.ax_helper as ax_helper
from src.ax_helper import get_train_Xy, get_obs_from_client, UnitCubeScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Default dtype for the visualiser tensors - keep high precision by default
dtype = torch.float64





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
        feature_range_params: RangeParameterConfig = None,
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
        self.dim_cols = dim_cols

        # Create mask that is True if there's NA in any of the specified columns for each row

        obs_X, obs_y = self.get_obs_X_y(obs)

        
        if feature_range_params is not None:
            self.scaler = UnitCubeScaler(ax_parameters=feature_range_params)
            self.scaler.set_output(transform="pandas")
        else:
            self.scaler = FunctionTransformer(lambda x: x, validate=False)


        self.gp = self._train_gp(gp)
        self.subplot_dims = subplot_dims(self.obs_X.shape[1])

        self.fig = None


    def get_obs_X_y(self, obs:pd.DataFrame):
        mask_na = obs[[self.response_col] + self.dim_cols].isna().any(axis=1)

        self.predict_X = obs.loc[mask_na, self.dim_cols]
        self.predict_y = obs.loc[mask_na, self.response_col]
        
        self.obs_X = obs.loc[~mask_na, self.dim_cols]
        self.obs_y = obs.loc[~mask_na, self.response_col]
        return self.obs_X, self.obs_y


    def _train_gp(self, gp: callable):
        """Train the GP model using the observed data."""

        train_X = self.scaler.fit_transform(self.obs_X.values)

        if istype(train_X, pd.DataFrame):
            train_X = train_X.values

        train_X = torch.tensor(train_X, dtype=dtype, device=device)
        train_Y = torch.tensor(self.obs_y.values, dtype=dtype, device=device).unsqueeze(
            -1
        )

        return gp(train_X, train_Y)
        

    

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

        test_X = pd.DataFrame(self.scaler.transform(test_X))


        test_X = torch.tensor(test_X.values, dtype=dtype, device=device)

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
    
    def get_best_observed_coord(self) -> pd.Series:
        """Get the coordinates of the best observed point."""
        best_idx = self.obs_y.idxmax()
        return self.obs_X.loc[best_idx]

    def plot_all(
        self, coordinates: list[float] | Tensor | pd.Series, linspace=None, figsize=(12,6)
    ) -> None:
        """Handle plotting all dimensions. One subplot per dimension.
        Requires definition of:
        * self.create_subplot,
        * self.plot_gp,
        * self._plot_observations,
        * self._vlines,
        * self._add_subplot_elements
        """

        if coordinates is None:
            coordinates = self.get_best_observed_coord()
            

        # Normalize coordinates into a torch.Tensor (avoid torch.tensor(tensor))
        if isinstance(coordinates, pd.Series):
            coordinates = torch.tensor(coordinates)
        elif isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates)


        self.fig, axs = self._create_subplots(figsize=figsize)

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
    
    def _get_distance_to_plane(self, obs:pd.DataFrame, coord:Tensor, fixed_dim:int) -> Tensor:
        """get distance from each observation to the plane parallel to fixed_dim and passing through coord"""

        if not isinstance(obs, Tensor):
            obs = torch.tensor(obs.values, dtype=dtype)
        if not isinstance(coord, Tensor):
            coord = torch.tensor(coord, dtype=dtype)

        obs[:, fixed_dim] = coord[fixed_dim]


        dist = torch.sqrt(torch.sum(torch.pow(torch.subtract(obs, coord), 2), dim=1)) 
        return dist.abs()



    def _get_size(self, obs:pd.DataFrame, coordinates:Tensor, dim) -> list[float]:
        if not isinstance(dim, int):
            dim = self.obs_X.columns.get_loc(dim)

        distances =  self._get_distance_to_plane(obs, coordinates, dim)

        return torch.tensor([(1 - d / max(distances+torch.tensor([0.001], dtype=dtype))) for d in distances], dtype=dtype)
    

    def _add_expected_improvement(self, ax:plt.Axes, dim:str, coordinates:Tensor):
        
        mean, std = self._eval_gp(self.predict_X)
        
        dim_x = self.predict_X.loc[:, dim]

        sizes = self._get_size(self.predict_X, coordinates, dim)

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
     
                    alpha=0.3,
                    linewidth=sz * 1+0.1,
                    capsize=sz * 1+0.1,
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



    def _create_subplots(self, figsize=(12,6)):
        fig, axs = plt.subplots(*self.subplot_dims, figsize=figsize)
        axs = axs.flatten() if self.obs_X.shape[1] > 1 else [axs]
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
        dim = self.obs_X.columns.get_loc(dim_name)
        point_size = self._get_size(self.obs_X, coordinates, dim)

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

    def return_fig(self):
        return self.fig
    


class PlotlyAxWrapper:
    """Wrapper to make Plotly subplots behave like matplotlib axes."""
    def __init__(self, fig, row, col):
        self.fig = fig
        self.row = row
        self.col = col

class GPVisualiserPlotly(GPVisualiser):
    def _plot_expected_improvement(self, ax, x, mean, std, sizes):
        if len(x) == 0:
            return
        elif len(x)==1:
            x, mean, std, sizes = [x.item()], [mean.item()], [std.item()], [sizes.item()]
        else:
            # Convert tensors to numpy/python types
            x = x.cpu().numpy() if hasattr(x, 'cpu') else x.values
            mean = mean.cpu().numpy() if hasattr(mean, 'cpu') else mean
            std = std.cpu().numpy() if hasattr(std, 'cpu') else std
            sizes = sizes.cpu().numpy() if hasattr(sizes, 'cpu') else sizes

        # Plot each predicted point as a separate error bar
        for xi, mi, si, sz in zip(x, mean, std, sizes):
            # Error bar line
            ax.fig.add_trace(
                go.Scatter(
                    x=[float(xi), float(xi)],
                    y=[float(mi - 2 * si), float(mi + 2 * si)],
                    mode='lines',
                    line=dict(color='red', width=float(sz * 1 + 0.1)),
                    opacity=0.3,
                    showlegend=False,
                    name='Predicted (selected point)'
                ),
                row=ax.row, col=ax.col
            )
            
            # Center point
            ax.fig.add_trace(
                go.Scatter(
                    x=[float(xi)],
                    y=[float(mi)],
                    mode='markers',
                    marker=dict(color='red', size=float(sz * 8 + 3)),
                    opacity=0.3,
                    showlegend=False,
                    name='Predicted (selected point)'
                ),
                row=ax.row, col=ax.col
            )

    @staticmethod
    def _plot_gp(grid: Tensor, mean: Tensor, std: Tensor, ax, coordinates):
        """Plot GP mean and confidence intervals along a single dimension."""
        grid_np = grid.cpu().numpy()
        mean_np = mean.cpu().numpy()
        std_np = std.cpu().numpy()
        
        # Add confidence interval
        ax.fig.add_trace(
            go.Scatter(
                x=np.concatenate([grid_np, grid_np[::-1]]),
                y=np.concatenate([mean_np - 2 * std_np, (mean_np + 2 * std_np)[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                showlegend=False
            ),
            row=ax.row, col=ax.col
        )
        
        # Add mean line
        ax.fig.add_trace(
            go.Scatter(
                x=grid_np,
                y=mean_np,
                mode='lines',
                name='GP mean',
                line=dict(color='blue'),
                showlegend=False
            ),
            row=ax.row, col=ax.col
        )

    def _vlines(self, ax, coordinates):
        # Get y-axis range for the subplot
        y_range = [float(self.obs_y.min() * 0.95), float(self.obs_y.max() * 1.05)]
        
        ax.fig.add_trace(
            go.Scatter(
                x=[float(coordinates), float(coordinates)],
                y=y_range,
                mode='lines',
                line=dict(color='black', dash='dash', width=2),
                opacity=0.7,
                name='Current Dim Value',
                showlegend=False
            ),
            row=ax.row, col=ax.col
        )

    def _create_subplots(self, figsize=(12, 6)):
        rows, cols = self.subplot_dims
        
        # Create subplot titles
        subplot_titles = [f"GP along {col}" for col in self.obs_X.columns]
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Create wrapper objects for each subplot
        axs = []
        for i in range(self.obs_X.shape[1]):
            row = (i // cols) + 1
            col = (i % cols) + 1
            axs.append(PlotlyAxWrapper(fig, row, col))
        
        return fig, axs

    def _add_subplot_elements(self, rounded_coords):
        # Update layout
        self.fig.update_layout(
            title=dict(
                text=f"GP Along Each Dimension for point {rounded_coords}",
                x=0.5,
                xanchor='center'
            ),
            height=400 * self.subplot_dims[0],
            width=600 * self.subplot_dims[1],
            showlegend=True,
            legend=dict(
                x=1.02,
                y=0.5,
                xanchor="left",
                yanchor="middle"
            )
        )

        # Add legend by making first occurrence of each trace name visible
        seen_names = set()
        for trace in self.fig.data:
            if trace.name and trace.name not in seen_names:
                trace.showlegend = True
                seen_names.add(trace.name)

    def _plot_observations(self, ax, dim_name: str, coordinates):
        """Plot observed data points along a single dimension."""
        dim = self.obs_X.columns.get_loc(dim_name)
        point_size = self._get_size(self.obs_X, coordinates, dim)
        
        # Convert tensors to numpy for plotly
        if hasattr(point_size, 'cpu'):
            point_size = point_size.cpu().numpy()
        elif hasattr(point_size, 'numpy'):
            point_size = point_size.numpy()
        
        ax.fig.add_trace(
            go.Scatter(
                x=self.obs_X[dim_name].values,
                y=self.obs_y.values,
                mode='markers',
                marker=dict(
                    size=[float(s * 20 + 5) for s in point_size],
                    color='orange',
                    line=dict(width=1, color='black'),
                    opacity=0.7
                ),
                name='Observations',
                showlegend=False
            ),
            row=ax.row, col=ax.col
        )

    def plot_all(self, coordinates: list[float] | Tensor | pd.Series, linspace=None, figsize=(12, 6)):
        """Handle plotting all dimensions using Plotly subplots."""
        
        if coordinates is None:
            coordinates = self.get_best_observed_coord()

        # Normalize coordinates into a torch.Tensor
        if isinstance(coordinates, pd.Series):
            coordinates = torch.tensor(coordinates.values, dtype=dtype)
        elif isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates, dtype=dtype)

        self.fig, axs = self._create_subplots(figsize=figsize)

        linspace = self._create_linspace(100) if linspace is None else linspace
        
        for i in range(self.obs_X.shape[1]):
            ax = axs[i]

            if not isinstance(linspace[i], Tensor):
                grid = torch.tensor(linspace[i], dtype=dtype, device=device)
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

    def _add_expected_improvement_plotly(self, fig, dim: str, coordinates: Tensor, row, col):
        mean, std = self._eval_gp(self.predict_X)
        dim_x = self.predict_X.loc[:, dim]
        sizes = self._get_size(self.predict_X, coordinates, dim)
        self._plot_expected_improvement(fig, dim_x, mean, std, sizes, row, col)

    def return_fig(self):
        return self.fig

    def show(self):
        """Display the Plotly figure."""
        self.fig.show()

if __name__ == "__main__":
    from src.toy_functions import ResponseFunction
    from botorch.models import SingleTaskGP


    dim_names = ["x0","x1"]
    dim_names = [f"x{i}" for i in range(len(dim_names))]
    simple_func = lambda x: sum(torch.sqrt(x))

    resp = ResponseFunction(simple_func, len(dim_names))
    resp.evaluate(torch.tensor([1., 4]))

    client = Client()

    parameters=[
        RangeParameterConfig(
            name=dim,
            bounds=(1, 100),
            parameter_type="float",
            # scaling = 'log',
        ) for dim in dim_names
    ]


    client.configure_experiment(
        name="batch_bo_test",
        parameters=parameters
    )


    client.configure_optimization(objective="response")

    client.get_next_trials(max_trials=10)




    for i, trial in get_obs_from_client(client).iterrows():
        if not pd.isna(trial['response']):
            continue

        response = resp.evaluate(trial[dim_names])
        client.complete_trial(trial_index=i, raw_data={"response": float(response)})




    client.get_next_trials(max_trials=6)

    obs = get_obs_from_client(client)
    
    # Test Plotly version
    plotter_plotly = GPVisualiserPlotly(SingleTaskGP, obs, dim_names, 'response', parameters)
    fig, _ = plotter_plotly.plot_all(coordinates=torch.tensor([10.0, 10.0]))
    fig.show(render="browser")
    import tempfile, webbrowser, plotly.io as pio

    # Explicitly write figure to an ephemeral HTML file and open in default browser
    _html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as _f:
        _f.write(_html)
    webbrowser.open("file://" + _f.name)


    None