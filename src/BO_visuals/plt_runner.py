# Created by Povilas (GitHub: Sauciu1) on 2025-09-15
# last updated on 2025-09-15 by Povilas


import torch
from torch import Tensor
from botorch.models import SingleTaskGP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from . import show_point


def plt_gp_along_dimension(
    grid: Tensor, mean: Tensor, std: Tensor, results: pd.DataFrame, ax: plt.Axes
):
    """Plot GP mean and confidence intervals along a single dimension."""

    ax.plot(grid.numpy(), mean, label="GP mean")
    ax.fill_between(
        grid.numpy(), mean - 2 * std, mean + 2 * std, alpha=0.3, label="95% CI"
    )


def get_euclidean_distance(
    point1: list[float] | Tensor, point2: list[float] | Tensor
) -> float:
    """Calculate Euclidean distance between two points."""
    if not isinstance(point1, Tensor):
        point1 = torch.tensor(point1, dtype=torch.float32)
    if not isinstance(point2, Tensor):
        point2 = torch.tensor(point2, dtype=torch.float32)
    return torch.dist(point1, point2).item()


def plt_gp_all_dims(
    gp: SingleTaskGP, results_df: pd.DataFrame, coordinates: list[float] | Tensor
):

    fig, axs = plt.subplots(*show_point.grid_dims(results_df), figsize=(12, 8))
    axs = axs.flatten()

    coordinates = (
        coordinates.numpy() if isinstance(coordinates, Tensor) else coordinates
    )

    plt.suptitle(f"GP Along Each Dimension for point {coordinates}", fontsize=16)

    for i in range(results_df.shape[1] - 1):
        ax: plt.Axes = axs[i]

        grid = torch.linspace(0, 20, 100)
        mean, std = show_point.get_plane_gaussian_for_point(
            gp, coordinates, fixed_dim=i, grid=grid
        )

        distances = [
            get_euclidean_distance(coordinates, row[:-1])
            for row in results_df.to_numpy()
        ]
        point_size = [
            (1 - d / max(distances)) * 100 + 10 for d in distances
        ]  # Scale sizes

        plt_gp_along_dimension(grid, mean, std, results_df, ax)

        ax.set_xlabel(results_df.columns[i])
        ax.set_ylabel("Response")
        ax.set_title(f"GP along {results_df.columns[i]} at point {coordinates}")

        y_min, y_max = ax.get_ylim()



        sns.scatterplot(
            x=results_df[f"x{i}"],
            y=results_df["response"],
            color="red",
            label="Guesses",
            ax=ax,
            s=point_size,
        )


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
