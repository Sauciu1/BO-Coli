import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import src.theme_branding
import seaborn as sns
import pandas as pd

colors = src.theme_branding.BRAND_COLOURS


def plot_gp(gp:GaussianProcessRegressor, X:np.array, y:np.array, X_train:np.array, y_train:np.array):
    gp.fit(X_train, y_train)
    mean_prediction, std_prediction = gp.predict(X, return_std=True)


    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
        color = colors[3]
    )

    sns.scatterplot(x=X_train.ravel(), y=y_train, marker='x', color='r', label='Noisy Observations', s=100)

    plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted", color='k')

    df = pd.DataFrame({'X_train': X_train.ravel(), 'y_train': y_train.ravel()})
    grouped = df.groupby('X_train')['y_train'].mean()
    X_unique = grouped.index.values
    y_matched = grouped.values



    std = df.groupby('X_train')['y_train'].std().values
    noise_alpha = 1.96 * std


    plt.errorbar(
        X_unique,
        y_matched,
        noise_alpha,
        linestyle="None",
        color="k",
        marker = '.',
        markersize=10,
        label="Observations",
    )

    plt.legend(loc="upper left")
    plt.xlabel("Input $x$")
    plt.ylabel("Output $y$")
    plt.title("Gaussian Process Regression")