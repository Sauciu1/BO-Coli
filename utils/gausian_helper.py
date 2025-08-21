import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import utils.theme_branding


colors = utils.theme_branding.BRAND_COLOURS


def plot_gaussian_process_no_noise(gp:GaussianProcessRegressor, X:np.array, y:np.array, X_train:np.array, y_train:np.array):
    gp.fit(X_train, y_train)
    mean_prediction, std_prediction = gp.predict(X, return_std=True)
    noise_alpha = gp.alpha

    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
        color = colors[3]
    )

    plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted", color='k')
    plt.errorbar(
        X_train,
        y_train,
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