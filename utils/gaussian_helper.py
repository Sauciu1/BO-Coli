import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import utils.theme_branding
import torch
import seaborn as sns


colors = utils.theme_branding.BRAND_COLOURS


def add_base_labels() -> None:
    plt.legend(loc="upper left")
    plt.xlabel("Input $x$")
    plt.ylabel("Output $y$")
    plt.title("Gaussian Process Regression")
    


def draw_ci(ax:plt.Axes, X:np.array, mean:np.array, std:float) -> None:

    
    lower = mean -  std[0] * 1.96
    upper = mean +  std[0] * 1.96

    for arr_name in ['X', 'mean', 'lower', 'upper']:
        arr = locals()[arr_name]
        if hasattr(arr, 'numpy'):
            locals()[arr_name] = arr.numpy()
            
   

    """Fills 95% confidence interval"""
    ax.fill_between(X.ravel(), lower[0], upper[0], alpha=0.5, label=r"95% CI", color=colors[3])


def plot_gaussian_process_no_noise(gp:GaussianProcessRegressor, X:np.array, y:np.array, X_train:np.array, y_train:np.array):
    gp.fit(X_train, y_train)
    mean_prediction, std_prediction = gp.predict(X, return_std=True)
    noise_alpha = gp.alpha

    plt.plot(X, mean_prediction, label="Mean prediction")
    draw_ci(plt.gca(), X, mean_prediction, std_prediction)

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
    add_base_labels()


def plot_gp_botorch(model, train_x, train_y, bounds, batch=None, batch_y=None,x = None, y = None) -> None:
    test_x = torch.linspace(bounds[0, 0], bounds[1, 0], 500).unsqueeze(-1).double()
    
    with torch.no_grad():
        posterior = model(test_x)
        mean = model.outcome_transform.untransform(posterior.mean)[0].flatten()
       
        std = model.outcome_transform.untransform(posterior.variance.sqrt())

    sns.lineplot(x=test_x.numpy().flatten(), y=mean.numpy().flatten(), label='Mean')

    draw_ci(plt.gca(), test_x, mean, std)

    plt.plot(x, y, 
            'k--', label='True')
    plt.scatter(train_x.numpy(), train_y.numpy(), c='r', s=50, label='Observations')
    
    if batch is not None:

        sns.scatterplot(x=batch.flatten().numpy(), y=batch_y.flatten().numpy(), color=colors[1], s=150, marker='*', label='New candidates')

    add_base_labels()