import numpy as np
import torch
import matplotlib.pyplot as plt


def sincx(x):
    return torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.sin(x) / x)

def linear_func(x, slope=1.0, intercept=0.0):
    return slope * x + intercept

def poly_func(roots, coeffs, x):
    p = torch.poly1d(coeffs)
    return p(x)


if __name__ == "__main__":
    x = torch.linspace(-10, 10, 1000)
    y_sincx = sincx(x)
    y_linear = linear_func(x, slope=2.0, intercept=1.0)
    y_poly = poly_func(roots=[-3, 1, 2], coeffs=[1, -0.5, -4, 6], x=x)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x.numpy(), y_sincx.numpy())
    plt.title("Sinc Function")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(x.numpy(), y_linear.numpy())
    plt.title("Linear Function")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(x.numpy(), y_poly.numpy())
    plt.title("Polynomial Function")
    plt.grid()

    plt.tight_layout()
    plt.show()