from operator import inv
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
import torch


class GammaNoiseSGP(SingleTaskGP):
    """Just add a lot of assumed noise"""

    def __init__(
        self,
        train_X,
        train_Y,
        noise_concentration: float = 10,
        noise_rate: float = 10,
    ):

        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(noise_concentration, noise_rate)
        )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
        )

class WhiteNoiseSGP(SingleTaskGP):
    def __init__(self, train_X, train_Y, white_noise: float = 1, **kwargs) -> None:
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(1.0, 1.0),
            noise_constraint=torch.distributions.constraints.greater_than(white_noise)
        )
        super().__init__(train_X, train_Y, likelihood=likelihood, **kwargs)


from ax import plot
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

class HeteroNoiseSGP(SingleTaskGP):
    def __init__(self, train_X, train_Y, **kwargs) -> None:
        std_unique, std_all, n_counts, inverse = self._calc_std(train_X, train_Y)

        if all(n_counts == 1):
            print("Warning: All points have only one repeat. Consider using WhiteNoiseSGP instead.")
            SingleTaskGP.__init__(self, train_X, train_Y, **kwargs)
        

        likelihood = FixedNoiseGaussianLikelihood(noise=std_all)


        
        super().__init__(train_X, train_Y, likelihood=likelihood, **kwargs)

    def _calc_std(self, train_X, train_Y):
        unique_X, inverse, n_counts = torch.unique(train_X, dim=0, return_inverse=True, return_counts=True)
        flat_y = train_Y.view(-1)
        std_list = []
        for i, x_val in enumerate(unique_X):
            mask = (inverse == i)
            y_std = flat_y[mask].std()

            std_list.append(y_std)
        std_list = torch.tensor(std_list)
        std_unique = std_list
        std_for_X = std_unique[inverse].to(train_Y.dtype).to(train_Y.device)
        return std_unique, std_for_X, n_counts, inverse
    

    

class HeteroWhiteSGP(HeteroNoiseSGP):
    def __init__(self, train_X, train_Y, quintile=0.05, **kwargs) -> None:

        std_unique, std_for_X, n_counts, inverse = self._calc_std(train_X, train_Y)

        if all(n_counts == 1):
            print("Warning: All points have only one repeat. Consider using WhiteNoiseSGP instead.")
            SingleTaskGP.__init__(self, train_X, train_Y, **kwargs)

        lower_noise = torch.quantile(std_unique, quintile)

        median_noise = torch.median(std_unique)
        hetero_noise = std_for_X.clamp_min(lower_noise)

        # Set noise to median noise for points with only one repeat
        single_repeat_mask = n_counts == 1
        hetero_noise[torch.isin(inverse, torch.where(single_repeat_mask)[0])] = median_noise

        white_noise = torch.nn.Parameter(torch.tensor(lower_noise.item(), dtype=hetero_noise.dtype))



        uncertainty = hetero_noise + white_noise
        likelihood = FixedNoiseGaussianLikelihood(noise=uncertainty)


        SingleTaskGP.__init__(self, train_X, train_Y, likelihood=likelihood, **kwargs)
