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


from ax import plot
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

class HeteroNoiseSGP(SingleTaskGP):
    def __init__(self, train_X, train_Y, **kwargs) -> None:
        std_unique, std_all = self._calc_std(train_X, train_Y)
        

        likelihood = FixedNoiseGaussianLikelihood(noise=std_all)
        
        super().__init__(train_X, train_Y, likelihood=likelihood, **kwargs)

    def _calc_std(self, train_X, train_Y):
        unique_X, inverse = torch.unique(train_X, dim=0, return_inverse=True)
        flat_y = train_Y.view(-1)
        std_list = []
        for i, x_val in enumerate(unique_X):
            mask = (inverse == i)
            y_std = flat_y[mask].std()

            std_list.append(y_std)
        std_list = torch.tensor(std_list)
        std_unique = std_list
        std_for_X = std_unique[inverse].to(train_Y.dtype).to(train_Y.device)
        return std_unique, std_for_X
    

class HeteroWhiteSGP(HeteroNoiseSGP):
    def __init__(self, train_X, train_Y, quintile=0.05, **kwargs) -> None:
        if train_X.unique(dim=0).size(0) == train_X.size(0):
            raise ValueError("All training points are unique, no heteroscedasticity estimates possible")
        
        std_unique, std_all = self._calc_std(train_X, train_Y)
        p10 = torch.quantile(std_unique, quintile)


        hetero_noise = std_all.clamp_min(p10)
        white_noise = torch.nn.Parameter(torch.tensor(p10.item(), dtype=hetero_noise.dtype))

        uncertainty = hetero_noise + white_noise
        print(uncertainty)
        likelihood = FixedNoiseGaussianLikelihood(noise=uncertainty)


        SingleTaskGP.__init__(self, train_X, train_Y, likelihood=likelihood, **kwargs)