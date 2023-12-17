import torch
import numpy as np
    
class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50):
        """Construct a Variance Exploding SDE.

        Args:
        sigma_min: smallest sigma.
        sigma_max: largest sigma.
        N: number of discretization steps
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std