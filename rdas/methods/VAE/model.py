import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Independent


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(in_dim, 2*in_dim),
            nn.BatchNorm1d(2*in_dim),
            nn.ReLU(),
            nn.Linear(2*in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x):
        return self.func(x) + self.residual(x)
        

class Encoder(nn.Module):
    def __init__(self, dim: int, latent_dim: int, intermediate_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            ResBlock(dim, dim),
            ResBlock(dim, intermediate_dim),
            ResBlock(intermediate_dim, intermediate_dim),
            ResBlock(intermediate_dim, intermediate_dim),
            ResBlock(intermediate_dim, 2*latent_dim),
        )
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, dim: int, latent_dim: int, intermediate_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            ResBlock(latent_dim, latent_dim),
            ResBlock(latent_dim, intermediate_dim),
            ResBlock(intermediate_dim, intermediate_dim),
            ResBlock(intermediate_dim, intermediate_dim),
            ResBlock(intermediate_dim, dim),
        )
    def forward(self, z):
        return self.model(z)
    
def get_normal_KL(mean_1, log_std_1, mean_2=None, log_std_2=None):
    """
        This function should return the value of KL(p1 || p2),
        where p1 = Normal(mean_1, exp(log_std_1)), p2 = Normal(mean_2, exp(log_std_2) ** 2).
        If mean_2 and log_std_2 are None values, we will use standard normal distribution.
        Note that we consider the case of diagonal covariance matrix.
    """
    if mean_2 is None:
        mean_2 = torch.zeros_like(mean_1)
    if log_std_2 is None:
        log_std_2 = torch.zeros_like(log_std_1)

    std_1 = torch.exp(log_std_1)
    std_2 = torch.exp(log_std_2)

    mean_1, mean_2 = mean_1.float(), mean_2.float()
    std_1, std_2  = std_1.float(), std_2.float()

    p  = Independent(torch.distributions.Normal(mean_1, std_1), 1)
    q  = Independent(torch.distributions.Normal(mean_2, std_2), 1)
    kl = torch.distributions.kl_divergence(p, q)

    return kl


def get_normal_nll(x, mean, log_std):
    """
        This function should return the negative log likelihood log p(x),
        where p(x) = Normal(x | mean, exp(log_std) ** 2).
        Note that we consider the case of diagonal covariance matrix.
    """
    # ====
    mean = mean.float()
    std  = torch.exp(log_std).float()

    prob = Independent(torch.distributions.Normal(mean, std), reinterpreted_batch_ndims = 1)
    nnl = -prob.log_prob(x)
    return nnl

class VAE(nn.Module):
    def __init__(self, enc, dec, n_latent, beta=1):
        super().__init__()
        # assert len(input_shape) == 3

        self.beta = beta
        self.n_latent = n_latent
      
        self.encoder = enc
        self.decoder = dec

    def prior(self, n, use_cuda=False):
     
        z = torch.randn(n, self.n_latent)
        if use_cuda:
            z = z.cuda()
        return z

    def forward(self, x):
        mu_z, log_std_z = torch.tensor_split(self.encoder(x), 2, dim = 1)
        # print(mu_z.shape, log_std_z.shape, self.prior(x.shape[0]).shape) # (100, 2) (100, 2) (100, 10)
        z = mu_z + torch.exp(log_std_z) * self.prior(x.shape[0])
        mu_x = self.decoder(z)
        return mu_z, log_std_z, mu_x
        
    def loss(self, x):

        mu_z, log_std_z, mu_x = self(x)
        recon_loss = torch.mean(get_normal_nll(x, mu_x, torch.zeros_like(mu_x)))
        kl_loss = torch.mean(get_normal_KL(mu_z, log_std_z, torch.zeros_like(mu_z), torch.zeros_like(log_std_z)))
        elbo_loss = self.beta * kl_loss + recon_loss
        dict_loss = {"recon_loss": recon_loss, "kl_loss":kl_loss, "elbo_loss":elbo_loss}
        return dict_loss

    def sample(self, n):
        with torch.no_grad():

            x_recon = self.decoder(self.prior(n))
            # samples = torch.clamp(x_recon, -1, 1)

        return x_recon.cpu().numpy()
