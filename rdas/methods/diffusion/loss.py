import torch

def loss_fn(sde, model, x, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=x.device) * (sde.T - eps) + eps  
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, random_t)
    perturbed_x = mean + z * std[:, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=(1,2,3)))
    return loss