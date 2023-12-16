import torch
import torch.nn as nn

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)
    
class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, in_dim, dims=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
            marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
            channels: The number of channels for feature maps of each resolution.
            embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.lin1 = nn.Linear(in_dim, dims[0], bias=False)
        self.dense1 = Dense(embed_dim, dims[0])
        self.norm1 = nn.BatchNorm1d(num_features=dims[0])
        self.lin2 = nn.Linear(dims[0], dims[1], bias=False)
        self.dense2 = Dense(embed_dim, dims[1])
        self.norm2 = nn.BatchNorm1d(num_features=dims[1])
        self.lin3 = nn.Linear(dims[1], dims[2], bias=False)
        self.dense3 = Dense(embed_dim, dims[2])
        self.norm3 = nn.BatchNorm1d(num_features=dims[2])
        self.lin4 = nn.Linear(dims[2], dims[3], bias=False)
        self.dense4 = Dense(embed_dim, dims[3])
        self.norm4 = nn.BatchNorm1d(num_features=dims[3])    

        # Decoding layers where the resolution increases
        self.rlin4 = nn.Linear(dims[3], dims[2], bias=False)
        self.dense5 = Dense(embed_dim, dims[2])
        self.rnorm4 = nn.BatchNorm1d(num_features=dims[2])
        self.rlin3 = nn.Linear(dims[2] + dims[2], dims[1], bias=False)    
        self.dense6 = Dense(embed_dim, dims[1])
        self.rnorm3 = nn.BatchNorm1d(num_features=dims[1])
        self.rlin2 = nn.Linear(dims[1] + dims[1], dims[0], bias=False)    
        self.dense7 = Dense(embed_dim, dims[0])
        self.rnorm2 = nn.BatchNorm1d(num_features=dims[0])
        self.rlin1 = nn.Linear(dims[0] + dims[0], in_dim)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
  
    def forward(self, x, t): 
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.embed(t))    
        # Encoding path
        h1 = self.lin1(x)    
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.norm1(h1)
        h1 = self.act(h1)
        h2 = self.lin2(h1)
        h2 += self.dense2(embed)
        h2 = self.norm2(h2)
        h2 = self.act(h2)
        h3 = self.lin3(h2)
        h3 += self.dense3(embed)
        h3 = self.norm3(h3)
        h3 = self.act(h3)
        h4 = self.lin4(h3)
        h4 += self.dense4(embed)
        h4 = self.norm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.rlin4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.rnorm4(h)
        h = self.act(h)
        h = self.rlin3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.rnorm3(h)
        h = self.act(h)
        h = self.rlin2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.rnorm2(h)
        h = self.act(h)
        h = self.rlin1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None]
        return h