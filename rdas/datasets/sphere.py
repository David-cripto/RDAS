import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path


class SphericalDataset(Dataset):
    def __init__(self, dim_of_space, dim_of_manifold, seed=42):
        self.n = dim_of_space
        self.m = dim_of_manifold
        self.seed = seed
        np.random.seed(seed)


    def get_sample(self, n_samples, radii=[1], noise=0):
        if self.m > self.n:  
            raise ValueError("Attempt to collapse a figure of a given dimension into a space of lower dimension. Dude.")
        coords = np.random.randn(n_samples, self.m)
        coords /= np.linalg.norm(coords, axis=1)[:, np.newaxis] 
        coords *= np.random.choice(radii, size=n_samples).reshape(-1, 1)
        coords = np.column_stack([coords, np.zeros((n_samples, self.n - self.m))])
        coords += noise * np.random.randn(n_samples, self.n)  # noise generated in every direction
        return coords


class TorchLinDataset(Dataset):
    def __init__(self, n_samples=10**5, **kwargs) -> None:
        super().__init__()
        self.line_generator = SphericalDataset(**kwargs).get_sample(n_samples)
        self.transform = transforms.Compose([
            transforms.Lambda(tensorize)
        ])

    def __len__(self):
        return len(self.line_generator)

    def __getitem__(self, index):
        return self.transform(self.line_generator[index])


def tensorize(x):
    return torch.tensor(x, dtype=torch.float32)
    
def get_dataset(**kwargs) -> tuple[Dataset, ...]:
    return TorchLinDataset(**kwargs)

