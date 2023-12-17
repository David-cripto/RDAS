import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path


PathLike = Path | str

TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(lambda x: torch.tensor(x, dtype = torch.float32))
    ]
)

class SphericalDataset():
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


class TorchSphereDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.sphere_generator = np.load(path)
        self.transform=TRANSFORM

    def __len__(self):
        return len(self.sphere_generator)

    def __getitem__(self, index):
        return self.transform(self.sphere_generator[index])


def get_dataset(dataset) -> tuple[Dataset, ...]:
    return TorchSphereDataset(dataset)

