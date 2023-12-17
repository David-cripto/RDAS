import numpy as np
import random
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
import torch

PathLike = Path | str
NUM_LIN_SAMPLES = 10**4
TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(lambda x: torch.tensor(x, dtype = torch.float32))
    ]
)

class DatasetGenerator:
    def __init__(self, dim_of_space, dim_of_manifold):
        self.dim_of_space = dim_of_space
        self.dim_of_manifold = dim_of_manifold

    def generate_dataset_in_subspace(self, n_samples, seed_value=42):
        """
        Generate a dataset that lies along a low-dimensional subspace within a high-dimensional space.

        Parameters:
        - num_samples: Number of data points to generate.
        - dim_of_space: Dimension of the high-dimensional space (L).
        - dim_of_subspace: Dimension of the low-dimensional subspace (l).

        Returns:
        - dataset: NumPy array with shape (num_samples, dim_of_space)
        """
        np.random.seed(seed_value)
        # Generate a random orthonormal basis for the subspace
        basis = np.linalg.qr(np.random.randn(self.dim_of_space, self.dim_of_manifold))[0]
        # Generate random coefficients for the linear combination
        coefficients = np.random.randn(n_samples, self.dim_of_manifold)
        # Compute the dataset by projecting random points onto the subspace
        dataset = np.dot(coefficients, basis.T)
        return dataset

    def generate_lines_planes(self, n_samples, scale, noise, line_params):
        """
        Generate a dataset of straight lines in either a 2D plane or 3D space.

        Parameters:
        - n_samples: Number of samples in the dataset.
        - plane: If True, generate lines in a 2D plane. If False, generate lines in 3D space.

        Returns:
        - dataset: NumPy array with shape (n_samples, dim_of_space)
        """
        if self.dim_of_space not in [2, 3]:
            raise ValueError("dim_of_space must be either 2 or 3.")
        dataset = np.zeros((n_samples, self.dim_of_space))
        if self.dim_of_space == 2:
            k, b = line_params
        else:
            if self.dim_of_manifold == 2: # plane in 3D space
                a, b, c = line_params
            if self.dim_of_manifold == 1:
                p_0, p_1, p_2, x_0, y_0, z_0 = line_params

        for i in range(n_samples):
            eps = noise * np.random.normal()
            # Generate random parameters for the line
            if self.dim_of_space == 2:
                x = scale * np.random.rand()
                y = k * x + b + eps
                dataset[i] = [x, y]
            else:
                if self.dim_of_manifold == 2:
                    x = scale * np.random.rand(self.dim_of_space - 1)
                    y = a * x[0] + b * x[1] + c + eps
                    dataset[i] = [x[0], x[1], y]
                if self.dim_of_manifold == 1:
                    t = scale * np.random.rand()
                    x = x_0 + p_0 * t + noise * np.random.normal()
                    y = y_0 + p_1 * t + noise * np.random.normal()
                    z = z_0 + p_2 * t + noise * np.random.normal()
                    dataset[i] = [x, y, z]
        return dataset
      

class TorchLinDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.line_generator = DatasetGenerator(**kwargs).generate_lines_planes(NUM_LIN_SAMPLES)
        self.transform=TRANSFORM

    def __len__(self):
        return len(self.line_generator)
    def __getitem__(self, index):
        return self.transform(self.line_generator[index])
    


def get_dataset(**kwargs) -> tuple[Dataset, ...]:
    return TorchLinDataset(**kwargs)
