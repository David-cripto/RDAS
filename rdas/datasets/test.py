from dataset import DatasetGenerator
import matplotlib.pyplot as plt
import numpy as np

# 1 Example usage
dim_of_space = 2  
dim_of_manifold = 1
n_samples = 1000
line_generator = DatasetGenerator(dim_of_space, dim_of_manifold)
dataset = line_generator.generate_lines_planes(n_samples)
print(dataset.shape)

# 2 Example
dim_of_space = 3  
dim_of_manifold = 2
n_samples = 10000
plane_generator = DatasetGenerator(dim_of_space, dim_of_manifold)
dataset = plane_generator.generate_lines_planes(n_samples)
print(dataset.shape)

# 3 Example
dim_of_space = 3  
dim_of_manifold = 1
n_samples = 1000

plane_generator = DatasetGenerator(dim_of_space, dim_of_manifold)
dataset = plane_generator.generate_lines_planes(n_samples)
print(dataset.shape)

# 4 Example usage:
n_samples = 100
dim_of_space = 50
dim_of_manifold = 5
plane_generator = DatasetGenerator(dim_of_space, dim_of_manifold)
dataset = plane_generator.generate_dataset_in_subspace(n_samples)
print(dataset.shape)  # (100, 50)
rank_check = np.linalg.matrix_rank(dataset) == dim_of_manifold
print("Dimensionality Check:", rank_check)