import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


class PlotterDataset():
    def plot_sample(self, sample, title='Scatter Plot of Sample', labels='data points'):
        try:
            dim = f'{len(sample[0])}D'
            return self.plots[dim](sample, title, labels)
        except ValueError as ve:
            print(f"Unsupported number of dimensions for plotting: {ve}")

    @staticmethod
    def plot_2D(sample, title, labels):
        fig, ax = plt.subplots(figsize=(6, 6))
        x, y = sample[:, 0], sample[:, 1]
        ax.scatter(x, y, c='blue', marker='o', label=labels)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title(title)
        ax.legend()
        plt.show()
        return

    @staticmethod
    def plot_3D(sample, title, labels):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = sample[:, 0], sample[:, 1], sample[:, 2]
        ax.scatter(x, y, z, c='blue', marker='o', label=labels)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(title)
        ax.legend()
        plt.show()
        return


class SphericalDataset(PlotterDataset):
    def __init__(self, figure, seed=42):
        self.figure = figure
        self.methods = {
            'circle' : self.get_sample_circle,
            'sphere' : self.get_sample_sphere
        }
        self.plots = {
            '2D' : PlotterDataset.plot_2D,
            '3D' : PlotterDataset.plot_3D
        }
        self.seed = seed
        np.random.seed(seed)

    
    def get_sample(self, n_samples, dim_space, first_radius=1, second_radius=2, noise=0):
        return self.methods[self.figure](n_samples, dim_space, first_radius, noise)

    
    def get_sample_circle(self, n_samples, dim_space, radius, noise):
        N = 2
        if dim_space < N:  raise ValueError("Attempt to collapse a figure of a given dimension into a space of lower dimension. Dude.")
        angles = np.random.uniform(0, 2*np.pi, n_samples)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        coords = np.column_stack([x, y]) + np.random.normal(0, noise, size=(n_samples, N))
        coords = np.column_stack([coords, np.zeros((n_samples, dim_space - 2))])
        return coords


    def get_sample_sphere(self, n_samples, dim_space, radius, noise):
        N = 3
        if dim_space < N:  raise ValueError("Attempt to collapse a figure of a given dimension into a space of lower dimension. Dude.")
        phi   = np.random.uniform(0, np.pi, n_samples)
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        coords = np.column_stack([x, y, z]) + np.random.normal(0, noise, size=(n_samples, N))
        coords = np.column_stack([coords, np.zeros((n_samples, dim_space - 3))])
        return coords
        
