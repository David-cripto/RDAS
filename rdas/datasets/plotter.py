import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


class PlotterDataset():
    def __init__(self):
        self.plots = {
            '2D' : self.plot_2D,
            '3D' : self.plot_3D
        }

    def plot_sample(self, sample, title='Scatter Plot of Sample', labels='data points'):
        try:
            dim = f'{len(sample[0])}D'
            return self.plots[dim](sample, title, labels)
        except KeyError as ke:
            raise KeyError(f"Unsupported number of dimensions for plotting: {ke}")


    @staticmethod
    def plot_2D(sample, title, labels):
        fig, ax = plt.subplots(figsize=(4, 4))
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
        fig = plt.figure(figsize=(4, 4))
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

