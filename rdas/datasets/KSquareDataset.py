import numpy as np
import matplotlib.pyplot as plt

class KSquareDataset:
    def __init__(self, ambient_dim=32, num_squares=1, square_size=3):
        self.ambient_dim = ambient_dim
        self.num_squares = num_squares
        self.square_size = square_size

    def generate_dataset(self, num_samples):
        dataset = []
        for _ in range(num_samples):
            # Initialize an empty image
            image = np.zeros((self.ambient_dim, self.ambient_dim))

            # Add squares to the image
            for _ in range(self.num_squares):
                center_x, center_y = np.random.randint(self.square_size, self.ambient_dim - self.square_size, size=2)
                brightness = np.random.uniform(0, 1)

                # Add the square to the image
                square = np.ones((self.square_size, self.square_size)) * brightness
                image[center_x:center_x + self.square_size, center_y:center_y + self.square_size] += square[:self.ambient_dim - center_x, :self.ambient_dim - center_y]

            # Clip the values to be in the range [0, 1]
            image = np.clip(image, 0, 1)

            dataset.append(image)

        return np.array(dataset)
