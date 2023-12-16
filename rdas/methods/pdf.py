import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.optimize import approx_fprime

class KDEProbabilityDensityFunction:
    def __init__(self, data, kernel='gaussian'):
        self.data = data
        self.kernel = kernel
        self.kde_model = self._fit_kde()

    def _fit_kde(self):
        kde_model = KernelDensity(kernel=self.kernel)
        kde_model.fit(self.data)
        return kde_model

    def value(self, x):
        # Reshape x to match the input dimensions
        x = np.array(x).reshape(1, -1)
        return np.exp(self.kde_model.score_samples(x))

    def grad(self, x, epsilon=1e-5):
        x = np.array(x)

        def func(x):
            return np.log(self.value(x))

        # Use numerical approximation for gradient
        gradient = approx_fprime(x, func, epsilon)
        return gradient
