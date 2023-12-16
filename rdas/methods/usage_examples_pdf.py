import numpy as np
from pdf import KDEProbabilityDensityFunction

# Example usage

data = np.random.randn(100, 2)  # Example 2D data
kde_pdf = KDEProbabilityDensityFunction(data, kernel='gaussian')
# Example of computing the value and gradient at a point
point = np.array([0, 0])
pdf_value = kde_pdf.value(point)
pdf_gradient = kde_pdf.grad(point)

print(f"PDF Value at {point}: {pdf_value}")
print(f"PDF Gradient at {point}: {pdf_gradient}")
