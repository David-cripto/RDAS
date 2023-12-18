# RDAS
## Reduction of Dimensionality through Active Subspaces approach. 

Manifold hypothesis states, that data points in high-dimensional space $N$ actually lie in close vicinity of a manifold of much lower dimension $n$. Thus in many cases, we can encode information about data in a smaller space, which will give us huge advantages in speed and memory consumption for storing the dataset. However it doesn't obvious how to estimate $n$ in order to not loose capacity of data and save as much as possible in resources.

Our project provides a method to estimate such n. We plan to create a new evaluation metric based in VAE with interpreted intuition and other metrics based on specific datasets.

In order to identify $n$ the method of active subspaces is used, which is extended by applying deterministic kernels and diffusion neural networks. The correctness of the proposed algorithm will be tested on MNIST dataset. Also we plan to create a synthetic multi-dimensional datasets with points, generated in order to test performance of our algorithm and visualize it.

## Concept

![alt text](https://github.com/David-cripto/RDAS/blob/VAE/pict/concept.png)

## Active Subspaces approach.
- Choose $m$, the number of estimations. This hyperparameter stands for the number of Monte Carlo estimations. The larger $m$, the more accurate the result is.
- Draw samples $\{x_i\}^m_{i=1}$ from $X$ (according to some prior probability density function).
- For each $x_i$ compute $\nabla f(x_i)$.
- Compute the SVD of the matrix:

$$
G := \frac{1}{\sqrt{m}}(\nabla f(x_1) \space \nabla f(x_2) \ldots \space \nabla f(x_m)) \approx U \Sigma V^*
$$

- Estimate the rank of $G\approx U_r \Sigma_rV^*_r$. The rank $r$ of the matrix G is the dimensionality of the active subspace. 
- Low-dimensional vectors are estimated as $x_{\mathrm{AS}} = U_r^*x$.

For further details, look into the book „Active Subspaces: Emerging Ideas in Dimension Reduction for Parameter Studies“ (2015) by Paul Constantine.

## Density distribution gradients obtained from diffusion models. 

![Alt Text](https://github.com/David-cripto/RDAS/blob/VAE/pict/grad.gif)

<!-- ## Results for MNIST zeros
![Alt Text](https://github.com/David-cripto/RDAS/blob/VAE/pict/zeros.gif) -->
