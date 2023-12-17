from typing import Callable
import numpy as np

def check_input(X):
    # X : ndarray of input samples

    assert len(X.shape) == 2 and type(X) == np.ndarray, 'X should be a 2d numpy array'
    N, d = X.shape
    return X, N, d

def get_n(S, eps=1e-8):
    ediff = np.fabs(np.diff(S.reshape((S.size,))))
    n = np.argmax(ediff)+1

    return n

def compute_AS(X: np.ndarray, kde_pdf) -> np.ndarray:
    # X : ndarray - input samples
    # f_deriv : Callable - function derivative

    X, N, d = check_input(X)

    G = 1/np.sqrt(N) * np.column_stack([kde_pdf.grad(x) for x in dataset])
    U, S, Vh = np.linalg.svd(G)
    
    return get_n(S)
