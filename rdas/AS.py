from typing import Callable
import numpy as np


def check_input(X):
    # X : ndarray of input samples
    
    assert len(X.shape) == 2 and type(X) == np.ndarray, 'X should be a 2d numpy array'
    N, d = X.shape
    return X, N, d

def sorted_eigh(C):
    e, W = np.linalg.eigh(C)
    e = abs(e)
    ind = np.argsort(e)
    e = e[ind[::-1]]
    W = W[:,ind[::-1]]
    s = np.sign(W[0,:])
    s[s==0] = 1
    W = W*s
    return e.reshape((e.size,1)), W

def get_n(eigh, eps=1e-8):
    ediff = np.fabs(np.diff(e.reshape((e.size,))))
    n = np.argmax(ediff)+1

    return n

def compute_AS(X: np.ndarray, f_deriv: Callable) -> np.ndarray:
    # X : ndarray - input samples
    # f_deriv : Callable - function derivative
    
    X, N, d = check_input(X)
    
    dF = np.array([f_deriv(x) for x in X])
    
    C = np.zeros((d, d))
    for df in dF:
        C += df@df.T
    C /= M

    eigh, W = sorted_eigh(C)

    return get_n(eigh)