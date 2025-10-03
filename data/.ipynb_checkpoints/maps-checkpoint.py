######
# 
# DT 10-03-2025
# 
# Define governing equations for various dynamical systems.
# Import this to use definitions standalone, or initialize DataGenerator object with
# argument map: str = $NAME_OF_FUNCTION and optional argument map_args: Any = {parameter set for map}
# to generate time-series data whose evolution is governed by the equation given in NAME_OF_FUNCTION().
# If map_args is not passed, the default parameter value(s) will be taken, e.g., \mu = 2 for tentmap,
# or r = 4 for logisticmap. 
# 
######

import numpy as np
import numba as nb
from typing import Any

### Define the equation governing the time evolution of 1D states under tent map.
# 
# args: x_n, scalar \in [0,1]
#       mu scalar
#
# returns: scalar f(x) = x_(n+1)
# 
@nb.njit(fastmath=True)
def tentmap(x: float | np.ndarray, mu: Any):
    
    # Set default value
    if mu is None:
        mu = 1

    return mu * np.min(np.array([x, 1-x]))

### Define the equation governing the time evolution of 1D states under logistic map.
# 
# args: x_n, scalar \in [0,1] or vector with elements \in [0,1]
#       r scalar
# 
# returns: scalar f(x) = x_(n+1), or vector f(X) = X_(n+1)
# 
@nb.njit(fastmath=True)
def logisticmap(x: float | np.ndarray, r: Any):

    # Set default value
    if r is None:
        r = 4
        
    return r * x * (1-x)

### Define the equation governing the time evolution of 2D states in the Anosov diffeomorphism.
# 
# args: X_n, 2D vector representing cartesian (x, y) with elements in [0, 1)
#       A, linear operator on X_n which must satisfy det(A) == 1
#
# returns: 2D vector f(X) = X_(n+1)
# 
@nb.njit(fastmath=True)
def anosov_diffeo(X: np.ndarray, A: Any):
    
    # Set default value
    if A is None:
        A = np.array([[2,1],[1,1]], dtype=np.float64)
        
    assert np.linalg.det(A) == 1
    return (A@X)%1