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
def tentmap(x: float | np.ndarray, map_args: Any):
    
    # Set default value
    if map_args is not None and 'mu' in map_args:
        mu = map_args['mu']
    else:
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
def logisticmap(x: float | np.ndarray, map_args: Any):

    # Set default value
    if map_args is not None and 'r' in map_args:
        r = map_args['r']
    else:
        r = 4
        
    return r * x * (1-x)

### Define the equation governing the time evolution of 2D states in the Anosov diffeomorphism.
# 
# args: X_n, 2D state vector representing cartesian (x, y) with elements in [0, 1)
#       A, linear operator on X_n which must satisfy det(A) == 1
#
# returns: 2D vector f(X) = X_(n+1)
# 
@nb.njit(fastmath=True)
def anosov_diffeo(X: np.ndarray, map_args: Any):

    # Set default value
    if map_args is not None:
        if 'A11' in map_args: A11 = map_args['A11']
        else: A11 = 2.0
        if 'A12' in map_args: A12 = map_args['A12']
        else: A12 = 1.0
        if 'A21' in map_args: A21 = map_args['A21']
        else: A21 = 1.0
        if 'A22' in map_args: A22 = map_args['A22']
        else: A22 = 1.0
    else:
        A11 = 2.0; A12 = 1.0; A21 = 1.0; A22 = 1.0

    A = np.array([[A11, A12],[A21, A22]], dtype=np.float64)

    assert np.linalg.det(A) == 1
    return (A@X)%1

### Define the equation governing the time evolution of 2D states of a Duffing oscillator.
#
# args: X_n, 2D state vector representing (x(t), dx(t)/dt) for t = t_n.
#       alpha, a scalar parameter controlling the linear stiffness of the oscillator
#       beta, a scalar parameter controlling the amount of nonlinearity in the restoring force. if beta = 0,
#             the Duffing equation describes a damped and driven simple harmonic oscillator.
#       delta, a scalar parameter controlling the amount of damping.
# 
# returns: 2D vector representing (dx(t)/dt, d2x(t)/dt2)
# 
# The second-order DE governing these dynamics, and the state-vector representation of that
# governing equation, are found in the notebook 2D_Duffing_Koopman.ipynb.
#
# This function should not be accessed outside of the scope of duffing_rk4, so I handle keyword arguments explicitly
# and assume their presence.
#
@nb.njit(fastmath=True)
def duffing_rhs(X: np.ndarray, alpha: float, beta: float, delta: float):

    assert X.shape[0] == 2
    Xdot = np.empty((2))
    Xdot[0] = X[1]; Xdot[1] = - delta * X[1] - beta * X[0] - alpha * X[0]**3
    return Xdot

### Define the integrator used to create the time-series data.
#
# This integrator represents a single integration step and will be called repeatedly from within
# DataGenerator class, when constructed to obey the dynamics of the Duffing oscillator. 
# 
# args: X_n, 2D state vector representing (x(t), dx(t)/dt)
#       dt, scalar timestep
# 
# returns: 2D vector X_(n+1) advanced from X_n by dt units using Runge-Kutta integrator.
# 
@nb.njit(fastmath=True)
def duffing_rk4(X: np.ndarray, map_args: Any):
    
    # Set default values
    if map_args is not None:
        if 'dt' in map_args: dt = map_args['dt']
        else: dt = 0.01
        if 'alpha' in map_args: alpha = map_args['alpha']
        else: alpha = 1.0
        if 'beta' in map_args: beta = map_args['beta']
        else: beta = -1.0
        if 'delta' in map_args: delta = map_args['delta']
        else: delta = 0.5
    else:
        dt = 0.01; alpha = 1.0; beta = -1.0; delta = 0.5

    assert X.shape[0] == 2
    k1 = duffing_rhs(X, alpha, beta, delta)
    k2 = duffing_rhs(X + 0.5 * dt * k1, alpha, beta, delta)
    k3 = duffing_rhs(X + 0.5 * dt * k2, alpha, beta, delta)
    k4 = duffing_rhs(X + dt * k3, alpha, beta, delta)

    return X + dt / 6 * (k1 + 2 * (k2 + k3) + k4)
    