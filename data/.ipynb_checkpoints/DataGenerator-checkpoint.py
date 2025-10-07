######
# 
# DT 10-03-2025
# 
# Define DataGenerator class to unify instances of map function and its parameters.
# Each initialized DataGenerator represents a single map function for maps defined in data.maps,
# and carries along with it a static parameter set for the map function (passed at initialization).
# 
# Use DataGenerator.generate_next() to advance a passed state Xn *one time*, using the parameterized map function.
#     returns Xn+1, of dimensions Xn.shape.
# 
# Use DataGenerator.generate_series() to advance an initial state X0 $num_steps times, using the parameterized map function.
#     returns X, a (num_steps * X0.shape) np.ndarray with axis 0 iterating over timesteps i \in num_steps, and all other axes
#                representing the dimension of the state at timestep i. 
# 
######

import numpy as np
import numba as nb
from typing import Callable, Any
from data import maps

class DataGenerator(object):

    def __init__(self, X0: float | np.ndarray = 0.1,
                 num_steps: int = 100,
                 mapname: str = "anosov_diffeo",
                 map_args: Any = None):

        self.X0 = np.array(X0)
            
        self.num_steps = num_steps
        
        try:
            self.map = getattr(maps, mapname)
        except AttributeError:
            print(f'Name {map} is not defined in maps.py!')

        self.map_args = map_args

    @staticmethod
    @nb.njit(fastmath=True)
    def evolve(Xn: float | np.ndarray,
               mapfn: Callable[[Any], Any],
               map_args: Any):
        
        return mapfn(Xn, map_args)

    @staticmethod
    @nb.njit(fastmath=True)
    def evolve_all(X0: np.ndarray,              # Initial condition. If X0 passed as float to constructor, it is converted to (1,) np.ndarray. 
                   X: np.ndarray,               # Empty, but initialized, data array.
                   num_steps: int,              # Number of steps to iterate with dyanmics governed by mapfn.
                   mapfn: Callable[[Any], Any], # The map used to advance states in time, which is an attribute of data.maps.
                   map_args: Any):              # Any additional arguments are passed to mapfn.
        
        # Create data array of dimensions ( num_steps * state_dimensions ): 0 axis indexes over timesteps.
        X[0] = X0
        for i in range(num_steps - 1):
            X[i+1] = mapfn(X[i], map_args)

        return X
        
    # External interfaces
    def generate_next(self, Xn: np.ndarray):
        return self.evolve(Xn, self.map, self.map_args)
    
    def generate_series(self, X0: np.ndarray = None, num_steps: int = None):
        if X0 is None: X0 = self.X0
        if num_steps is None: num_steps = self.num_steps

        X = np.empty((num_steps, *X0.shape))
        
        return self.evolve_all(X0, X, num_steps, self.map, self.map_args)
        