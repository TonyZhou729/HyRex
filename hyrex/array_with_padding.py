import jax.numpy as jnp
from jax import lax
import equinox as eqx


class array_with_padding(eqx.Module):
    """
    Array container with automatic padding management.

    Manages arrays with trailing infinite padding, tracking the last
    valid element for efficient concatenation operations.

    Attributes:
    -----------
    arr : array
        Full array including padding elements
    padding_size : int
        Number of infinite padding elements at end
    lastnum : int
        Index of last valid (non-infinite) element
    lastval : float
        Value of last valid element
    """

    arr : jnp.array
    padding_size : int
    lastnum : int
    lastval : jnp.float64

    def __init__(self,arr):
        self.arr = arr

        self.lastnum = jnp.argmax(jnp.isinf(arr)*1)-1
        self.lastval = arr[self.lastnum]
        self.padding_size = arr.size-jnp.argmax(jnp.isinf(arr)*1)

    def __call__(self):
        """
        Return the full array including padding.

        Returns:
        --------
        array
            Complete array with padding elements
        """
        return self.arr

    def concat(self,other_arr):
        """
        Concatenate with another padded array.

        Combines two padded arrays by removing padding from the first array
        and appending the second array, then recomputing padding length.

        Parameters:
        -----------
        other_arr : array_with_padding
            Second array to concatenate after this one

        Returns:
        --------
        array_with_padding
            New padded array containing concatenated data
        """

        if not isinstance(other_arr, array_with_padding):
            raise TypeError("Can only concatenate with another array_with_padding instance.")
        
        x = self.arr
        y = other_arr.arr
        padding_size = self.padding_size
        z = jnp.ones(x.size + y.size)*jnp.inf # neither of these is a tracer!!!
        z = z.at[0:x.size].set(x)
        concatenated_arr = lax.dynamic_update_slice(z,y,[x.size-padding_size])
        return array_with_padding(concatenated_arr)