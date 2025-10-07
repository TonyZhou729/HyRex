import jax.numpy as jnp
from jax import lax
import equinox as eqx


class array_with_padding(eqx.Module):

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
        return self.arr

    def concat(self,other_arr):
        """
        Concatenates self.arr with another array managed in another instnce of array_with_padding.  
        self.arr will appear first in the concatenation.  Padding size is updated to the padding
        of both arrays.
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