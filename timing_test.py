from hyrex.hyrex import recomb_model
# from grad_err import get_history
import time
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

h = jnp.float64(0.67)
omega_b = jnp.float64(0.02236)
omega_cdm = jnp.float64(0.18)
Neff = jnp.float64(3.045)
YHe = jnp.float64(0.025)

recomb = recomb_model()

for i in range(5):
    start = time.time()
    # res = get_history(h,omega_b,omega_cdm,Neff,YHe)
    res = recomb(h,omega_b,omega_cdm,Neff,YHe)
    # print(res)
    # res.block_until_ready()
    print("HyRex passed in {0} seconds".format(time.time()-start))