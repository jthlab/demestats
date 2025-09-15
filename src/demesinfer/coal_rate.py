import jax.numpy as jnp

from .pexp import PConst


# FIXME this is a shim, need to remove all references to PiecewiseConstant and
# just use PConst directly
def PiecewiseConstant(c, t):
    return PConst(1 / 2 / c, jnp.append(t, jnp.inf))
