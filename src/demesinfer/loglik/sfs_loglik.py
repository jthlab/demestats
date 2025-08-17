import jax.numpy as jnp
from jax.scipy.special import xlogy

def sfs_loglik(afs, esfs, sequence_length=None, theta=None):
    afs = afs.flatten()[1:-1]
    esfs = esfs.flatten()[1:-1]
    
    if theta:
        assert(sequence_length)
        tmp = esfs * sequence_length * theta
        return jnp.sum(-tmp + xlogy(afs, tmp))
    else:
        return jnp.sum(xlogy(afs, esfs/esfs.sum()))