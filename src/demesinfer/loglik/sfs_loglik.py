import jax.numpy as jnp
from jax.scipy.special import xlogy
import numpy as np # This can be deleted if I change rng in prepare_projection

def sfs_loglik(afs, esfs, sequence_length=None, theta=None):
    afs = afs.flatten()[1:-1]
    esfs = esfs.flatten()[1:-1]
    
    if theta:
        assert(sequence_length)
        tmp = esfs * sequence_length * theta
        return jnp.sum(-tmp + xlogy(afs, tmp))
    else:
        return jnp.sum(xlogy(afs, esfs/esfs.sum()))

def prepare_projection(afs, afs_samples, sequence_length, num_projections, seed):
    rng = np.random.default_rng(seed)
    proj_dict = {}
    pop_names = list(afs_samples.keys())
    n_dims = afs.ndim
    
    for i in range(n_dims):
        if sequence_length is None:
            # Multinomial
            proj_dict[pop_names[i]] = rng.integers(0, 2, size=(num_projections, afs.shape[i]), dtype=jnp.int32)

            # uniform0 = rng.uniform(0, 1, size=(num_projections, afs.shape[i]))
            # proj_dict[pop_names[i]] = (uniform0 / uniform0.sum(axis=1, keepdims=True)).astype(jnp.float32)
        else:
            proj_dict[pop_names[i]] = rng.integers(0, 2, size=(num_projections, afs.shape[i]), dtype=jnp.int32)

    # Ask JT if it's fine to leave it like this, don't fix it if it didn't break? :)
    input_subscripts = ",".join([f"z{chr(97+i)}" for i in range(n_dims)])  # "za,zb,zc"
    tensor_subscript = "".join([chr(97+i) for i in range(n_dims)])         # "abc"
    output_subscript = "z"                                                 # "z"
    einsum_str = f"{input_subscripts},{tensor_subscript}->{output_subscript}"
    input_arrays = [proj_dict[pop_names[i]] for i in range(n_dims)] + [afs]

    return proj_dict, einsum_str, input_arrays

def projection_sfs_loglik(esfs, params, proj_dict, einsum_str, input_arrays, sequence_length=None, theta=None):
    result1 = esfs.tensor_prod(proj_dict, params)
    result2 = jnp.einsum(einsum_str, *input_arrays)

    if theta:
        tmp = result1 * sequence_length * theta
        return jnp.sum(-tmp + xlogy(result2, tmp))
    else:
        return jnp.sum(xlogy(result2, result1/jnp.sum(result1)))