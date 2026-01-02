import jax.numpy as jnp
from jax.scipy.special import xlogy
import numpy as np # This can be deleted if I change rng in prepare_projection

def sfs_loglik(afs, esfs, sequence_length=None, theta=None):
    """
    This function evaluates the multinomial or Poisson log-likelihood of an
    observed site frequency spectrum (AFS) given an expected spectrum (ESFS).

    By default, the sequence length and mutation rate (theta) are None, indicating
    that the multinomial likelihood will be used. To use the Poisson likelihood, one must
    provide BOTH the sequence length and mutation rate (theta).

    Parameters
    ----------
    afs : array_like
        Observed allele frequency spectrum
    esfs : array_like
        Expected allele frequency spectrum. Must be the same shape as ``afs``
    sequence_length : int, optional
        Total number of sites in the sequence. Required if ``theta`` is given
    theta : float, optional
        Population-scaled mutation rate. If provided, a sequence length must also 
        be provided and the Poisson likelihood is used; 
        otherwise a multinomial likelihood is assumed. 

    Returns
    -------
    float
        Log-likelihood of the observed spectrum given the expected spectrum.

    Notes
    -----
    In tskit, given a tree sequence, to obtain the afs one can use the function
    ::
        afs = tree_sequence.allele_frequency_spectrum(*options)
    
    To obtain the esfs, with ``momi3`` one must first initialize an ExpectedSFS object
    with a ``demes`` demographic model and a dictionary of the number of samples used per population. 
    Then one would input a dictionary of parameter values into the Expected SFS object::
        ESFS_obj = demesinfer.sfs.ExpectedSFS(demes_model.to_demes(), num_samples=samples_per_population)
        params = {param_key: value}
        esfs = ESFS_obj(params)

        multinomial_loglik_value = sfs_loglik(afs, esfs)
        poisson_loglik_value = sfs_loglik(afs, esfs, sequence_length=1e8, theta=1e-8)

    To compute the gradient, one can use ``jax.grad`` or ``jax.value_and_grad``. 
    All loglikelihood functions are compatible with ``jax``.
    
    Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.
    
    See Also
    --------
    demesinfer.sfs.ExpectedSFS
    """
    afs = afs.flatten()[1:-1]
    esfs = esfs.flatten()[1:-1]
    
    if theta:
        assert(sequence_length)
        tmp = esfs * sequence_length * theta
        return jnp.sum(-tmp + xlogy(afs, tmp))
    else:
        return jnp.sum(xlogy(afs, esfs/esfs.sum()))

def prepare_projection(afs, afs_samples, sequence_length, num_projections, seed):
    """
    Creates the specified number of random projection vectors and appropriate inputs for 
    the Einstein summation for tensor operations that are used in ``ExpectedSFS.tensor_prod``.

    Parameters
    ----------
    afs : array_like
        Observed allele frequency spectrum
    esfs : array_like
        Expected allele frequency spectrum. Must be the same shape as ``afs``
    sequence_length : int, optional
        Total number of sites in the sequence. Required if ``theta`` is given
    num_projections : int
        Number of desired random projection vectors
    seed : int
        Seed for reproducibility

    Returns
    -------
    dict
        dictionary of random projection vectors
    str
        string containing axes names separated by commas for Einstein summation
    list
        list containing a dictionary specifying number of haploids per population and the afs.
        This list is a necessary input for the Einstein summation
        
    Notes
    -----
    proj_dict contains the random projection vectors that define the low-dimensional 
    subspace for approximating the full expected SFS, einsum_str is a string specifying 
    the Einstein summation for tensor operations, and input_arrays are preprocessed arrays 
    that serve as inputs to the jax.numpy.einsum call, optimized for JAX's just-in-time compilation
    
    Example:
    ::
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)

    Please refer to ``Random Projection`` section for a specific example, the above provided codes are just outlines of how to call on the functions.
    
    See Also
    --------
    demesinfer.loglik.sfs_loglik.prepare_projection
    """
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

def projection_sfs_loglik(esfs_obj, params, proj_dict, einsum_str, input_arrays, sequence_length=None, theta=None):
    """
    This function evaluates the **projected** multinomial or Poisson log-likelihood of an
    observed site frequency spectrum (AFS) given an expected spectrum (ESFS) via Einstein summation.

    By default, the sequence length and mutation rate (theta) are None, indicating
    that the multinomial likelihood will be used. To use the Poisson likelihood, one must
    provide BOTH the sequence length and mutation rate (theta).

    Parameters
    ----------
    esfs_obj : array_like
        An demesinfer.sfs.ExpectedSFS object
    params : dict
        a dictionary of model parameters and their values 
    proj_dict : dict 
        Dictionary of arrays that represent projection vectors
    einsum_str : string 
        Einstein summation string for projection
    input_arrays : array_like
        Input arrays for einsum operation, it must contain the original afs
    sequence_length : int, optional
        Total number of sites in the sequence. Required if ``theta`` is given
    theta : float, optional
        Population-scaled mutation rate. If provided, a sequence length must also 
        be provided and the Poisson likelihood is used, 
        otherwise a multinomial likelihood is assumed. 

    Returns
    -------
    float
        Log-likelihood of the projected observed spectrum given the projected expected spectrum.

    Notes
    -----
    proj_dict contains the random projection vectors that define the low-dimensional 
    subspace for approximating the full expected SFS, einsum_str is a string specifying 
    the Einstein summation for tensor operations, and input_arrays are preprocessed arrays 
    that serve as inputs to the jax.numpy.einsum call, optimized for JAX's just-in-time compilation

    Example:
    ::
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
        esfs_obj = ExpectedSFS(demo.to_demes(), num_samples=afs_samples)
        params = {param_key: val}
        projection_sfs_loglik(esfs_obj, params, proj_dict, einsum_str, input_arrays, sequence_length=None, theta=None)
    
    Internally this function will call on demesinfer.sfs.ExpectedSFS.tensor_prod, which performs the projection
    operations on the site frequency spectrum.

    Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.
    
    See Also
    --------
    demesinfer.sfs.ExpectedSFS
    demesinfer.sfs.ExpectedSFS.tensor_prod
    demesinfer.sfs.sfs_loglik.prepare_projection
    """
    result1 = esfs_obj.tensor_prod(proj_dict, params)
    result2 = jnp.einsum(einsum_str, *input_arrays)

    if theta:
        tmp = result1 * sequence_length * theta
        return jnp.sum(-tmp + xlogy(result2, tmp))
    else:
        return jnp.sum(xlogy(result2, result1/jnp.sum(result1)))