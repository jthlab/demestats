import jax.numpy as jnp

def icr_loglik(time, sample_config, params, icr_call, deme_names):
    """
    Compute the log-likelihood contribution from an ICR evaluation at the given
    times and sampling configuration.

    Parameters
    ----------
        time : ArrayLike
            One or more time points at which the ICR quantities are evaluated.
        sample_config : array of int
            A array giving the number of sampled haploids for each deme, ordered
            consistently with ``deme_names``.
        params : dict
            A dictionary of parameter values passed through to ``icr_call``.
        icr_call : Callable
            A callable object, typically an ``ICRCurve`` instance or compatible
            function, that accepts ``params``, ``t``, and ``num_samples`` and
            returns ICR-related quantities.
        deme_names : array of str
            The ordered deme names corresponding to entries in ``sample_config``.

    Returns
    -------
        Scalar
            The total log-likelihood, computed as the sum of
            ``log(result["c"]) + result["log_s"]`` over the given time points.

    Notes
    -----
    This function converts the positional sampling configuration into the deme-name
    mapping expected by ``icr_call``, evaluates the ICR quantities, and combines
    the returned components into a scalar log-likelihood.

    The callable ``icr_call`` is an ICR or CCR object that is expected to return a dictionary containing
    the entries ``"c"`` and ``"log_s"``. You may also pass in their respective mean-field objects,
    Any function that returns 'c' and 'log_s' will work. 

    ::
        icr_exact = ICRCurve(demo=g, k=2)

        ll = icr_loglik(
            time=jnp.array([10.0, 100.0, 1000.0]),
            sample_config=[2, 0],
            params={},
            icr_call=icr_exact,
            deme_names=["P0", "P1"],
        )

    See Also
    --------
    demestats.iicr.IICRCurve
    demestats.iicr.IICRCurve.__call__
    demestats.iicr.IICRMeanFieldCurve
    demestats.iicr.IICRMeanFieldCurve.__call__
    demestats.iicr.CCRCurve
    demestats.iicr.CCRCurve.__call__
    demestats.iicr.CCRMeanFieldCurve
    demestats.iicr.CCRMeanFieldCurve.__call__
    """
    ns = {name: sample_config[i] for i, name in enumerate(deme_names)}
    result = icr_call(params=params, t=time, num_samples=ns)
    # jax.debug.print("result: {}", result)
    return jnp.sum(jnp.log(result["c"]) + result["log_s"])