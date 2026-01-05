from typing import Any, List, Mapping, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from demestats.fit.util import (
    _dict_to_vec,
    _vec_to_dict,
    _vec_to_dict_jax,
    create_inequalities,
    make_whitening_from_hessian,
    pullback_objective,
)
from demestats.loglik.sfs_loglik import (
    prepare_projection,
    projection_sfs_loglik,
    sfs_loglik,
)
from demestats.sfs import ExpectedSFS

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]


def _compute_sfs_likelihood(vec, args_nonstatic, args_static):
    """
    Compute negative log-likelihood for SFS parameters for a given
    parameter vector. Supports both full SFS computation and projected
    approximations.

    Parameters
    ----------
    vec : jax.Array
        Parameter vector to evaluate.
    args_nonstatic : tuple
        Tuple containing non-static arguments:
        (path_order, proj_dict, input_arrays, sequence_length, theta, projection, afs)
    args_static : tuple
        Tuple containing static (compile-time) arguments:
        (esfs_obj, einsum_str)

    Returns
    -------
    float
        Negative log-likelihood value.

    Notes
    -----
    This function is JIT-compiled and used as the core objective function
    for optimization. When projection is enabled, it uses random projections
    to reduce computational cost. Debug prints are included for parameter
    and loss tracing during optimization.

    See Also
    --------
    demestats.fit.fit_sfs.fit
    """
    (path_order, proj_dict, input_arrays, sequence_length, theta, projection, afs) = (
        args_nonstatic
    )
    (esfs_obj, einsum_str) = args_static
    params = _vec_to_dict_jax(vec, path_order)
    jax.debug.print("Params: {params}", params=vec)

    if projection:
        loss = -projection_sfs_loglik(
            esfs_obj,
            params,
            proj_dict,
            einsum_str,
            input_arrays,
            sequence_length,
            theta,
        )
        jax.debug.print("Loss: {loss}", loss=loss)
        return loss
    else:
        esfs = esfs_obj(params)
        loss = -sfs_loglik(afs, esfs, sequence_length, theta)
        jax.debug.print("Loss full sfs: {loss}", loss=loss)
        return loss


def neg_loglik(vec, g, preconditioner_nonstatic, args_nonstatic, lb, ub):
    """
    Wrapper function that checks parameter boundaries before evaluating
    the objective function. Returns infinite loss for parameters outside
    bounds to enforce constraints.

    Parameters
    ----------
    vec : jax.Array
        Parameter vector in transformed space.
    g : callable
        Pullback objective function to evaluate.
    preconditioner_nonstatic : tuple
        Preconditioner arguments (x0, LinvT) for transforming parameters.
    args_nonstatic : tuple
        Non-static arguments for the objective function.
    lb : jax.Array
        Lower bounds in transformed space.
    ub : jax.Array
        Upper bounds in transformed space.

    Returns
    -------
    tuple
        (loss, gradient) where loss is the negative log-likelihood or
        infinity for invalid parameters, and gradient is a vector of
        penalty gradients for boundary violations.

    Notes
    -----
    This function serves as the interface between scipy.optimize and
    JAX-compiled functions. It provides gradient information for
    boundary violations to guide optimization back to feasible regions.

    See Also
    --------
    demestats.fit.fit_sfs.fit
    """
    if jnp.any(vec >= ub):
        return jnp.inf, jnp.full_like(vec, 1e10)

    if jnp.any(vec <= lb):
        return jnp.inf, jnp.full_like(vec, -1e10)

    return g(vec, preconditioner_nonstatic, args_nonstatic)


def fit(
    demo,
    paths: Params,
    afs,
    afs_samples,
    cons,
    lb,
    ub,
    *,
    method: str = "trust-constr",
    sequence_length: float = None,
    theta: float = None,
    projection: bool = False,
    num_projections: float = 200,
    seed: float = 5,
    gtol: float = 1e-5,
    xtol: float = 1e-5,  # default 1e-8
    maxiter: int = 1000,  # default 1000
    barrier_tol: float = 1e-5,
):
    """
    Fit demographic model parameters using SFS likelihood optimization.
    Main optimization function that estimates demographic parameters by
    maximizing the likelihood of the observed allele frequency spectrum.
    Supports both full SFS computation and accelerated projected methods.

    Parameters
    ----------
    demo : demes.Graph
        ``demes`` model graph.
    paths : Params
        Parameter paths to optimize. Each path specifies a demographic
        parameter in the model.
    afs : array_like
        Observed allele frequency spectrum.
    afs_samples : dictionary
        Dictionary specifying the number of haploids in each population for the afs
    cons : dict
        Dictionary containing equality and inequality constraints.
        Expected keys: 'eq' for (Aeq, beq) equality constraints Aeq@x = beq,
        and 'ineq' for (G, h) inequality constraints G@x <= h.
    lb : array_like
        Lower bounds for parameters.
    ub : array_like
        Upper bounds for parameters.
    method : str, optional
        Optimization method (default: "trust-constr").
    sequence_length : float, optional
        Sequence length. Required for Poisson likelihood when theta is given.
    theta : float, optional
        Population-scaled mutation rate. If provided, Poisson likelihood
        is used instead of multinomial.
    projection : bool, optional
        Whether to use random projections for acceleration (default: False).
    num_projections : int, optional
        Number of random projections to use if projection=True (default: 200).
    seed : int, optional
        Random seed for projection matrix generation (default: 5).
    gtol : float, optional
        Gradient tolerance for convergence (default: 1e-5).
    xtol : float, optional
        Parameter tolerance for convergence (default: 1e-5).
    maxiter : int, optional
        Maximum number of iterations (default: 1000).
    barrier_tol : float, optional
        Barrier tolerance for interior-point methods (default: 1e-5).

    Returns
    -------
    tuple
        (params_opt, opt_value, x_opt) where:
        - params_opt: Dictionary of optimized parameters
        - opt_value: Optimal negative log-likelihood value
        - x_opt: Optimized parameter vector

    Notes
    -----
    This function implements a sophisticated optimization pipeline:
    1. Parameter space transformation using Hessian-based whitening
    2. Constraint handling with equality and inequality constraints
    3. Optional random projections for computational efficiency
    4. Boundary enforcement with penalty gradients

    The optimization is performed in a transformed space where the Hessian
    is approximately identity, improving convergence rates.
    """
    path_order: List[Var] = list(paths)
    x0 = _dict_to_vec(paths, path_order)
    x0 = jnp.array(x0)
    lb = jnp.array(lb)
    ub = jnp.array(ub)
    afs = jnp.array(afs)

    esfs_obj = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(
            afs, afs_samples, sequence_length, num_projections, seed
        )
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    args_nonstatic = (
        path_order,
        proj_dict,
        input_arrays,
        sequence_length,
        theta,
        projection,
        afs,
    )
    args_static = (esfs_obj, einsum_str)
    L, LinvT = make_whitening_from_hessian(
        _compute_sfs_likelihood, x0, args_nonstatic, args_static
    )
    preconditioner_nonstatic = (x0, LinvT)
    g = pullback_objective(_compute_sfs_likelihood, args_static)
    y0 = np.zeros_like(x0)

    lb_tr = L.T @ (lb - x0)
    ub_tr = L.T @ (ub - x0)

    linear_constraints: list[LinearConstraint] = []

    Aeq, beq = cons["eq"]
    A_tilde = Aeq @ LinvT
    b_tilde = beq - Aeq @ x0
    if Aeq.size:
        linear_constraints.append(LinearConstraint(A_tilde, b_tilde, b_tilde))

    G, h = cons["ineq"]
    if G.size:
        linear_constraints.append(create_inequalities(G, h, LinvT, x0))

    res = minimize(
        fun=neg_loglik,
        x0=y0,
        jac=True,
        args=(g, preconditioner_nonstatic, args_nonstatic, lb_tr, ub_tr),
        method=method,
        constraints=linear_constraints,
        options={
            "gtol": gtol,
            "xtol": xtol,
            "maxiter": maxiter,
            "barrier_tol": barrier_tol,
        },
    )

    x_opt = np.array(x0) + LinvT @ res.x
    print("optimal value: ")
    print(x_opt)
    print(res)

    return _vec_to_dict(jnp.asarray(x_opt), path_order), res.fun, x_opt
