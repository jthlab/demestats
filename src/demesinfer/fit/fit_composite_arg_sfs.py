from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import jax
import jax.numpy as jnp
from scipy.optimize import LinearConstraint, minimize
from jax import vmap
from demesinfer.iicr import IICRCurve
import numpy as np
from demesinfer.fit.fit_arg import _compute_arg_likelihood
from demesinfer.fit.fit_sfs import _compute_sfs_likelihood
from demesinfer.sfs import ExpectedSFS
from demesinfer.loglik.sfs_loglik import prepare_projection

from loguru import logger
logger.disable("demesinfer")


from demesinfer.fit.util import _dict_to_vec, _vec_to_dict_jax, _vec_to_dict, create_inequalities, make_whitening_from_hessian, pullback_objective, process_arg_data, reformat_data
Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def _compute_composite_arg_sfs_likelihood(vec, path_order, esfs, proj_dict, einsum_str, input_arrays, sequence_length, theta, projection, afs, unique_cfg, deme_names, iicr_call, rho, unique_times, group_membership, associated_indices, batch_size, chunking_length, rearranged_data, new_max_indices, n1, n2):
    # params = _vec_to_dict_jax(vec, path_order)
    jax.debug.print("Params: {params}", params=vec)
    
    arg_loss = _compute_arg_likelihood(vec, path_order, unique_cfg, deme_names, iicr_call, rho, unique_times, group_membership, associated_indices, batch_size, chunking_length, rearranged_data, new_max_indices)
    sfs_loss = _compute_sfs_likelihood(vec, path_order, esfs, proj_dict, einsum_str, input_arrays, sequence_length, theta, projection, afs)
    loss = arg_loss/n1 + sfs_loss/n2
    jax.debug.print("Loss: {loss}", loss=loss)
    return loss

def neg_loglik(vec, g, lb, ub):
    if jnp.any(vec > ub) or jnp.any(vec < lb):
        return jnp.inf, jnp.full_like(vec, jnp.inf)

    return g(vec)

def fit(
    demo,
    paths: Params,
    data_list,
    cfg_list,
    afs,
    afs_samples,
    cons,
    lb,
    ub,
    *,
    method: str = "trust-constr",
    recombination_rate = 1e-8,
    sequence_length = None,
    num_projections = 200,
    projection = False,
    theta = None,
    seed: float = 5, 
    gtol: float = 1e-8,
    xtol: float = 1e-8, #default 1e-8
    maxiter: int = 1000, #default 1000
    barrier_tol: float = 1e-8,
):
    path_order: List[Var] = list(paths)
    x0 = _dict_to_vec(paths, path_order)
    x0 = jnp.array(x0)
    lb = jnp.array(lb)
    ub = jnp.array(ub)
    data_pad, deme_names, max_indices, unique_cfg, matching_indices = process_arg_data(data_list, cfg_list)
    chunking_length = data_pad.shape[1]
    unique_times, rearranged_data, new_matching_indices, new_max_indices, associated_indices, unique_groups, batch_size, group_membership = reformat_data(data_pad, matching_indices, max_indices, chunking_length)

    num_samples = len(max_indices)
    rho = recombination_rate = 1e-8
    iicr = IICRCurve(demo=demo, k=2)
    iicr_call = iicr.curve

    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None
        
    n1 = _compute_arg_likelihood(x0, path_order, unique_cfg, deme_names, iicr_call, recombination_rate, unique_times, group_membership, associated_indices, batch_size, chunking_length, rearranged_data, new_max_indices)
    n2 = _compute_sfs_likelihood(x0, path_order, esfs, proj_dict, einsum_str, input_arrays, sequence_length, theta, projection, afs)

    args = (path_order, esfs, proj_dict, einsum_str, input_arrays, sequence_length, theta, projection, afs, unique_cfg, deme_names, iicr_call, rho, unique_times, group_membership, associated_indices, batch_size, chunking_length, rearranged_data, new_max_indices, n1, n2)
    L, LinvT = make_whitening_from_hessian(_compute_composite_arg_sfs_likelihood, x0, *args)
    g = pullback_objective(_compute_composite_arg_sfs_likelihood, x0, LinvT, *args)
    y0 = np.zeros_like(x0)

    lb_tr = L.T @ (lb - x0)
    ub_tr = L.T @ (ub - x0)

    linear_constraints: list[LinearConstraint] = []
    
    Aeq, beq = cons["eq"]
    A_tilde = Aeq @ LinvT
    b_tilde = beq - Aeq@x0
    if Aeq.size:
        linear_constraints.append(LinearConstraint(A_tilde, b_tilde, b_tilde))

    G, h = cons["ineq"]
    if G.size:
        linear_constraints.append(create_inequalities(G, h, LinvT, x0, size=len(paths)))
    
    res = minimize(
        fun=neg_loglik,
        x0=y0,
        jac=True,
        args = (g, lb_tr, ub_tr),
        method=method,
        constraints=linear_constraints,
        options={
            'gtol': gtol,
            'xtol': xtol, 
            'maxiter': maxiter,
            'barrier_tol': barrier_tol,
        }
    )

    x_opt = np.array(x0) + LinvT @ res.x
    print("optimal value: ")
    print(x_opt)
    print(res)

    return _vec_to_dict(jnp.asarray(res.x), path_order), res