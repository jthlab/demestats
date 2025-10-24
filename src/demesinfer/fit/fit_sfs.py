from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from demesinfer.fit.util import _dict_to_vec, _vec_to_dict_jax, _vec_to_dict, finite_difference_hessian, create_inequalities, make_whitening_from_hessian, pullback_objective
from demesinfer.sfs import ExpectedSFS
from demesinfer.loglik.sfs_loglik import prepare_projection, projection_sfs_loglik, sfs_loglik
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import LinearConstraint, minimize

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def _compute_actual_likelihood(vec, path_order, esfs, proj_dict, einsum_str, input_arrays, sequence_length, theta, projection, afs):
    params = _vec_to_dict_jax(vec, path_order)
    jax.debug.print("Params: {params}", params=vec)
    
    if projection:
        loss = -projection_sfs_loglik(esfs, params, proj_dict, einsum_str, input_arrays, sequence_length, theta)
        jax.debug.print("Loss: {loss}", loss=loss)
        return loss
    else:
        e1 = esfs(params)
        loss = -sfs_loglik(afs, e1, sequence_length, theta)
        jax.debug.print("Loss full sfs: {loss}", loss=loss)
        return loss

def neg_loglik(vec, g, lb, ub):
    if jnp.any(vec > ub) or jnp.any(vec < lb):
        return jnp.inf, jnp.full_like(vec, jnp.inf)

    return g(vec)
    
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

    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None
    
    args = (path_order, esfs, proj_dict, einsum_str, input_arrays, sequence_length, theta, projection, afs)
    L, LinvT = make_whitening_from_hessian(_compute_actual_likelihood, x0, *args)
    g = pullback_objective(_compute_actual_likelihood, x0, LinvT, *args)
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

    return _vec_to_dict(jnp.asarray(res.x), path_order), res, x_opt