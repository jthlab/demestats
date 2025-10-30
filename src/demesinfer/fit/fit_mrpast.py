from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from demesinfer.fit.util import _dict_to_vec, _vec_to_dict_jax, _vec_to_dict, create_inequalities, make_whitening_from_hessian, pullback_objective
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import LinearConstraint, minimize
from demesinfer.iicr import IICRCurve
from jax import vmap


Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def compute_loglik(time, sample_config, params, iicr_call, deme_names):
   ns = {name: sample_config[i] for i, name in enumerate(deme_names)}
   result = iicr_call(params=params, t=time, num_samples=ns)
   return jnp.sum(jnp.log(result["c"]) + result["log_s"])

def _compute_mrpast_likelihood(vec, path_order, data, cfg_mat, iicr_call, deme_names):
    params = _vec_to_dict_jax(vec, path_order)
    jax.debug.print("param: {vec}", vec=vec)
    batched_loglik = vmap(compute_loglik, in_axes=(0, 0, None, None, None))(data, cfg_mat, params, iicr_call, deme_names)
    loss = -jnp.sum(batched_loglik)
    jax.debug.print("Loss: {loss}", loss=loss)
    return loss

def neg_loglik(vec, g, lb, ub):
    if jnp.any(vec > ub) or jnp.any(vec < lb):
        return jnp.inf, jnp.full_like(vec, jnp.inf)

    return g(vec)
    
def fit(
    demo,
    paths: Params,
    data, 
    cfg_list,
    cons,
    lb,
    ub,
    k,
    *,
    method: str = "trust-constr",
    gtol: float = 1e-8,
    xtol: float = 1e-8, #default 1e-8
    maxiter: int = 1000, #default 1000
    barrier_tol: float = 1e-8,
):
    path_order: List[Var] = list(paths)
    x0 = _dict_to_vec(paths, path_order)
    data = jnp.array(data)
    x0 = jnp.array(x0)
    lb = jnp.array(lb)
    ub = jnp.array(ub)

    cfg_mat, deme_names = process_data(cfg_list)
    num_samples = len(cfg_mat)
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    args = (path_order, data, cfg_mat, iicr_call, deme_names)
    L, LinvT = make_whitening_from_hessian(_compute_mrpast_likelihood, x0, *args)
    g = pullback_objective(_compute_mrpast_likelihood, x0, LinvT, *args)
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