from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from fit.util import _dict_to_vec, _vec_to_dict_jax, _vec_to_dict, create_bounds, finite_difference_hessian, create_inequalities
from demesinfer.sfs import ExpectedSFS
from demesinfer.loglik.sfs_loglik import prepare_projection, projection_sfs_loglik, sfs_loglik
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import LinearConstraint, minimize

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def fit(
    demo,
    paths: Params,
    afs,
    afs_samples,
    cons,
    *,
    method: str = "trust-constr",
    options: Optional[dict] = None,
    recombination_rate: float = None,
    sequence_length: float = None,
    theta: float = None,
    projection: bool = False,
    num_projections: float = 200,
    seed: float = 5, 
    gtol: float = 1e-8,
    xtol: float = 1e-8, #default 1e-8
    maxiter: int = 1000, #default 1000
    barrier_tol: float = 1e-8,
    lower_bound: float = 0.0,
    upper_bound: float = 0.1,
    bounds = None,
):
    path_order: List[Var] = list(paths)
    x0 = _dict_to_vec(paths, path_order)
    x0 = jnp.array(x0)

    bounds = True
    if not bounds:
        bounds = create_bounds(paths)

    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    def make_whitening_from_hessian(f, x0, tau=1e-3, lam=1e-3):
        # H = jax.hessian(f)(x0)
        H = finite_difference_hessian(f, x0)
        print(H)
        H = 0.5 * (H + H.T)
        evals, evecs = jnp.linalg.eigh(H)
        evals = jnp.maximum(jnp.abs(evals), tau) + lam
        # L = sqrt(M)
        L = evecs @ jnp.diag(jnp.sqrt(evals)) @ evecs.T
        LinvT = jnp.linalg.solve(L, jnp.eye(L.shape[0])).T
        return L, LinvT
    
    # Build g(y) so you can hand it to ANY optimizer you already use.
    # @jax.value_and_grad
    def pullback_objective(f, x0, LinvT):
        def g(y):
            x = x0 + LinvT @ y
            return f(x)
        return g
    
    # @jax.value_and_grad
    def neg_loglik(vec):
        # if (vec > np.array([0.001, 0.001, 0.001, 0.001, np.inf, np.inf, np.inf, np.inf])).any():
        # if (vec > np.array([0.001, 0.001, np.inf, np.inf])).any():
        # if (vec > np.array([np.inf, np.inf, np.inf, np.inf])).any():
        if (vec > np.array([0.001, 0.001])).any():
            return np.inf

        # vif (vec < np.array([0, 0, 0, 0, 0, 0, 0, 0])).any():
        # if (vec < np.array([0, 0, 0, 0])).any():
        if (vec < np.array([0, 0])).any():
            return np.inf 
            
        params = _vec_to_dict_jax(vec, path_order)
        # jax.debug.print("Param: {}", params)
        jax.debug.print("Param values: {}", jnp.array(list(params.values())))

        if projection:
            tp = jax.jit(lambda X: esfs.tensor_prod(X, params))
            loss = -projection_sfs_loglik(afs, tp, proj_dict, einsum_str, input_arrays, sequence_length, theta) # / 1e3
            jax.debug.print("Loss: {loss}", loss=loss)
            return loss
        else:
            e1 = esfs(params)
            return -sfs_loglik(afs, e1, sequence_length, theta)

    L, LinvT = make_whitening_from_hessian(neg_loglik, x0)
    g = pullback_objective(neg_loglik, x0, LinvT)
    y0 = np.zeros_like(x0)

    linear_constraints: list[LinearConstraint] = []
    
    Aeq, beq = cons["eq"]
    if Aeq.size:
        linear_constraints.append(LinearConstraint(Aeq, beq, beq))

    G, h = cons["ineq"]
    if G.size:
        lincon = create_inequalities(G, h, LinvT, x0, size=len(paths))
    
    res = minimize(
        fun=jax.value_and_grad(g),
        x0=y0,
        jac=True,
        method=method,
        # method = "SLSQP",
        constraints=(lincon, ),
        # options={
        # 'ftol':1e-8,
        # 'disp':True,
        # 'maxiter':1000,
        # }
    )

    x_opt = np.array(x0) + LinvT @ res.x
    print("optimal value: ")
    print(x_opt)

    return _vec_to_dict(jnp.asarray(res.x), path_order), res