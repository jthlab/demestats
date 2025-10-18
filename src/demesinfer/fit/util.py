from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from scipy.optimize import Bounds
import jax.numpy as jnp
import jax
import numpy as np
from demesinfer.constr import EventTree, constraints_for
from scipy.optimize import LinearConstraint

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def _dict_to_vec(d: Params, keys: Sequence[Var]) -> jnp.ndarray:
    return jnp.asarray([d[k] for k in keys], dtype=jnp.float64)

def _vec_to_dict_jax(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, jnp.ndarray]:
    return {k: v[i] for i, k in enumerate(keys)}

def _vec_to_dict(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, float]:
    return {k: float(v[i]) for i, k in enumerate(keys)}

def create_constraints(demo, paths):
    path_order: List[Var] = list(paths)
    et = EventTree(demo)
    cons = constraints_for(et, *path_order)
    return cons

def finite_difference_hessian(f, x0, *args, eps=1e-6):
    """Compute only the diagonal of Hessian using finite differences"""
    n = len(x0)
    diag_H = jnp.zeros(n)
    
    for i in range(n):
        # For diagonal elements ∂²f/∂x_i², we can use central difference on the gradient
        def grad_i(x):
            return jax.grad(f)(x, *args)[i]
        
        # Central difference for ∂²f/∂x_i²
        x_plus = x0.at[i].add(eps)
        x_minus = x0.at[i].add(-eps)
        diag_H = diag_H.at[i].set((grad_i(x_plus) - grad_i(x_minus)) / (2 * eps))
    
    return jnp.diag(diag_H)

def make_whitening_from_hessian(f, x0, *args, tau=1e-3, lam=1e-3):
    H = finite_difference_hessian(f, x0, *args)
    H = 0.5 * (H + H.T)
    evals, evecs = jnp.linalg.eigh(H)
    evals = jnp.maximum(jnp.abs(evals), tau) + lam
    L = evecs @ jnp.diag(jnp.sqrt(evals)) @ evecs.T
    LinvT = jnp.linalg.solve(L, jnp.eye(L.shape[0])).T
    return L, LinvT

def pullback_objective(f, x0, LinvT, *args):
    def g(y, *args):
        x = x0 + LinvT @ y
        return f(x, *args)
    return g

###### TWO FUNCTIONS BELOW ARE STILL IN NEED OF EDITS ##########
def create_inequalities(A, b, LinvT, x0, size):
    # Group by variable to create more efficient constraints
    variable_constraints = {i: {'lb': -np.inf, 'ub': np.inf} for i in range(size)}
    
    for i, (a_row, b_val) in enumerate(zip(A, b)):
        var_idx = np.where(a_row != 0)[0][0]
        coefficient = a_row[var_idx]
        
        if coefficient < 0:  # Lower bound: x_i >= -b_val
            current_lb = variable_constraints[var_idx]['lb']
            variable_constraints[var_idx]['lb'] = max(current_lb, -b_val)
        else:  # Upper bound: x_i <= b_val
            current_ub = variable_constraints[var_idx]['ub']
            variable_constraints[var_idx]['ub'] = min(current_ub, b_val)
    
    # Create efficient combined constraints
    A_combined = np.eye(size)  # Identity matrix for individual variable constraints
    lb_combined = np.array([variable_constraints[i]['lb'] for i in range(size)])
    ub_combined = np.array([variable_constraints[i]['ub'] for i in range(size)])

    print(A_combined)
    print(lb_combined)
    print(ub_combined)

    A_tilde = A_combined @ LinvT
    lb_tilde = lb_combined - A_combined@x0
    ub_tilde = ub_combined - A_combined@x0

    return LinearConstraint(A_tilde, lb_tilde, ub_tilde)

def process_data(cfg_list):
    num_samples = len(cfg_list)

    deme_names = cfg_list[0].keys()
    D = len(deme_names)
    cfg_mat = jnp.zeros((num_samples, D), dtype=jnp.int32)
    for i, cfg in enumerate(cfg_list):
        for j, n in enumerate(deme_names):
            cfg_mat = cfg_mat.at[i, j].set(cfg.get(n, 0))

    unique_cfg = jnp.unique(cfg_mat, axis=0)

    # Find matching indices
    def find_matching_index(row, unique_arrays):
        matches = jnp.all(row == unique_arrays, axis=1)
        return jnp.where(matches)[0][0]

    # Vectorize over all rows in `arr`
    matching_indices = jnp.array([find_matching_index(row, unique_cfg) for row in cfg_mat])
    
    return cfg_mat, deme_names, unique_cfg, matching_indices