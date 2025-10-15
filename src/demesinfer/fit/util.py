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

def create_bounds(param_list, lower_bound=0.0, upper_bound=0.0004):
    """
    Create bounds where any tuple parameter with 'migration' in first position is bounded
    """
    n_params = len(param_list)
    lb_list = [-np.inf] * n_params
    ub_list = [np.inf] * n_params
    
    for i, param in enumerate(param_list):
        if isinstance(param, tuple) and "migration" in str(param[0]):
            lb_list[i] = lower_bound
            ub_list[i] = upper_bound
    
    return Bounds(lb=lb_list, ub=ub_list, keep_feasible=[True] * len(param_list))

def create_constraints(demo, paths):
    path_order: List[Var] = list(paths)
    et = EventTree(demo)
    cons = constraints_for(et, *path_order)
    return cons

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

    print(A_tilde)
    print(lb_tilde)
    print(ub_tilde)

    return LinearConstraint(A_tilde, lb_tilde, ub_tilde)

def finite_difference_hessian(f, x0, eps=1e-4):
    """Compute Hessian using finite differences"""
    n = len(x0)
    H = jnp.zeros((n, n))
    
    for i in range(n):
        def grad_i(x):
            return jax.grad(f)(x)[i]
        
        for j in range(n):
            # Central difference for ∂²f/∂x_i∂x_j
            def f_ij(x):
                return grad_i(x)[j]
            
            # Simple finite difference
            x_plus = x0.at[j].add(eps)
            x_minus = x0.at[j].add(-eps)
            H = H.at[i, j].set((grad_i(x_plus) - grad_i(x_minus)) / (2 * eps))
    
    return H

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