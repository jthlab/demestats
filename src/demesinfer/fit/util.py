from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from scipy.optimize import Bounds
import jax.numpy as jnp
import jax
import numpy as np
from demesinfer.constr import EventTree, constraints_for
from scipy.optimize import LinearConstraint
import equinox as eqx

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

def finite_difference_hessian(f, x0, args_nonstatic, args_static, eps=1e-6):
    """Compute only the diagonal of Hessian using finite differences"""
    n = len(x0)
    diag_H = jnp.zeros(n)

    def loglik_static(params, args_nonstatic):
        return f(params, args_nonstatic, args_static)
        
    # For diagonal elements ∂²f/∂x_i², we can use central difference on the gradient
    grad_f = eqx.filter_jit(jax.grad(loglik_static))
    
    for i in range(n):
        # Central difference for ∂²f/∂x_i²
        x_plus = x0.at[i].add(eps)
        x_minus = x0.at[i].add(-eps)

        # Evaluate gradient at perturbed points and take i-th component
        grad_plus_i = grad_f(x_plus, args_nonstatic)[i]
        grad_minus_i = grad_f(x_minus, args_nonstatic)[i]
        diag_H = diag_H.at[i].set((grad_plus_i - grad_minus_i) / (2 * eps))
    
    return jnp.diag(diag_H)

def make_whitening_from_hessian(f, x0, args_nonstatic, args_static, tau=1e-3, lam=1e-3):
    H = finite_difference_hessian(f, x0, args_nonstatic, args_static)
    H = 0.5 * (H + H.T)
    evals, evecs = jnp.linalg.eigh(H)
    evals = jnp.maximum(jnp.abs(evals), tau) + lam
    L = evecs @ jnp.diag(jnp.sqrt(evals)) @ evecs.T
    LinvT = jnp.linalg.solve(L, jnp.eye(L.shape[0])).T
    return L, LinvT

def pullback_objective(f, args_static):
    def g(y, preconditioner_nonstatic, args_nonstatic):
        x0, LinvT = preconditioner_nonstatic
        x = x0 + LinvT @ y
        return f(x, args_nonstatic, args_static)

    g = eqx.filter_jit(eqx.filter_value_and_grad(g))
    return g

def apply_jit(f, *args):
    def g(x):
        x = jnp.atleast_1d(x)
        return f(x, *args)

    g = eqx.filter_jit(g)
    return g

def create_inequalities(A, b, LinvT, x0, size):
    # Group by variable to create more efficient constraints
    replace_idx = np.ones(len(b), dtype=bool)
    A_combined = []
    lb_combined = []
    ub_combined = []

    # only conditions with one SINGLE index that's repeated will be joined together. If a row has two non-zero indices, that must be copied exactly
    for i in range(len(A)):
        a_row1, b_val1 = A[i], b[i]
        idx1 = np.where(a_row1 != 0)[0]
        for j in range(i + 1, len(A)):  # Start from i+1
            a_row2, b_val2 = A[j], b[j]
            idx2 = np.where(a_row2 != 0)[0]
            if (len(idx1) == 1) and np.array_equal(idx1, idx2) and replace_idx[i] == True:
                replace_idx[i] = False
                replace_idx[j] = False
                if a_row1[idx1[0]] == -1:
                    A_combined.append(a_row2)
                    ub_combined.append(b_val2)
                    lb_combined.append(b_val1)
                else:
                    A_combined.append(a_row1)
                    ub_combined.append(b_val1)
                    lb_combined.append(b_val2)

        if replace_idx[i]:
            if len(idx1) == 1:
                a_row1 = -1 * a_row1
                A_combined.append(a_row1)
                ub_combined.append(np.inf)
                lb_combined.append(b_val1)
            else:
                A_combined.append(a_row1)
                ub_combined.append(b_val1)
                lb_combined.append(-np.inf)

    A_combined = jnp.array(A_combined)
    ub_combined = jnp.array(ub_combined)
    lb_combined = jnp.array(lb_combined)
    print(A_combined)
    print(lb_combined)
    print(ub_combined)

    A_tilde = A_combined @ LinvT
    lb_tilde = lb_combined - A_combined@x0
    ub_tilde = ub_combined - A_combined@x0

    return LinearConstraint(A_tilde, lb_tilde, ub_tilde)

def modify_constraints_for_equality(constraint, indices_for_equality):
    # Extract existing constraints
    A_eq, b_eq = constraint["eq"]
    
    # Build a new equality constraint: rate_0 - rate_1 = 0
    for index1, index2 in indices_for_equality:
        new_rule = np.zeros((1, A_eq.shape[1]))
        new_rule[0, index1] = 1.0
        new_rule[0, index2] = -1.0
        
        # Append to the existing constraint matrices
        A_eq = np.vstack([A_eq, new_rule])
        b_eq = np.concatenate([b_eq, [0.0]])

    constraint["eq"] = (A_eq, b_eq)
    return constraint

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