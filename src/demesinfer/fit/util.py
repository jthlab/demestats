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

def finite_difference_hessian(f, x0, *args, eps=1e-6):
    """Compute only the diagonal of Hessian using finite differences"""
    n = len(x0)
    diag_H = jnp.zeros(n)

    def loglik_static(params):
        return f(params, *args)
        
    # For diagonal elements ∂²f/∂x_i², we can use central difference on the gradient
    grad_f = eqx.filter_jit(jax.grad(loglik_static))
    
    for i in range(n):
        # Central difference for ∂²f/∂x_i²
        x_plus = x0.at[i].add(eps)
        x_minus = x0.at[i].add(-eps)

        # Evaluate gradient at perturbed points and take i-th component
        grad_plus_i = grad_f(x_plus)[i]
        grad_minus_i = grad_f(x_minus)[i]
        diag_H = diag_H.at[i].set((grad_plus_i - grad_minus_i) / (2 * eps))
    
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
    def g(y):
        x = x0 + LinvT @ y
        return f(x, *args)

    g = eqx.filter_jit(eqx.filter_value_and_grad(g))
    return g

def apply_jit(f, *args):
    def g(x):
        x = jnp.atleast_1d(x)
        return f(x, *args)

    g = eqx.filter_jit(g)
    return g

###### TWO FUNCTIONS BELOW ARE STILL IN NEED OF EDITS ##########
# def create_inequalities(A, b, LinvT, x0, size):
#     # Group by variable to create more efficient constraints
#     variable_constraints = {i: {'lb': -np.inf, 'ub': np.inf} for i in range(size)}
    
#     for i, (a_row, b_val) in enumerate(zip(A, b)):
#         var_idx = np.where(a_row != 0)[0][0]
#         coefficient = a_row[var_idx]
        
#         if coefficient < 0:  # Lower bound: x_i >= -b_val
#             current_lb = variable_constraints[var_idx]['lb']
#             variable_constraints[var_idx]['lb'] = max(current_lb, -b_val)
#         else:  # Upper bound: x_i <= b_val
#             current_ub = variable_constraints[var_idx]['ub']
#             variable_constraints[var_idx]['ub'] = min(current_ub, b_val)
    
#     # Create efficient combined constraints
#     A_combined = np.eye(size)  # Identity matrix for individual variable constraints
#     lb_combined = np.array([variable_constraints[i]['lb'] for i in range(size)])
#     ub_combined = np.array([variable_constraints[i]['ub'] for i in range(size)])

#     print(A_combined)
#     print(lb_combined)
#     print(ub_combined)

#     A_tilde = A_combined @ LinvT
#     lb_tilde = lb_combined - A_combined@x0
#     ub_tilde = ub_combined - A_combined@x0

#     return LinearConstraint(A_tilde, lb_tilde, ub_tilde)

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

###################### ARG RELATED ################################
def reformat_data(data_pad, matching_indices, max_indices, chunking_length):
    unique_groups = jnp.unique(matching_indices)
    group_unique_times = []
    rearranged_data = []
    new_max_indices = []
    new_matching_indices = []
    associated_indices = []
    group_membership = []

    # Each group is a sampling configuration
    for group in unique_groups:
        positions = jnp.where(matching_indices == group) # extract positions matching a group
        group_data = data_pad[positions] # find all tmrca + spans that share a sampling config
        all_first_col = np.array(group_data[:,:,0].flatten())
        unique_values = np.unique(all_first_col) # extra all unique tmrca associated to a sampling config

        unique_value_to_index = {value: idx for idx, value in enumerate(unique_values)}
        indices_in_mapping = np.array([unique_value_to_index[value] for value in all_first_col])
        associated_indices.append(indices_in_mapping) # figure how ever tmrca in data_pad gets mapped to unique_values

        group_unique_times.append(unique_values)
        rearranged_data.append(group_data)
        new_matching_indices.append(matching_indices[positions])
        new_max_indices.append(max_indices[positions])
        group_membership.append(jnp.full(indices_in_mapping.size, group))

    # Find the maximum length
    max_length = max(len(arr) for arr in group_unique_times)

    # Pad each array with zeros at the end
    padded_unique_times = []
    for arr in group_unique_times:
        pad_length = max_length - len(arr)
        padded = np.pad(arr, (0, pad_length), mode='constant', constant_values=0)
        padded_unique_times.append(padded)

    padded_unique_times = jnp.array(padded_unique_times)
    rearranged_data = jnp.concatenate(rearranged_data, axis=0)
    new_matching_indices = jnp.concatenate(new_matching_indices, axis=0)
    new_max_indices = jnp.concatenate(new_max_indices, axis=0)
    associated_indices = jnp.concatenate(associated_indices, axis=0)
    group_membership = jnp.concatenate(group_membership, axis=0)
    total_elements = group_membership.size
    batch_size = total_elements // chunking_length
    return padded_unique_times, rearranged_data, new_matching_indices, new_max_indices, associated_indices, unique_groups, batch_size, group_membership

def process_arg_data(data_list, cfg_list):
    max_indices = jnp.array([arr.shape[0]-1 for arr in data_list])
    num_samples = len(max_indices)
    lens = jnp.array([d.shape[0] for d in data_list], dtype=jnp.int32)
    Lmax = int(lens.max())
    Npairs = len(data_list)
    data_pad = jnp.full((Npairs, Lmax, 2), jnp.array([1.0, 0.0]), dtype=jnp.float64)

    for i, d in enumerate(data_list):
        data_pad = data_pad.at[i, : d.shape[0], :].set(d)

    deme_names = cfg_list[0].keys()
    D = len(deme_names)
    cfg_mat = jnp.zeros((num_samples, D), dtype=jnp.int32)
    for i, cfg in enumerate(cfg_list):
        for j, n in enumerate(deme_names):
            cfg_mat = cfg_mat.at[i, j].set(cfg.get(n, 0))

    unique_cfg = jnp.unique(cfg_mat, axis=0)

    # Find matching indices for sampling configs
    def find_matching_index(row, unique_arrays):
        matches = jnp.all(row == unique_arrays, axis=1)
        return jnp.where(matches)[0][0]

    # Vectorize over all rows in `arr`
    matching_indices = jnp.array([find_matching_index(row, unique_cfg) for row in cfg_mat])
    
    return data_pad, deme_names, max_indices, unique_cfg, matching_indices

def compile(ts, subkey, a=None, b=None):
    # using a set to pull out all unique populations that the samples can possibly belong to
    pop_cfg = {ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()}
    pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

    if a == None and b == None:
        samples = jax.random.choice(subkey, ts.num_samples, shape=(2,), replace=False)
        a, b = samples[0].item(0), samples[1].item(0)

    spans = []
    curr_t = None
    curr_L = 0.0
    for tree in ts.trees():
        L = tree.interval.right - tree.interval.left
        t = tree.tmrca(a, b)
        if curr_t is None or t != curr_t:
            if curr_t is not None:
                spans.append([curr_t, curr_L])
            curr_t = t
            curr_L = L
        else:
            curr_L += L
    spans.append([curr_t, curr_L])
    data = jnp.asarray(spans, dtype=jnp.float64)
    pop_cfg[ts.population(ts.node(a).population).metadata["name"]] += 1
    pop_cfg[ts.population(ts.node(b).population).metadata["name"]] += 1
    return data, pop_cfg

def get_tmrca_data(ts, seed=2, num_samples=200, option="random"):
    key=jax.random.PRNGKey(seed)
    data_list = []
    cfg_list = []
    key, subkey = jax.random.split(key)
    if option == "random":
        for i in range(num_samples):
            data, cfg = compile(ts, subkey)
            data_list.append(data)
            cfg_list.append(cfg)
            key, subkey = jax.random.split(key)
    elif option == "all":
        from itertools import combinations
        all_config = list(combinations(ts.samples(), 2))
        for a, b in all_config:
            data, cfg = compile(ts, subkey, a, b)
            data_list.append(data)
            cfg_list.append(cfg)
    elif option == "unphased":
        all_config = ts.samples().reshape(-1, 2)
        for a, b in all_config:
            data, cfg = compile(ts, subkey, a, b)
            data_list.append(data)
            cfg_list.append(cfg)

    return data_list, cfg_list     

