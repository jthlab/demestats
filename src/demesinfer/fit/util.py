from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from scipy.optimize import Bounds
import jax.numpy as jnp
import jax
import numpy as np
from demesinfer.constr import EventTree, constraints_for
from scipy.optimize import LinearConstraint
import equinox as eqx
import copy

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def _dict_to_vec(d: Params, keys: Sequence[Var]) -> jnp.ndarray:
    """
    Convert dictionary of parameters to vector representation.

    Parameters
    ----------
    d : Params
        Dictionary mapping parameter paths to values.
    keys : Sequence[Var]
        Ordered list of parameter paths.

    Returns
    -------
    jax.Array
        Vector representation of parameters in the order specified by keys.

    Notes
    -----
    This utility function is used internally to convert between dictionary
    and vector representations of parameters for optimization algorithms.
    """
    return jnp.asarray([d[k] for k in keys], dtype=jnp.float64)

def _vec_to_dict_jax(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, jnp.ndarray]:
    """
    Convert vector to dictionary with JAX arrays.

    Parameters
    ----------
    v : jax.Array
        Vector of parameter values.
    keys : Sequence[Var]
        Ordered list of parameter paths.

    Returns
    -------
    Dict[Var, jax.Array]
        Dictionary mapping parameter paths to JAX array values.
    """
    return {k: v[i] for i, k in enumerate(keys)}

def _vec_to_dict(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, float]:
    """
    Convert vector to dictionary with Python floats.

    Parameters
    ----------
    v : jax.Array
        Vector of parameter values.
    keys : Sequence[Var]
        Ordered list of parameter paths.

    Returns
    -------
    Dict[Var, float]
        Dictionary mapping parameter paths to Python float values.
    """
    return {k: float(v[i]) for i, k in enumerate(keys)}

def create_constraints(demo, paths):
    """
    Create constraints for demographic parameters directly from a demes graph and
    paths to parameters of interest.

    Parameters
    ----------
    demo : demes.Graph
        Demographic model.
    paths : Iterable[Var]
        Parameter paths to include in constraints.

    Returns
    -------
    dictionary
        Constraint representation of A @ x = b and A @ x <= b.

    Notes
    -----
    This function builds an EventTree from the demographic model and
    extracts equality and inequality constraints for the specified parameters.
    Constraints enforce demographic consistency (e.g., positive population
    sizes, chronological ordering of events).

    See Also
    --------
    demesinfer.event_tree.EventTree
    demesinfer.constr.constraints_for
    """
    path_order: List[Var] = list(paths)
    et = EventTree(demo)
    cons = constraints_for(et, *path_order)
    return cons

def finite_difference_hessian(f, x0, args_nonstatic, args_static, eps=1e-6):
    """
    Compute diagonal Hessian approximation using finite differences.

    Parameters
    ----------
    f : callable
        Function f(x, args_nonstatic, args_static) returning scalar.
    x0 : jax.Array
        Point at which to evaluate Hessian.
    args_nonstatic : tuple
        Non-static arguments to f.
    args_static : tuple
        Static arguments to f.
    eps : float, optional
        Finite difference step size (default: 1e-6).

    Returns
    -------
    jax.Array
        Diagonal Hessian matrix approximation.

    Notes
    -----
    This function computes only the diagonal elements of the Hessian
    using second-order central differences on the gradient. This is
    computationally cheaper than computing the full Hessian and is
    often sufficient for preconditioning.

    See Also
    --------
    demesinfer.fit.util.make_whitening_from_hessian
    """
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
    """
    Create whitening transformation (preconditioner) from Hessian diagonal approximation.

    Parameters
    ----------
    f : callable
        Objective function.
    x0 : jax.Array
        Reference point for Hessian evaluation.
    args_nonstatic : tuple
        Non-static arguments to f.
    args_static : tuple
        Static arguments to f.
    tau : float, optional
        Minimum eigenvalue threshold (default: 1e-3).
    lam : float, optional
        Regularization added to eigenvalues (default: 1e-3).

    Returns
    -------
    tuple
        (L, LinvT) where L is the whitening matrix and LinvT is its
        inverse transpose for constraint transformation.

    Notes
    -----
    Computes L = V @ sqrt(Λ) @ V.T where H = V @ Λ @ V.T is the
    eigendecomposition of the Hessian. The transformation whitens
    the parameter space, making contours more spherical and improving
    optimization performance.
    """
    H = finite_difference_hessian(f, x0, args_nonstatic, args_static)
    H = 0.5 * (H + H.T)
    evals, evecs = jnp.linalg.eigh(H)
    evals = jnp.maximum(jnp.abs(evals), tau) + lam
    L = evecs @ jnp.diag(jnp.sqrt(evals)) @ evecs.T
    LinvT = jnp.linalg.solve(L, jnp.eye(L.shape[0])).T
    return L, LinvT

def pullback_objective(f, args_static):
    """
    Create whitened objective function with gradient.

    Parameters
    ----------
    f : callable
        Original objective function f(x, args_nonstatic, args_static).
    args_static : tuple
        Static arguments to f.

    Returns
    -------
    callable
        Whitened function g(y, preconditioner_nonstatic, args_nonstatic)
        that returns (value, gradient) tuple.

    Notes
    -----
    This function creates a transformed objective that operates in
    whitened parameter space. The transformation is:
    x = x0 + LinvT @ y, where y are whitened parameters.

    The returned function is JIT-compiled and returns both the function
    value and its gradient, suitable for use with optimization algorithms.

    See Also
    --------
    demesinfer.fit.util.make_whitening_from_hessian
    """
    def g(y, preconditioner_nonstatic, args_nonstatic):
        x0, LinvT = preconditioner_nonstatic
        x = x0 + LinvT @ y
        return f(x, args_nonstatic, args_static)

    g = eqx.filter_jit(eqx.filter_value_and_grad(g))
    return g

def apply_jit(f, args_nonstatic, args_static):
    def g(x):
        x = jnp.atleast_1d(x)
        return f(x, args_nonstatic, args_static)

    g = eqx.filter_jit(g)
    return g

def alternative_constraint_rep(A, b):
    """
    Returns an alternative representation of inequality constraints with a lower and upper bound. 
    Depending on the numerical optimizer one would like to use, sometimes 
    it's more preferable to express inequality constraints explicitly with a 
    lower and upper bound. The input for the function are inequality constraints of the form
    Ax <= b which is exactly what the ``demesinfer.constr.constraints_for`` function returns.

    Parameters
    ----------
        A : array_like
            Coefficients for inequalities of the form Ax <= b
        b : array_list
            Values for inequalities of the form Ax <= b

    Notes
    -----
    Example:
    ::
        # See tutorial for a detailed example
        parameters = [
            ('demes', 0, 'epochs', 0, 'end_size'), # The ancestral population size
            ('migrations', 0, 'rate'), # Rate of migration from P0 to P1
            ('demes', 0, 'epochs', 0, 'end_time') # Time of divergence
        ]

        momi3_parameters = [et.variable_for(param) for param in parameters]
        constraint = constraints_for(et, *momi3_parameters)
        G, h = constraint["ineq"]
        A_alt, ub_alt, lb_alt = alternative_constraint_rep(G, h)
        print(A_alt)
        print("lower bound: ", lb_alt)
        print("upper bound: ", ub_alt)

    Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.
    
    See Also
    --------
    demesinfer.constr.constraint_for
    """
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

    return A_combined, ub_combined, lb_combined

def create_inequalities(A, b, LinvT, x0):
    """
    Create linear constraints that follow the format of ``scipy.optimize.LinearConstraint``

    Parameters
    ----------
    A : jax.Array or numpy.ndarray
        Constraint matrix of shape (m, n).
    b : jax.Array or numpy.ndarray
        Constraint bounds of shape (m,).
    LinvT : jax.Array
        Whitening transformation matrix (inverse transpose of preconditioner).
    x0 : jax.Array
        Reference point in original parameter space.

    Returns
    -------
    scipy.optimize.LinearConstraint
        Constraints transformed to whitened space: A_tilde @ y ≤ ub_tilde,
        where y = Linv @ (x - x0).

    Notes
    -----
    This function transforms constraints from the original parameter space
    to a whitened space where parameters are decorrelated. The transformation
    is: y = Linv @ (x - x0), where Linv is derived from Hessian information.

    See Also
    --------
    demesinfer.fit.util.alternative_constraint_rep
    demesinfer.fit.util.make_whitening_from_hessian
    """
    A_combined, ub_combined, lb_combined = alternative_constraint_rep(A, b)
    
    print(A_combined)
    print(lb_combined)
    print(ub_combined)

    A_tilde = A_combined @ LinvT
    lb_tilde = lb_combined - A_combined@x0
    ub_tilde = ub_combined - A_combined@x0

    return LinearConstraint(A_tilde, lb_tilde, ub_tilde)

def modify_constraints_for_equality(constraint, indices_for_equality):
    """
    Returns a modified version of the input ``constraint`` where all parameters associated with 
    indicies in ``indicies_for_equality`` will now have an equality constraint.

    Parameters
    ----------
        constraint : dict
            A dictionary of equality and inequality constraints
        indices_for_equality : list
            List of tuples, where each tuple are the indices of parameters you want to impose an equality constraint
    Returns:
        dict : A modified ``constraint`` with the new equality constraints
        
    Notes
    -----
    Example:
    ::
        # See tutorial for a detailed example
        parameters = [
            ('demes', 0, 'epochs', 0, 'end_size'), # The ancestral population size
            ('migrations', 0, 'rate'), # Rate of migration from P0 to P1
            ('demes', 0, 'epochs', 0, 'end_time') # Time of divergence
        ]

        momi3_parameters = [et.variable_for(param) for param in parameters]
        constraint = constraints_for(et, *momi3_parameters)

        # new_constraint will have the 2nd and 3rd variable be constrained to be equal
        new_constraint = modify_constraints_for_equality(constraint, [(1, 2)])
        print(new_constraint)

    Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.
    
    See Also
    --------
    demesinfer.constr.constraint_for
    """
    # Create a deep copy of the constraint dictionary
    constraint_copy = copy.deepcopy(constraint)
    
    # Extract constraints from the copy
    A_eq, b_eq = constraint_copy["eq"]
    
    # Build a new equality constraint: rate_0 - rate_1 = 0
    for index1, index2 in indices_for_equality:
        new_rule = np.zeros((1, A_eq.shape[1]))
        new_rule[0, index1] = 1.0
        new_rule[0, index2] = -1.0
        
        # Append to the existing constraint matrices
        A_eq = np.vstack([A_eq, new_rule])
        b_eq = np.concatenate([b_eq, [0.0]])

    constraint_copy["eq"] = (A_eq, b_eq)
    return constraint_copy

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