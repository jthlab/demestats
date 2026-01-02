"Constraints from event tree"

from collections import defaultdict
from itertools import combinations, count

import numpy as np
from beartype.typing import Sequence, TypedDict
from jaxtyping import ArrayLike, Float
from scipy.optimize import linprog
from typing import List, Any

from .event_tree import EventTree, Variable
from .path import Path


class ConstraintSet(TypedDict):
    var: Sequence[Variable]
    eq: tuple[Float[ArrayLike, "eq d"], Float[ArrayLike, "eq"]]
    ineq: tuple[Float[ArrayLike, "ineq d"], Float[ArrayLike, "ineq"]]


def constraints_for(et: EventTree, *vars_: Variable) -> ConstraintSet:
    """
    Return a list of constraints for the given variables in the event tree.

    Parameters
    ----------
        et : EventTree
            The event tree to extract constraints from
        var : Variable
            The variables for which to extract constraints, which must exist in `et.variables`.
    Returns:
        ConstraintSet : A dictionary containing the equality and inequality constraints,
        as well as a mapping of columns of the constraint matrices to the variables.

    Notes
    -----
    Example:
    ::
        et = EventTree(demo.to_demes())
        et.variables

    Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.
    """
    missing = [v for v in vars_ if v not in et.variables]
    if missing:
        raise ValueError(
            f"Variables {missing} not found in event tree. Use et.variables to see available variables."
        )

    A = []
    b = []
    G = []
    h = []

    # we iterate through each (set of) variables.
    all_variables = list(et.variables)

    def var_index(p: Path) -> int:
        return next(j for j, v in enumerate(all_variables) if p == v or p in v)

    n = len(all_variables)
    I = np.eye(n)
    for i, v in enumerate(all_variables):
        vs = v
        if isinstance(v, tuple):
            vs = set([v])
        v0 = next(iter(vs))
        val = et.get_path(v0)

        if v not in vars_:
            # this is a fixed value, so constrain it to be equal to its
            # initial value
            if np.isfinite(val):
                # fixed value
                A.append(I[i])
                b.append(val)
            continue

        # this is a variable we optimize over, so add the relevant constraints
        for path in vs:
            match path:
                case (*_, "time" | "start_time" | "end_time"):
                    G.append(-I[i])
                    h.append(0.0)
                    # find the time immediately above this in the event tree

                    # and constrain time ordering
                    # 1: less than parent
                    node = next(
                        node for node in et.nodes if et.nodes[node]["t"] == path
                    )
                    parent = node
                    while et.nodes[parent]["t"] in vs:
                        (parent,) = et.T.successors(parent)
                        parent_t = et.nodes[parent]["t"]
                    if np.isfinite(et.get_time(parent)):
                        parent_t_var = var_index(parent_t)
                        G.append(I[i] - I[parent_t_var])
                        h.append(0.0)
                    # 2: greater than children: traverse down until we find a child
                    # with a different time
                    q = list(et.T.predecessors(node))
                    while q:
                        child = q.pop()
                        child_t = et.nodes[child]["t"]
                        if et.nodes[child]["t"] in vs:
                            q.extend(et.T.predecessors(child))
                            continue
                        child_t_var = var_index(child_t)
                        G.append(-I[i] + I[child_t_var])
                        h.append(0.0)
                case (*_, "start_size" | "end_size"):
                    # this is a size variable, so constrain it to be non-negative
                    G.append(-I[i])
                    h.append(0.0)
                case (*_, "rate") | (*_, "proportions", _):
                    # this is a rate variable or proportion, so constrain it to be in [0, 1]
                    # per-generation migration rates are constrained to be in [0, 1] by spec
                    G.append(-I[i])
                    h.append(0.0)
                    G.append(I[i])
                    h.append(1.0)
        # end for

    # there is a linear constraint on all proportions adding up to 1
    for i, deme in enumerate(et.demo.demes):
        if deme.proportions:
            paths = [
                ("demes", i, "proportions", j) for j, _ in enumerate(deme.proportions)
            ]
            indices = map(var_index, paths)
            A.append(sum(I[j] for j in indices))
            b.append(1.0)

    for i, pulse in enumerate(et.demo.pulses):
        paths = [
            ("pulses", i, "proportions", j) for j, _ in enumerate(pulse.proportions)
        ]
        indices = map(var_index, paths)
        A.append(sum(I[j] for j in indices))
        b.append(1.0)

    A, b, G, h = map(np.array, (A, b, G, h))
    A, G = [x.reshape(-1, n) for x in (A, G)]
    i_r = list(map(all_variables.index, vars_))  # indices of variables
    i_f = [p for p in range(n) if p not in i_r]  # fixed indices
    xf = [et.get_var(var) for var in all_variables if var not in vars_]
    A_r = A[:, i_r]
    b_r = b - A[:, i_f].dot(xf)
    G_r = G[:, i_r]
    h_r = h - G[:, i_f].dot(xf)
    A_r, b_r, G_r, h_r = _reduce(A_r, b_r, G_r, h_r)
    return dict(
        eq=(A_r, b_r),
        ineq=(G_r, h_r),
    )


def _reduce(
    A: Float[ArrayLike, "E P"],
    b: Float[ArrayLike, "E"],
    G: Float[ArrayLike, "I P"],
    h: Float[ArrayLike, "I"],
) -> tuple[
    Float[ArrayLike, "Er P"],
    Float[ArrayLike, "Er"],
    Float[ArrayLike, "Ir P"],
    Float[ArrayLike, "Ir"],
]:
    """Reduce the system of equations Ax = b by removing redundant equalities."""
    U, s, Vt = np.linalg.svd(A)
    rank = np.sum(s > 1e-10)
    A_red = U[:, :rank].T @ A
    b_red = U[:, :rank].T @ b

    G_red = G
    h_red = h

    def f():
        nonlocal G_red, h_red
        for i in range(len(h_red)):
            if _is_redundant_inequality(A_red, b_red, G_red, h_red, i):
                G_red = np.delete(G_red, i, axis=0)
                h_red = np.delete(h_red, i)
                return False
        return True

    while not f():
        pass

    return A_red, b_red, G_red, h_red


def _is_redundant_inequality(A, b, G, h, i, tol=1e-7):
    c = -G[i]  # maximize g_i^T x => minimize -g_i^T x

    # Remove i-th inequality
    G_rest = np.delete(G, i, axis=0)
    h_rest = np.delete(h, i)

    res = linprog(
        c, A_eq=A, b_eq=b, A_ub=G_rest, b_ub=h_rest, bounds=(None, None), method="highs"
    )
    if not res.success:
        return False  # LP failed or infeasible, be conservative

    max_value = -res.fun
    return max_value <= h[i] + tol

def display_constraint(A: np.ndarray, b: np.ndarray, x: List[Any], equality: bool) -> None:
    """
    Compact display of constraints without variable definitions.
    """
    m, n = A.shape
    
    print("\nCONSTRAINTS:")
    print("-" * 50)
    
    for i in range(m):
        # Build expression
        expr_parts = []
        for j in range(n):
            coeff = A[i, j]
            if coeff != 0:
                if coeff == 1:
                    expr_parts.append(f"x{j+1}")
                elif coeff == -1:
                    expr_parts.append(f"-x{j+1}")
                else:
                    if coeff.is_integer():
                        coeff_str = str(int(coeff))
                    else:
                        coeff_str = f"{coeff:.3f}".rstrip('0').rstrip('.')
                    expr_parts.append(f"{coeff_str}·x{j+1}")
        
        if not expr_parts:
            expr = "0"
        else:
            expr = " + ".join(expr_parts).replace("+ -", " - ")
        
        # Format RHS
        if b[i].is_integer():
            rhs = str(int(b[i]))
        else:
            rhs = f"{b[i]:.3f}".rstrip('0').rstrip('.')

        if equality:
            print(f"Row {i+1}: {expr} = {rhs}")
        else:
            print(f"Row {i+1}: {expr} <= {rhs}")
    
    print("-" * 50)

def display_constraint_strings(A: np.ndarray, b: np.ndarray, x: List[Any], equality: bool) -> List[str]:
    """
    Return list of constraint strings.
    """
    m, n = A.shape
    constraints = []
    
    print("\nAS STRINGS:")
    print("-" * 50)
    
    for i in range(m):
        terms = []
        for j in range(n):
            coeff = A[i, j]
            if coeff != 0:
                if coeff == 1:
                    terms.append(f"{x[j]}")
                elif coeff == -1:
                    terms.append(f"-{x[j]}")
                else:
                    terms.append(f"{coeff} * {x[j]}")
        
        if not terms:
            lhs = "0"
        else:
            lhs = " + ".join(terms).replace("+ -", " - ")

        if equality:
            constraints.append(f"{lhs} = {b[i]}")
        else:
            constraints.append(f"{lhs} <= {b[i]}")

    for i, constr in enumerate(constraints):
        print(f"Row {i+1}: {constr}")

    print("-" * 50)
    
def print_constraints(constraint_dict, variable_list):
    """
    Print out all equality and inequality constraints in a more interpretable way.

    Parameters
    ----------
        constraint_dict : dict
            A dictionary of equality and inequality constraints
        variable : list
            List of associated variable names

    Notes
    -----
    The constraint_dict is obtained from the output ``constraints_for`` function.
    
    Example:
    ::
        # See example in the tutorial
        parameters = [
            ('demes', 0, 'epochs', 0, 'end_size'), # The ancestral population size
            ('migrations', 0, 'rate'), # Rate of migration from P0 to P1
            ('demes', 0, 'epochs', 0, 'end_time') # Time of divergence
        ]

        variable_list = [et.variable_for(param) for param in parameters]
        constraint = constraints_for(et, *variable_list)
        print_constraints(constraint, variable_list)


    Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.
    
    See Also
    --------
    demesinfer.constr.constraints_for
    """
    print("\n" + "=" * 50)
    print("Linear Equalities: Ax = b")
    print("=" * 50)
    
    if len(constraint_dict['eq'][0]) > 0:
        display_constraint(constraint_dict['eq'][0], constraint_dict['eq'][1], variable_list, equality=True)
        display_constraint_strings(constraint_dict['eq'][0], constraint_dict['eq'][1], variable_list, equality=True)
    else:
        print("\nNone")
        
    print("\n" + "=" * 50)
    print("Linear Inequalities: Ax <= b")
    print("=" * 50)
    
    if len(constraint_dict['ineq'][0]) > 0:
        display_constraint(constraint_dict['ineq'][0], constraint_dict['ineq'][1], variable_list, equality=False)
        display_constraint_strings(constraint_dict['ineq'][0], constraint_dict['ineq'][1], variable_list, equality=False)
    else:
        print("\nNone")
