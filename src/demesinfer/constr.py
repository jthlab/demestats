"Constraints from event tree"

from collections import defaultdict
from collections.abc import Set
from itertools import combinations, count

import numpy as np
from beartype.typing import Sequence, TypedDict
from jaxtyping import ArrayLike, Float
from scipy.optimize import linprog

from .event_tree import EventTree
from .path import Path


class ConstraintSet(TypedDict):
    var: Sequence[Path | Set[Path]]
    eq: tuple[Float[ArrayLike, "eq d"], Float[ArrayLike, "eq"]]
    ineq: tuple[Float[ArrayLike, "ineq d"], Float[ArrayLike, "ineq"]]


def constraints_for(et: EventTree, *vars_: Path | Set[Path]) -> ConstraintSet:
    """
    Return a list of constraints for the given variables in the event tree.

    Params:
        et: The event tree to extract constraints from.
        var: The variables for which to extract constraints, which must exist in `et.variables`.

    Returns:
        ConstraintSet: A dictionary containing the equality and inequality constraints,
        as well as a mapping of columns of the constraint matrices to the variables.
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
            if np.isinf(val):
                continue
            else:
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
                    node = next(
                        node for node in et.T.nodes if et.T.nodes[node]["t"] == path
                    )
                    (parent,) = et.T.successors(node)
                    parent_t = et.T.nodes[parent]["t"]
                    parent_t_var = next(
                        j
                        for j, p in enumerate(all_variables)
                        if parent_t == p or parent_t in p
                    )
                    G.append(I[i] - I[parent_t_var])
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
            indices = [all_variables.index(p) for p in paths]
            A.append(sum(I[j] for j in indices))
            b.append(1.0)

    for i, pulse in enumerate(et.demo.pulses):
        paths = [
            ("pulses", i, "proportions", j) for j, _ in enumerate(pulse.proportions)
        ]
        indices = [all_variables.index(p) for p in paths]
        A.append(sum(I[j] for j in indices))
        b.append(1.0)

    A, b, G, h = map(np.array, (A, b, G, h))
    P = A.shape[1]
    assert G.shape[1] == P
    i_r = list(map(all_variables.index, vars_))  # indices of variables
    i_f = [p for p in range(P) if p not in i_r]  # fixed indices
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
