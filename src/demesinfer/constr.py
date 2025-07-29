"Constraints from event tree"

from collections import defaultdict
from itertools import combinations, count

import numpy as np
from beartype.typing import Sequence, TypedDict
from jaxtyping import ArrayLike, Float
from scipy.optimize import linprog
from sympy import Expr, Symbol, symbols

from .event_tree import EventTree
from .path import Path


class ConstraintSet(TypedDict):
    eq: tuple[Float[ArrayLike, "eq d"], Float[ArrayLike, "eq"]]
    ineq: tuple[Float[ArrayLike, "ineq d"], Float[ArrayLike, "ineq"]]


def constraints(et: EventTree) -> ConstraintSet:
    """
    Return a list of constraints for the given event tree.

    Params:
        et: The event tree to extract constraints from.

    Returns
    -------
    list of Path
        A list of paths representing the constraints.
    """
    eq = []
    ineq = []
    mapping = {}

    # times
    all_times = {}
    for attrs in et.nodes.values():
        path = attrs["t"]
        all_times.setdefault(et.get_path(path), set()).add(path)

    i = count()
    path_times = {}
    for t, paths in sorted(all_times.items()):
        if np.isinf(t):
            assert len(paths) == 1
            path_times[paths.pop()] = t
            continue
        tau_p = []
        for p in paths:
            tau_i = symbols(f"tau_{next(i)}")
            eq.append(tau_i - t)
            mapping[tau_i] = p
            tau_p.append(tau_i)
        for tau_i, tau_j in combinations(tau_p, 2):
            ineq.append(tau_i - tau_j)

    path_times.update({p: tau_i for tau_i, p in mapping.items()})

    # next, require that times are increasing along edges
    for edge in et.edges:
        t_u, t_v = [path_times[et.nodes[x]["t"]] for x in edge]
        if edge[1] == et.root:
            assert np.isinf(t_v)
            continue
        ineq.append(t_u - t_v)

    # other parameters
    demo = et._demo
    # sizes
    for i, d in enumerate(demo.demes):
        for j, e in enumerate(d.epochs):
            eta0, eta1 = symbols([f"eta_{{{i}{j}{x}}}" for x in "01"])
            mapping[eta0] = ("demes", i, "epochs", j, "start_size")
            mapping[eta1] = ("demes", i, "epochs", j, "end_size")
            ineq.append(-eta0)
            ineq.append(-eta1)
            if e.size_function == "constant":
                eq.append(eta0 - eta1)

        if d.ancestors:
            alphas = []
            for j, a in enumerate(d.ancestors):
                alpha = symbols(f"alpha_{{{i}{j}}}")
                mapping[alpha] = ("demes", i, "proportions", j)
                alphas.append(alpha)
                ineq.append(-alpha)
            eq.append(sum(alphas) - 1.0)

    # rates
    k = count()
    for i, m in enumerate(demo.migrations):
        mu = symbols(f"mu_{next(k)}")
        mapping[mu] = ("migrations", i, "rate")
        ineq.append(-mu)
        ineq.append(mu - 1)

    # proportions
    for i, p in enumerate(demo.pulses):
        rhos = []
        for j, p in enumerate(p.proportions):
            rho = symbols(f"rho_{{{i}{j}}}")
            mapping[rho] = ("demes", i, "proportions", j)
            rhos.append(rho)
            ineq.append(-rho)
        eq.append(sum(rhos) - 1.0)

    indets = list(mapping)
    A, b = _to_real(eq, indets)
    A_red, b_red = _reduce_equalities(A, np.zeros(len(eq)))
    G, h = _to_real(ineq, indets)
    G_red, h_red = _reduce_inequalities(G, h)

    # I decided not to use these
    eq = _to_symbolic(A_red, b_red, indets)
    ineq = _to_symbolic(G_red, h_red, indets)

    return dict(ineq=(G_red, h_red), eq=(A_red, b_red), mapping=mapping)


def constraints_for(et: EventTree, params: Sequence[Path]) -> ConstraintSet:
    """Return a dict of equality and inequality constraints for a given set of paths.

    Params:
        et: The event tree to extract constraints from.
        params: A sequence of paths to extract constraints for.

    Returns:
        dict[str, np.ndarray]: A dictionary with keys 'ineq' and 'eq', containing the constraints.
        The columns of the returned arrays correspond to the parameters in `params`, in order.
    """
    demo = et.demodict
    cons = constraints(et)
    A, b = cons["eq"]
    G, h = cons["ineq"]
    P = A.shape[1]
    assert G.shape[1] == P
    path_list = list(cons["mapping"].values())
    i_r = list(map(path_list.index, params))  # indices of variables
    i_f = [p for p in range(P) if p not in i_r]  # fixed indices
    xf = [et.get_path(path) for path in path_list if path not in params]
    A_r = A[:, i_r]
    b_r = b - A[:, i_f].dot(xf)
    G_r = G[:, i_r]
    h_r = h - G[:, i_f].dot(xf)
    A_r, b_r = _reduce_equalities(A_r, b_r)
    G_r, h_r = _reduce_inequalities(G_r, h_r)

    # the equality constraints can never have rows with only a single non-zero entry
    # because this would force the parameter to equal a constant and then there would be
    # no point of optimization
    rows_to_keep = np.where(np.count_nonzero(A_r, axis=1) != 1)[0]
    A_r = A_r[rows_to_keep, :]
    b_r = b_r[rows_to_keep]

    return dict(eq=(A_r, b_r), ineq=(G_r, h_r))


def _to_symbolic(A, b, indets) -> list[Expr]:
    new_eq = []
    for row, bi in zip(A, b):
        s = 0.0
        for i in range(len(indets)):
            if row[i] != 0:
                if row[i] == 1.0:
                    s += indets[i]
                if row[i] == -1.0:
                    s -= indets[i]
                else:
                    s += row[i] * indets[i]
        new_eq.append(s - bi)
    return new_eq


def _to_real(sys: list[Expr], indets: list[Symbol]) -> tuple[np.ndarray, np.ndarray]:
    A = []
    b = []
    for expr in sys:
        row = [0.0] * len(indets)
        cd = expr.as_coefficients_dict()
        for i, indet in enumerate(indets):
            row[i] = float(cd.get(indet, 0))
        cons = cd.get(1, 0.0)
        A.append(row)
        b.append(-float(cons))
    return tuple(map(np.array, (A, b)))


def _reduce_equalities(A, b) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the system of equations Ax = b by removing redundant equalities."""
    U, s, Vt = np.linalg.svd(A)
    rank = np.sum(s > 1e-10)
    A_red = U[:, :rank].T @ A
    b_red = U[:, :rank].T @ b
    return A_red, b_red


def _reduce_inequalities(A, b) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove redundant linear constraints from the system Ax <= b.
    """
    i = 0
    while i < len(A):
        e = np.eye(len(A))[i]
        bi = b + e
        res = linprog(-A[i], A, bi, bounds=(None, None), method="simplex")
        if -res.fun <= b[i]:
            # constraint is redundant
            A = np.delete(A, i, axis=0)
            b = np.delete(b, i, axis=0)
        else:
            i += 1
    return A, b
