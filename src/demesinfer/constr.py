"Constraints from event tree"

from sympy import symbols, Symbol, Expr
import numpy as np
from itertools import count
from scipy.optimize import linprog

from .event_tree import EventTree


def constraints(et: EventTree):
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

    taus = symbols("tau_0:%d" % len(all_times))
    for tau_i, (t, paths) in zip(taus, sorted(all_times.items())):
        eq.append(tau_i - t)
        mapping[tau_i] = paths

    path_times = {
        p: tau_i
        for tau_i, (_, paths) in zip(taus, sorted(all_times.items()))
        for p in paths
    }

    # next, require that times are increasing along edges
    for edge in et.edges:
        t_u, t_v = [path_times[et.nodes[x]["t"]] for x in edge]
        ineq.append(t_u - t_v)

    # other parameters
    demo = et._demo
    # sizes
    for i, d in enumerate(demo.demes):
        for j, e in enumerate(d.epochs):
            eta0, eta1 = symbols([f"eta_{{{i}{j}{x}}}" for x in "01"])
            mapping[eta0] = ("demes", i, "epochs", "j", "start_size")
            mapping[eta1] = ("demes", i, "epochs", "j", "end_size")
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
    k = count(0)
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
    A = _to_matrix(eq, indets)
    A_red, b_red = _reduce_equalities(A, np.zeros(len(eq)))
    G = _to_matrix(ineq, indets)
    h = np.zeros(len(ineq))
    G_red, h_red = _reduce_inequalities(G, h)

    # I decided not to use these
    eq = _to_symbolic(A_red, b_red, indets, eq=True)
    ineq = _to_symbolic(G_red, h_red, indets, eq=False)

    return dict(ineq=(G_red, h_red), eq=(A_red, b_red), mapping=mapping)


def _to_symbolic(A, b, indets):
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


def _to_matrix(sys: list[Expr], indets: list[Symbol]) -> tuple[np.ndarray, np.ndarray]:
    ret = []
    for expr in sys:
        row = [0.0] * len(indets)
        cd = expr.as_coefficients_dict()
        for i, indet in enumerate(indets):
            row[i] = float(cd.get(indet, 0))
        ret.append(row)
    return np.array(ret)


def _reduce_equalities(A, b):
    """Reduce the system of equations Ax = b by removing redundant equalities."""
    U, s, Vt = np.linalg.svd(A)
    rank = np.sum(s > 1e-10)
    A_red = U[:, :rank].T @ A
    b_red = U[:, :rank].T @ b
    return A_red, b_red


def _reduce_inequalities(A, b):
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
