from collections.abc import Collection
from secrets import token_hex

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Iterator
from jax.scipy.special import betaln
from jaxtyping import Scalar, ScalarLike

from .path import Path, get_path
from .pexp import PExp


def unique_strs(col: Collection[str], k: int = 1, ell: int = 8) -> list[str]:
    "return k unique strings of length 2 * ell which are not in col"
    # each byte is converted to two hex digits, so the length of the string is 2 * ell
    ret = []
    while len(ret) < k:
        s = token_hex(ell)
        if s not in col and s not in ret:
            ret.append(s)
    return ret


def migrations_in(demo: dict, t0: Path, t1: Path) -> Iterator[tuple[str, str]]:
    a = get_path(demo, t0)
    b = get_path(demo, t1)
    for m in demo["migrations"]:
        c = m["end_time"]
        d = m["start_time"]
        if max(a, c) < min(b, d):
            yield (m["source"], m["dest"])


def migration_rate(
    demo: dict, source: str, dest: str, t: Scalar | float
) -> Scalar | float:
    ret = 0.0
    for m in demo["migrations"]:
        if m["source"] == source and m["dest"] == dest:
            ret = jnp.where(
                (m["end_time"] <= t) & (t < m["start_time"]), m["rate"], ret
            )
    return ret


def coalescent_rates(demo: dict) -> dict[str, PExp]:
    ret = {}
    for d in demo["demes"]:
        t = []
        N0 = []
        N1 = []
        for e in d["epochs"][::-1]:
            if e["size_function"] == "linear":
                raise NotImplementedError("linear size function is not implemented yet")
            t.append(e["end_time"])
            N0.append(e["end_size"])
            N1.append(e["start_size"])
        t.append(d["start_time"])
        ret[d["name"]] = PExp(N0=jnp.array(N0), N1=jnp.array(N1), t=jnp.array(t))
    return ret


def log_hypergeom(k, M, n, N):
    """
    Returns the log of hyper geometric coefficient
    k: number of selected Type I objects
    M: total number of objects
    n: total number of Type I objects
    N: Number of draws without replacement from the total population
    """
    # https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_discrete_distns.py
    tot, good = M, n
    bad = tot - good
    result = (
        betaln(good + 1, 1)
        + betaln(bad + 1, 1)
        + betaln(tot - N + 1, N + 1)
        - betaln(k + 1, good - k + 1)
        - betaln(N - k + 1, bad - N + k + 1)
        - betaln(tot + 1, 1)
    )
    return result


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = jax.tree.flatten(tree)
    if leaves == []:
        return []
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def B_matrix(n: int, d: int):
    return np.array(list(np.ndindex((n + 1,) * d))).reshape((n + 1,) * d + (d,))
