from collections.abc import Collection
from secrets import token_hex

import jax.numpy as jnp
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


def constant_growth_in(demo: dict, t0: Path, t1: Path) -> Iterator[tuple[str, bool]]:
    a = get_path(demo, t0)
    b = get_path(demo, t1)
    for d in demo["demes"]:
        start_time = d["start_time"]
        for e in d["epochs"]:
            u = start_time
            v = e["end_time"]
            if max(a, u) < min(b, v):
                if e["size_function"] != "constant":
                    yield d["name"], False
            start_time = v
        yield d["name"], True


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
