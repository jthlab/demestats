import jax.numpy as jnp

from .pexp import PExp


def migration_rate(demo: dict, source: str, dest: str, t: float) -> float:
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
