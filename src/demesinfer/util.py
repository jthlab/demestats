import jax.numpy as jnp


def migration_rate(params: dict, source: str, dest: str, t: float) -> float:
    migrations = params["migrations"]
    ret = 0.0
    for m in params["migrations"]:
        if m["source"] == source and m["dest"] == dest:
            ret = jnp.where(
                (m["end_time"] <= t) & (t < m["start_time"]), m["rate"], ret
            )
    return ret


def coalescent_rate(params: dict, pop: str, t: float) -> float:
    ret = jnp.nan
    for d in params["demes"]:
        if d["name"] == pop:
            for e in d["epochs"]:
                start_size = e["start_size"]
                end_size = e["end_size"]
                sf = e["size_function"]
                dt = (start_time - t) / (start_time - end_time)
                if sf == "constant":
                    N = jnp.where(in_epoch, end_size, ret)
                elif sf == "exponential":
                    r = jnp.log(end_size / start_size)
                    N = start_size * math.exp(r * dt)
                elif sf == "linear":
                    N = start_size + (end_size - start_size) * dt
                else:
                    raise NotImplementedError(
                        f"unknown size_function '{epoch.size_function}'"
                    )
                ret = jnp.where((end_time <= t) & (t < start_time), N, ret)
    return ret
