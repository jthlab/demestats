import demes
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

import demesinfer.iicr.events as events
import demesinfer.event_tree
from demesinfer.traverse import traverse

@dataclass
class IICRCurve:
    demo: demes.Graph
    k: int
    _et: demesinfer.event_tree.EventTree = field(init=False)

    def __post_init__(self):
        self._et = demesinfer.event_tree.EventTree(self.demo, events=events)
        self._aux = pass

    def __call__(self, t: float, num_samples: dict[str, int]):
        return _call(
            et=self._et,
            t=t,
            num_samples=num_samples,
        )


@partial(jax.jit, static_argnums=(0, 1))
def _call(
    et: EventTree,
    k: int,
    params: dict,
    t: float,
    num_samples: dict[str, int],
):
    """Call the IICR curve with a time and number of samples."""
    for node in et.leaves:
        pop = list(et.nodes[node]["block"])[0]
        # idea here is that the state is a (d+1, d+1, ..., d+1)-tensor where
        # T[i, j, ..., k] is the probability that lineage 1 is in deme i, lineage 2 is in deme j, etc.
        # deme d+1 is a special deme that represents the "outside" deme
        e = jnp.ones(k, dtype=jnp.int32)
        for p in num_samples:
            for j in range(k):
                accept = (p == pop) & (j < num_samples[p])
                i = jnp.where(accept, i + 1, i)
                e1 = e.at[i].set(0)
                e = jnp.where(accept, e1, e)
        p = jnp.zeros((2,) * k).at[tuple(e)].set(1.0)
        states[node] = events.State(
            p=p, log_s=jnp.zeros((k,) * k), c=jnp.zeros((k,) * k), t=t, 
        )

    states = traverse(et, states, node_callback, lift_callback, _fuse_lifts=True)
    return states[et.root]
