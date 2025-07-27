from dataclasses import dataclass, field
from functools import partial

import demes
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int, Scalar, ScalarLike

import demesinfer.event_tree as event_tree
import demesinfer.iicr.events as events
from demesinfer.path import Path, bind
from demesinfer.traverse import traverse


@dataclass
class IICRCurve:
    demo: demes.Graph
    k: int
    et: event_tree.EventTree = field(init=False)

    def __post_init__(self):
        self.et = event_tree.EventTree(self.demo, events=events)
        self._aux = self._setup()

    def _setup(self) -> dict[tuple[event_tree.Node, ...], dict]:
        et = self.et
        setup_state = {(node,): None for leaf, node in et.leaves.items()}
        _, aux = traverse(
            self.et,
            setup_state,
            node_callback=lambda node, node_attrs, **kw: (None, {}),
            lift_callback=partial(events.setup_lift, demo=et.demodict),
            aux=None,
        )
        return aux

    def __call__(
        self,
        t: Float[ArrayLike, "*T"],
        num_samples: dict[str, Int[ScalarLike, ""]],
        params: dict[Path, ScalarLike] = {},
    ) -> dict[str, Float[Array, "*T"]]:
        pops = {pop.name for pop in self.demo.demes}
        assert num_samples.keys() <= pops, (
            "num_samples must contain only deme names from the demo, found {} which is not in {}".format(
                num_samples.keys() - {pop.name for pop in self.demo.demes}, pops
            )
        )
        state = _call(
            jnp.atleast_1d(t),
            self.et,
            self.k,
            bind(self.et.demodict, params),
            num_samples,
            self._aux,
        )
        ret = dict(c=state.c, log_s=state.log_s)
        return jax.tree.map(lambda a: a.reshape(t.shape), ret)


@eqx.filter_vmap(in_axes=(0,) + (None,) * 5)
def _call(
    t: Float[Array, ""],
    et: event_tree.EventTree,
    k: int,
    demo: dict,
    num_samples: dict[str, int | Scalar],
    aux: dict,
):
    """Call the IICR curve with a time and number of samples."""
    states = {}
    i = -1
    for pop, node in et.leaves.items():
        # idea here is that the state is a (d+1, d+1, ..., d+1)-tensor where
        # T[i, j, ..., k] is the probability that lineage 1 is in deme i, lineage 2 is in deme j, etc.
        # deme d+1 is a special deme that represents the "outside" deme
        e = jnp.ones(k, dtype=jnp.int32)
        for pop1 in num_samples:
            for j in range(k):
                accept = (pop1 == pop) & (j < num_samples[pop1])
                i = jnp.where(accept, i + 1, i)
                e1 = e.at[i].set(0)
                e = jnp.where(accept, e1, e)
        p = jnp.zeros((2,) * k).at[tuple(e)].set(1.0)
        states[node,] = events.State(
            p=p,
            log_s=jnp.array(0.0),
            c=jnp.array(0.0),
            t=t,
            pops=(pop,),
        )

    def node_callback(node, node_attrs, **kw):
        kw["demo"] = demo
        return node_attrs["event"](**kw)

    def lift_callback(
        state, t0: Path, t1: Path, terminal: bool, aux: dict
    ) -> tuple[events.State, dict]:
        return events.lift(
            state=state, t0=t0, t1=t1, terminal=terminal, demo=demo, aux=aux
        )

    states, _ = traverse(
        et, states, node_callback, lift_callback, aux=aux, _fuse_lifts=True
    )
    return states[et.root,]
