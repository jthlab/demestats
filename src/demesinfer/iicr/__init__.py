from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import demes
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Sequence
from jaxtyping import Array, ArrayLike, Float, Int, Scalar, ScalarLike

import demesinfer.event_tree as event_tree
import demesinfer.iicr.events as events
from demesinfer.path import Path
from demesinfer.traverse import traverse


@jax.tree_util.register_dataclass
@dataclass
class IICRCurve:
    demo: demes.Graph
    k: int = field(metadata=dict(static=True))

    @property
    def et(self) -> event_tree.EventTree:
        return event_tree.EventTree(self.demo, events=events, _merge_contemp=True)

    def _setup_aux(
        self, et: event_tree.EventTree
    ) -> dict[tuple[event_tree.Node, ...], dict]:
        setup_state = {(node,): None for leaf, node in et.leaves.items()}
        return traverse(
            et,
            setup_state,
            node_callback=lambda node, node_attrs, **kw: (None, {}),
            lift_callback=partial(events.setup_lift, demo=et.demodict),
            aux=None,
        )[1]

    def variables(self) -> Sequence[event_tree.Variable]:
        """
        Return the parameters that can be optimized.
        """
        return self.et.variables()

    @jax.jit
    def __call__(
        self,
        t: Float[ArrayLike, "T"],
        num_samples: dict[str, Int[ScalarLike, ""]],
        params: dict[event_tree.Variable, ScalarLike] = {},
    ) -> dict[str, Float[Array, "T"]]:
        f = self.curve(num_samples, params)
        return jax.vmap(f)(t)

    def curve(
        self,
        num_samples: dict[str, Int[ScalarLike, ""]],
        params: dict[event_tree.Variable, ScalarLike] = {},
    ) -> Callable[[ScalarLike], dict[str, Scalar]]:
        pops = {pop.name for pop in self.demo.demes}
        assert num_samples.keys() <= pops, (
            "num_samples must contain only deme names from the demo, found {} which is not in {}".format(
                num_samples.keys() - {pop.name for pop in self.demo.demes}, pops
            )
        )
        demo = self.bind(params)
        et = self.et
        f = _call(
            et,
            self.k,
            demo,
            num_samples,
            self._setup_aux(et),
        )

        def g(u, scaling_factor=et.scaling_factor):
            d = f(u / scaling_factor)
            return dict(c=d["c"] / scaling_factor, log_s=d["log_s"])

        return g

    def bind(self, params: dict[event_tree.Variable, ScalarLike]) -> dict:
        """
        Bind the parameters to the event tree's demo.
        """
        return self.et.bind(params, rescale=True)


def _call(
    et: event_tree.EventTree,
    k: int,
    demo: dict,
    num_samples: dict[str, int | Scalar],
    aux: dict,
) -> Callable[[ScalarLike], dict[str, Scalar]]:
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

    _, auxs = traverse(et, states, node_callback, lift_callback, aux=aux)

    def f(t, auxs=auxs):
        c = log_s = 0.0
        for d in auxs.values():
            if "lift" not in d:
                continue
            y = d["lift"](t)
            mask = (d["t0"] <= t) & (t < d["t1"])
            c += y["c"] * mask
            log_s += y["log_s"] * mask
        return dict(c=c, log_s=log_s)

    return f
