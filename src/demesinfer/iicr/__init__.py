from dataclasses import dataclass, field
from functools import partial

import demes
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from beartype.typing import Callable, Sequence
from jaxtyping import Array, ArrayLike, Float, Int, Scalar, ScalarLike
from penzai import pz

import demesinfer.event_tree as event_tree
import demesinfer.iicr.events as events
from demesinfer.path import Path
from demesinfer.traverse import traverse

from .interp import FilterInterp


class _BoundCurve(eqx.Module):
    f: Callable[[ScalarLike], dict[str, Scalar]]
    scaling_factor: ScalarLike
    demo: dict

    def __call__(self, u: ScalarLike) -> dict:
        d = self.f(u / self.scaling_factor, self.demo)
        return dict(c=d["c"] / self.scaling_factor, log_s=d["log_s"])

    @property
    def jump_ts(self) -> Float[Array, "J"]:
        return self.f.jumps(self.demo) * self.scaling_factor

    def quantile(self, q: ScalarLike) -> Scalar:
        # s = 1 - q => log_s = log(1 - q)
        lq = jnp.log1p(-q)

        def f(t, _):
            return self(t)["log_s"] - lq

        solver = optx.Newton(rtol=1e-4, atol=1e-4)
        sol = optx.root_find(f, solver, 1.0)
        return sol.value


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

    def variable_for(self, path: Path) -> event_tree.Variable:
        """Return the variable associated with a given path."""
        return self.et.variable_for(path)

    @property
    def variables(self) -> Sequence[event_tree.Variable]:
        """
        Return the parameters that can be optimized.
        """
        return self.et.variables

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
    ) -> _BoundCurve:
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

        return _BoundCurve(f=f, demo=demo, scaling_factor=et.scaling_factor)

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
) -> Callable[[ScalarLike, dict], dict[str, Scalar]]:
    """Call the IICR curve with a time and number of samples."""
    states = {}
    i = -1
    e = jnp.zeros(k + 1)
    for pop, node in et.leaves.items():
        p = e.at[num_samples.get(pop, 0)].set(1.0)
        p = pz.nx.wrap(p, pop)
        states[node,] = events.State(
            p=p,
            log_s=jnp.array(0.0),
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
    interps = [d["lift"] for d in auxs.values() if "lift" in d]
    return FilterInterp(interps)
