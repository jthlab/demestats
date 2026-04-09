from dataclasses import dataclass, field
from functools import partial

import demes
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from beartype.typing import Callable
from jaxtyping import Array, ArrayLike, Float, Int, Scalar, ScalarLike

import demestats.event_tree as event_tree
from demestats.path import Path
from demestats.traverse import traverse

from ..interp import MergedInterp
from . import events


class _BoundCurve(eqx.Module):
    f: Callable[[ScalarLike, dict], dict[str, Scalar]]
    scaling_factor: ScalarLike
    demo: dict

    def __call__(self, u: ScalarLike) -> dict:
        d = self.f(u / self.scaling_factor, self.demo)
        return dict(c=d["c"] / self.scaling_factor, log_s=d["log_s"])

    @property
    def jump_ts(self) -> Float[Array, "J"]:
        return self.f.jumps(self.demo) * self.scaling_factor

    def quantile(self, q: ScalarLike) -> Scalar:
        lq = jnp.log1p(-q)

        def f(t, _):
            return self(t)["log_s"] - lq

        solver = optx.Newton(rtol=1e-4, atol=1e-4)
        sol = optx.root_find(f, solver, 1.0)
        return sol.value


@jax.tree_util.register_dataclass
@dataclass
class IICRMeanFieldCurve:
    demo: demes.Graph
    k: int = field(metadata=dict(static=True))

    @property
    def et(self) -> event_tree.EventTree:
        return event_tree.EventTree(self.demo, events=events, _merge_contemp=True)

    def variable_for(self, path: Path) -> event_tree.Variable:
        return self.et.variable_for(path)

    @property
    def variables(self):
        return self.et.variables

    def bind(
        self, params: dict[event_tree.Variable | tuple[str | int, ...], ScalarLike]
    ) -> dict:
        return self.et.bind(params, rescale=True)

    def __call__(
        self,
        t: Float[ArrayLike, "T"],
        num_samples: dict[str, Int[ScalarLike, ""]],
        params: dict[event_tree.Variable | tuple[str | int, ...], ScalarLike] = {},
    ) -> dict[str, Float[Array, "T"]]:
        f = self.curve(num_samples, params)
        return jax.vmap(f)(t)

    def curve(
        self,
        num_samples: dict[str, Int[ScalarLike, ""]],
        params: dict[event_tree.Variable | tuple[str | int, ...], ScalarLike] = {},
    ) -> _BoundCurve:
        pops = {pop.name for pop in self.demo.demes}
        assert num_samples.keys() <= pops, (
            "num_samples must contain only deme names from the demo, found {} which is not in {}".format(
                num_samples.keys() - pops, pops
            )
        )
        demo = self.bind(params)
        et = self.et
        f = _call(et, self.k, demo, dict(num_samples))
        return _BoundCurve(f=f, demo=demo, scaling_factor=et.scaling_factor)


def _call(
    et: event_tree.EventTree,
    k: int,
    demo: dict,
    num_samples: dict[str, int | Scalar],
):
    states = {}
    for pop in et.leaves:
        num_samples.setdefault(pop, 0)
    d = events.State.setup(num_samples, k)
    for pop, node in et.leaves.items():
        states[node,] = d[pop]

    def node_callback(node, node_attrs, **kw):
        kw["demo"] = demo
        return node_attrs["event"](**kw)

    lift_callback = partial(events.lift.execute, demo=demo)
    _, auxs = traverse(
        et,
        states,
        node_callback=node_callback,
        lift_callback=lift_callback,
        aux={},
        scan_over_lifts=False,
    )
    interps = [d["interp"] for d in auxs.values() if "interp" in d]
    return MergedInterp(interps)


__all__ = ["IICRMeanFieldCurve", "events"]
