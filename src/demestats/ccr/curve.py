from functools import partial
from typing import Callable, ClassVar, Protocol

import demes
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, ArrayLike, Float, Int, Scalar, ScalarLike

import demestats.event_tree as event_tree
from demestats.path import Path
from demestats.traverse import traverse

from ..iicr.interp import MergedInterp


class _LiftModule(Protocol):
    def execute(self, *args, **kwargs):  # pragma: no cover - protocol stub
        ...


class _EventsModule(Protocol):
    State: object


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


def _validate_num_samples(
    demo: demes.Graph,
    k: int,
    num_samples: dict[str, tuple[Int[ScalarLike, ""], Int[ScalarLike, ""]]],
) -> None:
    pops = {pop.name for pop in demo.demes}
    if not (num_samples.keys() <= pops):
        raise ValueError(
            "num_samples must contain only deme names from the demo; found "
            f"{num_samples.keys() - pops}"
        )
    total = 0
    for pop, (nr, nb) in num_samples.items():
        nr = int(nr)
        nb = int(nb)
        if nr < 0 or nb < 0:
            raise ValueError(f"Invalid CCR sample counts for {pop}: {(nr, nb)}")
        total += nr + nb
    if total != k:
        raise ValueError(f"CCR requires sum(red+blue) == k; got sum={total} and k={k}")


def _call(
    et: event_tree.EventTree,
    k: int,
    demo: dict,
    num_samples: dict[str, tuple[int | Scalar, int | Scalar]],
    aux: dict,
    events_mod: _EventsModule,
    lift_mod: _LiftModule,
    scan_over_lifts: bool,
) -> Callable[[ScalarLike, dict], dict[str, Scalar]]:
    states: dict[tuple[event_tree.Node, ...], object] = {}
    for pop in et.leaves:
        num_samples.setdefault(pop, (0, 0))
    d = events_mod.State.setup(num_samples, k)
    for pop, node in et.leaves.items():
        states[node,] = d[pop]

    def node_callback(node, node_attrs, **kw):
        kw["demo"] = demo
        return node_attrs["event"](**kw)

    lift_callback = partial(lift_mod.execute, demo=demo)
    _, auxs = traverse(
        et,
        states,
        node_callback,
        lift_callback,
        aux=aux,
        scan_over_lifts=scan_over_lifts,
    )
    interps = [d["interp"] for d in auxs.values() if "interp" in d]
    return MergedInterp(interps)


class CCRCurveBase:
    events_mod: ClassVar[_EventsModule]
    lift_mod: ClassVar[_LiftModule]
    scan_over_lifts: ClassVar[bool] = False

    demo: demes.Graph
    k: int

    @property
    def events(self) -> _EventsModule:
        return type(self).events_mod

    @property
    def lift(self) -> _LiftModule:
        return type(self).lift_mod

    @property
    def et(self) -> event_tree.EventTree:
        return event_tree.EventTree(self.demo, events=self.events, _merge_contemp=True)

    def variable_for(self, path: Path) -> event_tree.Variable:
        return self.et.variable_for(path)

    @property
    def variables(self):
        return self.et.variables

    def bind(self, params: dict[event_tree.Variable, ScalarLike]) -> dict:
        return self.et.bind(params, rescale=True)

    def __call__(
        self,
        t: Float[ArrayLike, "T"],
        num_samples: dict[str, tuple[Int[ScalarLike, ""], Int[ScalarLike, ""]]],
        params: dict[event_tree.Variable, ScalarLike] = {},
    ) -> dict[str, Float[Array, "T"]]:
        f = self.curve(num_samples, params)
        t = jnp.asarray(t)
        if t.ndim == 0:
            return f(t)
        return self._map_times(f, t)

    def curve(
        self,
        num_samples: dict[str, tuple[Int[ScalarLike, ""], Int[ScalarLike, ""]]],
        params: dict[event_tree.Variable, ScalarLike] = {},
    ) -> _BoundCurve:
        _validate_num_samples(self.demo, self.k, num_samples)
        demo = self.bind(params)
        et = self.et
        self._check_state_space(et)
        f = _call(
            et,
            self.k,
            demo,
            dict(num_samples),
            self._setup_aux(et),
            self.events,
            self.lift,
            self.scan_over_lifts,
        )
        return _BoundCurve(f=f, demo=demo, scaling_factor=et.scaling_factor)

    def _map_times(self, f, t: Float[Array, "T"]) -> dict[str, Float[Array, "T"]]:
        return jax.vmap(f)(t)

    def _setup_aux(self, et: event_tree.EventTree) -> dict:
        return {}

    def _check_state_space(self, et: event_tree.EventTree) -> None:
        return None


__all__ = ["CCRCurveBase"]
