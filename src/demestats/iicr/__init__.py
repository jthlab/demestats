import os
from dataclasses import dataclass, field
from functools import partial
from types import ModuleType

import demes
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from beartype.typing import Callable, Sequence
from jaxtyping import Array, ArrayLike, Float, Int, Scalar, ScalarLike

import demestats.event_tree as event_tree
from demestats.path import Path
from demestats.traverse import traverse

from . import dn, nd
from .interp import MergedInterp
from .state import SetupState


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
    """
    Build an ICRCurve object that can be later used to evaluate the instantaneous
    coalescence rate (ICR) through time for a sampling configuration and demographic model.

    Parameters
    ----------
        demo : demes.Graph
            A ``demes`` graph describing the demographic history.
        k : int
            The number of sampled lineages used to define the IICR curve.

    Returns
    -------
        IICRCurve
            An ``ICRCurve`` object that can be called directly on a time grid, 
            sampling configuration, and parameter value to construct the ICR curve.

    Notes
    -----
    From a user perspective, understanding the underlying structure of an ICRCurve
    object is not necessary. The only function that a user would use is ``ICRCurve.__call__``,
    which evaluates the ICR on a grid of time points given a sampling configuration.

    One can choose between two computational backends
    depending on the size of the problem, see "ICR: Exact vs Mean-Field" in the ICR tutorial.

    ::
        # pairwise coalescence uses k = 2
        icr = ICRCurve(demo.to_demes(), k=2)

    See Also
    --------
    demestats.iicr.IICRCurve.__call__
    demestats.iicr.IICRMeanFieldCurve
    """
    demo: demes.Graph
    k: int = field(metadata=dict(static=True))

    @property
    def nd_or_dn(self) -> ModuleType:
        if os.environ.get("DEMESTATS_FORCE_IICR") == "ND":
            return nd
        elif os.environ.get("DEMESTATS_FORCE_IICR") == "DN":
            return dn
        d = len(self.demo.demes)
        if (d + 1) ** self.k < self.k**d:
            return dn
        else:
            return nd

    @property
    def events(self) -> ModuleType:
        return self.nd_or_dn.events

    @property
    def et(self) -> event_tree.EventTree:
        return event_tree.EventTree(
            self.demo, events=self.nd_or_dn.events, _merge_contemp=True
        )

    def _setup_aux(
        self, et: event_tree.EventTree
    ) -> dict[tuple[event_tree.Node, ...], dict]:
        events = self.nd_or_dn.events
        setup_state = {
            (node,): SetupState(n=self.k, pops=(leaf,))
            for leaf, node in et.leaves.items()
        }

        def node_cb(node, node_attrs, **kw):
            return node_attrs["event"].setup(**kw, demo=et.demodict)

        return traverse(
            et,
            setup_state,
            node_callback=node_cb,
            lift_callback=partial(events.lift.setup, demo=et.demodict),
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
        """
        Evaluate the ICR curve on a collection of time points given a specified
        sampling configuration and parameters.

        Parameters
        ----------
            t : ArrayLike
                One or more time points at which to evaluate the ICR curve.
            num_samples : dict
                A dictionary specifying the number of sampled haploids for each deme.
                The population names must match the deme names in demographic model.
            params : dict, optional
                A dictionary mapping ``event_tree.Variable`` objects to
                parameter values.

        Returns
        -------
            dict of str to Array
                A dictionary  with the coalescence hazard ``"c"`` and log-survival ``"log_s"`` 
                whose values are evaluated at the input times. 

        Notes
        -----
        You must first construct an ICRCurve object. See the `ICRCurve` API.

        ::
            import jax.numpy as jnp

            # pairwise coalescence uses k = 2
            icr = ICRCurve(demo.to_demes(), k=2)
            t=jnp.geomspace(1.0, 5000., 250)
            sampling_config = {"P0": 1, "P1": 1}
            expected_icr = icr(params={}, t=t, num_samples = sampling_config)

            # you can optionally use a one liner
            expected_icr = ICRCurve(demo=g, k=2)(param={}, t=t, num_samples=sampling_config)

        See Also
        --------
        demestats.iicr.IICRCurve
        """
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
        f = _call(et, self.k, demo, num_samples, self._setup_aux(et), self.events)

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
    events_mod: ModuleType,
) -> Callable[[ScalarLike, dict], dict[str, Scalar]]:
    """Call the IICR curve with a time and number of samples."""
    states = {}
    for pop in et.leaves:
        num_samples.setdefault(pop, 0)
    d = events_mod.State.setup(num_samples, k)
    for pop, node in et.leaves.items():
        states[node,] = d[pop]

    def node_callback(node, node_attrs, **kw):
        kw["demo"] = demo
        return node_attrs["event"](**kw)

    lift_callback = partial(events_mod.lift.execute, demo=demo)

    _, auxs = traverse(et, states, node_callback, lift_callback, aux=aux)
    interps = [d["interp"] for d in auxs.values() if "interp" in d]
    return MergedInterp(interps)


from .mf import IICRMeanFieldCurve

__all__ = ["IICRCurve", "IICRMeanFieldCurve"]
