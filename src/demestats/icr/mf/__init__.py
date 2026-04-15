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
class ICRMeanFieldCurve:
    """
    Build an ICRMeanFieldCurve object that can be later used to evaluate the mean-field approximation
    of the instantaneous coalescence rate (ICR) through time for a sampling configuration and demographic model. 

    Parameters
    ----------
        demo : demes.Graph
            A ``demes`` graph describing the demographic history.
        k : int
            The number of sampled lineages used to define the ICR curve.

    Returns
    -------
        ICRMeanFieldCurve
            An ``ICRMeanFieldCurve`` object that can be called directly on a time grid, 
            sampling configuration, and parameter value to construct the mean-field approximation of the ICR curve.

    Notes
    -----
    From a user perspective, understanding the underlying structure of an ICRMeanFieldCurve
    object is not necessary. The only function that a user would use is ``ICRMeanFieldCurve.__call__``,
    which evaluates the ICR on a grid of time points given a sampling configuration.

    One can choose between two computational backends
    depending on the size of the problem, see "ICR: Exact vs Mean-Field" in the ICR tutorial.

    ::
        # pairwise coalescence uses k = 2
        icr = ICRMeanFieldCurve(demo.to_demes(), k=2)

    See Also
    --------
    demestats.icr.ICRMeanFieldCurve.__call__
    demestats.icr.ICRCurve
    """
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
        """
        Evaluate the ICR curve on a collection of time points given a specified
        sampling configuration and parameters using the mean-field approximation.

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
        You must first construct an ICRMeanFieldCurve object. See the `ICRMeanFieldCurve` API.

        ::
            import jax.numpy as jnp

            # pairwise coalescence uses k = 2
            icr_mf = ICRMeanFieldCurve(demo.to_demes(), k=2)
            t=jnp.geomspace(1.0, 5000., 250)
            sampling_config = {"P0": 1, "P1": 1}
            expected_icr_mf = icr_mf(params={}, t=t, num_samples = sampling_config)

            # you can optionally use a one liner
            expected_icr_mf = ICRMeanFieldCurve(demo=g, k=2)(param={}, t=t, num_samples=sampling_config)

        See Also
        --------
        demestats.icr.ICRMeanFieldCurve
        """
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


__all__ = ["ICRMeanFieldCurve", "events"]
