"""Exact CCR implementation (colored lineage-count CTMC)."""

import os
from dataclasses import dataclass, field
from functools import partial

import demes
import jax

import demestats.event_tree as event_tree
from demestats.traverse import traverse

from ...iicr.state import SetupState
from ..curve import CCRCurveBase
from . import events, interp, lift, state


@jax.tree_util.register_dataclass
@dataclass
class CCRCurve(CCRCurveBase):
    demo: demes.Graph
    k: int = field(metadata=dict(static=True))

    events_mod = events
    lift_mod = lift
    scan_over_lifts = False

    def _setup_aux(
        self, et: event_tree.EventTree
    ) -> dict[tuple[event_tree.Node, ...], dict]:
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
            lift_callback=partial(lift.setup, demo=et.demodict),
            aux=None,
            scan_over_lifts=self.scan_over_lifts,
        )[1]

    def _map_times(self, f, t):
        # Avoid vmapping `expm_multiply` for large CCR state spaces, which can
        # cause massive peak memory usage (especially on GPU).
        max_vmap = int(os.environ.get("DEMESTATS_CCR_VMAP_MAX_T", "32"))
        if t.shape[0] <= max_vmap:
            return jax.vmap(f)(t)
        return jax.lax.map(f, t)


__all__ = ["CCRCurve", "events", "interp", "lift", "state"]
