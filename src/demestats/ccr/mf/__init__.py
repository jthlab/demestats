from dataclasses import dataclass, field

import demes
import jax

from ..curve import CCRCurveBase
from . import events, lift


@jax.tree_util.register_dataclass
@dataclass
class CCRMeanFieldCurve(CCRCurveBase):
    demo: demes.Graph
    k: int = field(metadata=dict(static=True))

    events_mod = events
    lift_mod = lift


__all__ = ["CCRMeanFieldCurve", "events", "lift"]
