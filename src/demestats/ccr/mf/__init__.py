from dataclasses import dataclass, field

import demes
import jax

from ..curve import CCRCurveBase
from . import events, lift


@jax.tree_util.register_dataclass
@dataclass
class CCRMeanFieldCurve(CCRCurveBase):
    """
    Build an CCRMeanFieldCurve object that can be later used to evaluate the mean-field approximation
    of the cross-coalescence rate (CCR) through time for a demographic model. 

    Parameters
    ----------
        demo : demes.Graph
            A ``demes`` graph describing the demographic history.
        k : int
            The number of sampled lineages used to define the CCR curve.

    Returns
    -------
        CCRMeanFieldCurve
            An ``CCRMeanFieldCurve`` object that can be called directly on a time grid, 
            sampling configuration, and parameter value to construct the mean-field approximation of the CCR curve.

    Notes
    -----
    From a user perspective, understanding the underlying structure of an CCRMeanFieldCurve
    object is not necessary. The only function that a user would use is ``__call__``,
    which evaluates the CCR on a grid of time points given a sampling configuration.

    One can choose between two computational backends
    depending on the size of the problem, see "CCR: Exact vs Mean-Field" in the CCR tutorial.

    ::
        # pairwise coalescence uses k = 10
        ccr_mf = CCRMeanFieldCurve(demo.to_demes(), k=10)

    See Also
    --------
    demestats.ccr.CCRMeanFieldCurve.__call__
    demestats.ccr.CCRCurve
    """
    demo: demes.Graph
    k: int = field(metadata=dict(static=True))

    events_mod = events
    lift_mod = lift


__all__ = ["CCRMeanFieldCurve", "events", "lift"]
