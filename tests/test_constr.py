"test constraints"

import numpy as np

from demesinfer.constr import constraints, constraints_for
from demesinfer.event_tree import EventTree


def test_reduced(iwm):
    et = EventTree(iwm.model.to_demes())
    cons = constraints_for(et, [("migrations", 0, "rate")])
    A, b = cons["eq"]
    G, h = cons["ineq"]
    assert A.shape == (0, 1)
    assert b.shape == (0,)
    for x in [0.0, 0.5, 1.0]:
        assert (G @ np.array([x]) <= h).all()
    for x in [-1.0, 2.0]:
        assert (G @ np.array([x]) > h).any()
