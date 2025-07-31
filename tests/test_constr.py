"test constraints"

import numpy as np
import pytest

from demesinfer.constr import constraints_for
from demesinfer.event_tree import EventTree


def test_proportions(iwm):
    et = EventTree(iwm.model.to_demes())

    # proportions
    cons = constraints_for(et, ("demes", 1, "proportions", 0))
    A, b = cons["eq"]
    G, h = cons["ineq"]
    np.testing.assert_allclose(A, [[1.0]])
    np.testing.assert_allclose(b, [1.0])
    assert G.size == h.size == 0


def test_no_var(iwm):
    et = EventTree(iwm.model.to_demes())
    # sizes
    with pytest.raises(ValueError):
        cons = constraints_for(et, ("demes", 0, "start_size"))


def test_sizes(iwm):
    et = EventTree(iwm.model.to_demes())
    # sizes
    cons = constraints_for(
        et,
        {
            ("demes", 0, "epochs", 0, "start_size"),
            ("demes", 0, "epochs", 0, "end_size"),
        },
    )
    A, b = cons["eq"]
    G, h = cons["ineq"]
    assert A.size == b.size == 0
    np.testing.assert_allclose(G, [[-1.0]])
    np.testing.assert_allclose(h, [0.0])


def test_migration(iwm):
    et = EventTree(iwm.model.to_demes())
    # migration
    cons = constraints_for(et, ("migrations", 0, "rate"), ("migrations", 1, "rate"))
    A, b = cons["eq"]
    G, h = cons["ineq"]
    assert A.size == b.size == 0
    np.testing.assert_allclose(G, [[-1.0, 0.0], [0.0, -1.0]])
    np.testing.assert_allclose(h, [0.0, 0.0])
