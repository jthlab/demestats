import pickle

import jax
import jax.numpy as jnp
import msprime as msp
import numpy as np
import stdpopsim as sps

from demestats.constr import constraints_for
from demestats.event_tree import EventTree
from demestats.iicr import IICRCurve


def test_missing_epoch_bug():
    # Create demography object
    demo = msp.Demography()
    # Add populations
    demo.add_population(initial_size=3000, name="anc")
    demo.add_population(initial_size=1000, name="P0")
    demo.add_population(initial_size=1000, name="P1")
    # Set initial migration rate
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
    # population growth at 500 generations
    demo.add_population_parameters_change(
        time=500,
        initial_size=3000,  # Bottleneck: reduce to 1000 individuals
        population="P0",
    )
    demo.add_population_parameters_change(
        time=500,
        initial_size=3000,  # Bottleneck: reduce to 1000 individuals
        population="P1",
    )
    # Migration rate change changed to 0.001 AFTER 500 generation (going into the past)
    demo.add_migration_rate_change(time=500, rate=0.001, source="P0", dest="P1")
    demo.add_migration_rate_change(time=500, rate=0.001, source="P1", dest="P0")
    # THEN add the older events (population split at 1000)
    demo.add_population_split(time=1000, derived=["P0", "P1"], ancestral="anc")

    et = EventTree(demo.to_demes())
    fs = frozenset(
        {
            ("demes", 1, "epochs", 1, "end_time"),
            ("demes", 2, "epochs", 1, "end_time"),
            ("migrations", 2, "end_time"),
            ("migrations", 3, "end_time"),
        }
    )
    assert fs in et.variables

    cons = constraints_for(
        et,
        frozenset(
            {
                ("demes", 0, "epochs", 0, "end_time"),
                ("demes", 1, "start_time"),
                ("demes", 2, "start_time"),
                ("migrations", 0, "start_time"),
                ("migrations", 1, "start_time"),
            }
        ),
    )

    G, h = cons["ineq"]
    y = np.array([999.0])
    assert np.all(G @ y <= h)
    y = np.array([400.0])
    assert np.any(G @ y > h)


def test_neg_coal_rate_bug1():
    iwm = sps.IsolationWithMigration(
        NA=1e4, N1=1e3, N2=2e3, T=1e5, M12=0.1 / 4e4, M21=0.1 / 4e4
    )
    ii = IICRCurve(iwm.model.to_demes(), 2)
    x = 0.0005179474679231213
    v = ii.variable_for(("migrations", 0, "rate"))
    curve = ii.curve(params={v: x}, num_samples={"pop1": 1, "pop2": 1})
    t = jnp.geomspace(1e2, 1e6, 100)
    y = jax.vmap(curve)(t)["c"]
    assert (y >= 0).all()


def test_neg_coal_rate_bug2():
    d = pickle.load(open("tests/assets/bug1.pkl", "rb"))
    ii = IICRCurve(d["g"], 2)
    crv = ii.curve(d["ns"], d["params"])
    c = crv(197.0)
    assert c["c"] >= 0


def test_neg_coal_rate_bug3():
    d = pickle.load(open("tests/assets/bug2.pkl", "rb"))
    ii = IICRCurve(d["g"], 2)

    def f(t, ns):
        c = ii.curve(ns, d["params"])
        return jax.vmap(c)(t)

    jax.vmap(f)(d["times"], d["ns"])
