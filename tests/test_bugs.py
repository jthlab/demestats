import msprime as msp
import numpy as np

from demesinfer.constr import constraints_for
from demesinfer.event_tree import EventTree


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
