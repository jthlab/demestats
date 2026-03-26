import numpy as np
import pytest

from demestats.sfs import ExpectedSFS
from demestats.tsfs import ExpectedTSFS

from .demos import SingleDeme, TwoDemes


def test_tsfs_time_grid_validation():
    g, _ = SingleDeme.Constant().base()
    ns = {"A": 4}

    with pytest.raises(ValueError, match="start at 0"):
        ExpectedTSFS(g, num_samples=ns, time_grid=[1.0, np.inf])

    with pytest.raises(ValueError, match="end at \\+inf"):
        ExpectedTSFS(g, num_samples=ns, time_grid=[0.0, 1.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        ExpectedTSFS(g, num_samples=ns, time_grid=[0.0, 1.0, 1.0, np.inf])


def test_tsfs_sum_matches_sfs_1d():
    g, _ = SingleDeme.Exponential().base()
    ns = {"A": 8}
    grid = [0.0, 0.25, 1.0, np.inf]

    tsfs = ExpectedTSFS(g, num_samples=ns, time_grid=grid)()
    sfs = ExpectedSFS(g, num_samples=ns)()

    assert tsfs.shape == (len(grid) - 1, ns["A"] + 1)
    np.testing.assert_allclose(tsfs.sum(axis=0), sfs, rtol=1e-5, atol=1e-6)
    assert np.allclose(tsfs[:, 0], 0.0)
    assert np.allclose(tsfs[:, -1], 0.0)


def test_tsfs_sum_matches_sfs_with_pruning():
    g, _ = TwoDemes.Constant().base()
    ns = {"A": 6, "B": 5}
    prune = {"A": 3}
    grid = [0.0, 0.1, 0.6, 1.2, np.inf]

    tsfs = ExpectedTSFS(g, num_samples=ns, time_grid=grid, prune=prune)()
    sfs = ExpectedSFS(g, num_samples=ns, prune=prune)()

    assert tsfs.shape == (len(grid) - 1, ns["A"] + 1, ns["B"] + 1)
    np.testing.assert_allclose(tsfs.sum(axis=0), sfs, rtol=1e-5, atol=1e-6)
    assert np.allclose(tsfs[:, 0, 0], 0.0)
    assert np.allclose(tsfs[:, -1, -1], 0.0)


def test_tsfs_rescaling():
    g1, _ = TwoDemes.Exponential(t=1.0, size=1.0, g=1.0).migration(
        tstart=1.0, tend=0.5, rate=0.05
    )
    g2, _ = TwoDemes.Exponential(t=1e3, size=1e3, g=1e-3).migration(
        tstart=1.0 * 1e3, tend=0.5 * 1e3, rate=0.05 / 1e3
    )
    ns = {"A": 5, "B": 4}
    grid1 = [0.0, 0.25, 0.75, np.inf]
    grid2 = [0.0, 0.25 * 1e3, 0.75 * 1e3, np.inf]

    tsfs1 = ExpectedTSFS(g1, num_samples=ns, time_grid=grid1)()
    tsfs2 = ExpectedTSFS(g2, num_samples=ns, time_grid=grid2)()

    np.testing.assert_allclose(tsfs1 * 1e3, tsfs2, rtol=1e-5, atol=1e-6)
