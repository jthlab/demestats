import itertools as it
from collections import Counter

import demes
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import stdpopsim

from demesinfer.ccr import CCRCurve, CCRMeanFieldCurve
from demesinfer.iicr import IICRCurve

from .conftest import enumerate_stdpopsim_models


def _idfun(x):
    if isinstance(x, stdpopsim.DemographicModel):
        return x.id
    return "".join(x)


def _supported_for_ccr(g: demes.Graph) -> bool:
    # CCR lifting currently requires piecewise-constant population sizes.
    for d in g.demes:
        if any(e.size_function != "constant" for e in d.epochs):
            return False
    return True


def _as_ccr_samples(counter: Counter) -> dict[str, tuple[int, int]]:
    # Encode the k=2 sample configuration as a single red and a single blue lineage.
    items = list(counter.items())
    if len(items) == 1:
        (pop, n) = items[0]
        assert n == 2
        return {pop: (1, 1)}
    assert len(items) == 2
    (p1, n1), (p2, n2) = items
    assert n1 == 1 and n2 == 1
    return {p1: (1, 0), p2: (0, 1)}


def _time_grid_for_pair(g: demes.Graph, pops: Counter) -> np.ndarray:
    names = set(pops)
    start_times = [d.start_time for d in g.demes if d.name in names]
    end_times = [d.end_time for d in g.demes if d.name in names]
    t0 = max(end_times)
    t1 = min(start_times)
    if np.isinf(t1):
        t1 = 5 * t0 if t0 > 0 else 1e4

    t = np.linspace(t0, t1, 200)
    model_times = np.array(
        [epoch.end_time for d in g.demes for epoch in d.epochs]
        + [d.start_time for d in g.demes]
    )
    model_times = model_times[np.isfinite(model_times)]
    model_times = model_times.clip(t0, t1)
    # Remove model times to dodge left/right continuity differences.
    tm = (
        np.isclose(t[:, None], model_times[None, :]).any(axis=1)
        if model_times.size
        else np.zeros_like(t, dtype=bool)
    )
    t = t[~tm]
    if t.size == 0:
        t = np.array([0.5 * (t0 + t1)])
    # Keep it small: CCR state space is much larger than IICR for the same demes.
    return t[:5]


def model_single_deme_constant(N=10_000.0) -> demes.Graph:
    b = demes.Builder()
    b.add_deme("A", epochs=[dict(start_size=N)])
    return b.resolve()


def model_two_demes_isolated(N1=5_000.0, N2=5_000.0) -> demes.Graph:
    # Use an ancestral deme so the event tree has a single root.
    T = 1000.0
    b = demes.Builder()
    b.add_deme("anc", epochs=[dict(start_size=max(N1, N2), end_time=T)])
    b.add_deme("A", ancestors=["anc"], start_time=T, epochs=[dict(start_size=N1)])
    b.add_deme("B", ancestors=["anc"], start_time=T, epochs=[dict(start_size=N2)])
    return b.resolve()


def model_two_demes_sym_mig(N=5_000.0, m=1e-3) -> demes.Graph:
    T = 1000.0
    b = demes.Builder()
    b.add_deme("anc", epochs=[dict(start_size=N, end_time=T)])
    b.add_deme("A", ancestors=["anc"], start_time=T, epochs=[dict(start_size=N)])
    b.add_deme("B", ancestors=["anc"], start_time=T, epochs=[dict(start_size=N)])
    b.add_migration(source="A", dest="B", rate=m, start_time=T, end_time=0.0)
    b.add_migration(source="B", dest="A", rate=m, start_time=T, end_time=0.0)
    return b.resolve()


def test_ccr_constant_single_deme_k2_matches_analytic():
    N = 10_000.0
    g = model_single_deme_constant(N=N)
    ccr = CCRCurve(g, k=2)
    t = jnp.array([0.0, 1.0, 10.0])
    out = ccr(t=t, num_samples={"A": (1, 1)}, params={})
    np.testing.assert_allclose(np.asarray(out["c"]), 1.0 / (2.0 * N), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(out["log_s"]), -np.asarray(t) / (2.0 * N), rtol=1e-3, atol=1e-7
    )


def test_ccr_mean_field_single_deme_k2_matches_exact():
    N = 10_000.0
    g = model_single_deme_constant(N=N)
    t = jnp.array([0.0, 1.0, 10.0])
    exact = CCRCurve(g, k=2)(t=t, num_samples={"A": (1, 1)}, params={})
    mf = CCRMeanFieldCurve(g, k=2)(t=t, num_samples={"A": (1, 1)}, params={})
    np.testing.assert_allclose(
        np.asarray(mf["c"]), np.asarray(exact["c"]), rtol=1e-6, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(mf["log_s"]), np.asarray(exact["log_s"]), rtol=5e-4, atol=1e-7
    )


@pytest.mark.slow
def test_ccr_mean_field_iwm_large_k_runs_and_invariant():
    # Large-k mean-field smoke test: should run and produce finite outputs.
    m = stdpopsim.IsolationWithMigration(
        NA=5000, N1=4000, N2=1000, T=1000, M12=0.01, M21=0.001
    )
    g = m.model.to_demes()

    k = 100
    t = jnp.array([0.0, 10.0, 100.0, 500.0, 900.0])

    # Put all red in pop1 and all blue in pop2; cross hazard at t=0 should be ~0,
    # but migration mixes colors and creates a nonzero hazard for t>0.
    mf_12 = CCRMeanFieldCurve(g, k=k)(
        t=t, num_samples={"pop1": (50, 0), "pop2": (0, 50)}
    )
    mf_21 = CCRMeanFieldCurve(g, k=k)(
        t=t, num_samples={"pop1": (0, 50), "pop2": (50, 0)}
    )

    for out in (mf_12, mf_21):
        c = np.asarray(out["c"])
        ls = np.asarray(out["log_s"])
        assert np.all(np.isfinite(c))
        assert np.all(np.isfinite(ls))
        assert np.all(c >= -1e-8)
        # survival should be non-increasing (log_s non-increasing)
        assert np.all(np.diff(ls) <= 1e-6)

    np.testing.assert_allclose(
        np.asarray(mf_12["c"]), np.asarray(mf_21["c"]), rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(mf_12["log_s"]), np.asarray(mf_21["log_s"]), rtol=1e-5, atol=1e-7
    )
    assert abs(float(mf_12["c"][0])) < 1e-6


@pytest.mark.parametrize("counts", [{"A": (2, 0)}, {"A": (0, 2)}])
def test_ccr_no_cross_when_all_one_color(counts):
    g = model_single_deme_constant(N=10_000.0)
    ccr = CCRCurve(g, k=2)
    t = jnp.array([0.0, 1.0, 10.0])
    out = ccr(t=t, num_samples=counts, params={})
    np.testing.assert_allclose(np.asarray(out["c"]), 0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(out["log_s"]), 0.0, atol=1e-6)


@pytest.mark.parametrize(
    "g,iicr_samples",
    [
        (model_single_deme_constant(), Counter({"A": 2})),
        (model_two_demes_isolated(), Counter({"A": 1, "B": 1})),
        (model_two_demes_sym_mig(), Counter({"A": 1, "B": 1})),
    ],
)
def test_ccr_equals_iicr_k2_known_cases(g, iicr_samples):
    t = jnp.array([0.0, 1.0, 10.0])
    iicr = IICRCurve(g, k=2)(t=t, num_samples=dict(iicr_samples), params={})
    ccr = CCRCurve(g, k=2)(t=t, num_samples=_as_ccr_samples(iicr_samples), params={})
    np.testing.assert_allclose(
        np.asarray(ccr["c"]), np.asarray(iicr["c"]), rtol=5e-4, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(ccr["log_s"]), np.asarray(iicr["log_s"]), rtol=5e-4, atol=2e-6
    )


def test_ccr_iwm_no_migration_zero_before_split():
    m = stdpopsim.IsolationWithMigration(
        NA=5000, N1=4000, N2=1000, T=1000, M12=0.0, M21=0.0
    )
    g = m.model.to_demes()
    t = jnp.array([0.0, 100.0, 500.0, 900.0])
    out = CCRCurve(g, k=2)(t=t, num_samples={"pop1": (1, 0), "pop2": (0, 1)}, params={})
    np.testing.assert_allclose(np.asarray(out["c"]), 0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(out["log_s"]), 0.0, atol=1e-6)


def test_ccr_iwm_matches_iicr_and_invariant_to_coloring():
    m = stdpopsim.IsolationWithMigration(
        NA=5000, N1=4000, N2=1000, T=1000, M12=0.01, M21=0.001
    )
    g = m.model.to_demes()
    t = jnp.array([0.0, 100.0, 500.0, 900.0, 1500.0])

    iicr = IICRCurve(g, k=2)(t=t, num_samples={"pop1": 1, "pop2": 1}, params={})
    ccr_12 = CCRCurve(g, k=2)(
        t=t, num_samples={"pop1": (1, 0), "pop2": (0, 1)}, params={}
    )
    ccr_21 = CCRCurve(g, k=2)(
        t=t, num_samples={"pop1": (0, 1), "pop2": (1, 0)}, params={}
    )

    np.testing.assert_allclose(
        np.asarray(ccr_12["c"]), np.asarray(ccr_21["c"]), rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(ccr_12["log_s"]), np.asarray(ccr_21["log_s"]), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(ccr_12["c"]), np.asarray(iicr["c"]), rtol=5e-4, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(ccr_12["log_s"]), np.asarray(iicr["log_s"]), rtol=5e-4, atol=1e-7
    )


def test_ccr_jit_vmap_grad_smoke():
    g = model_single_deme_constant(N=10_000.0)
    ccr = CCRCurve(g, k=2).curve(num_samples={"A": (1, 1)}, params={})

    def f_c(t):
        return ccr(t)["c"]

    def f_ls(t):
        return ccr(t)["log_s"]

    t0 = jnp.array(10.0)
    tvec = jnp.array([0.0, 1.0, 10.0, 100.0])

    np.testing.assert_allclose(
        np.asarray(jax.jit(f_c)(t0)), np.asarray(f_c(t0)), rtol=1e-6, atol=0.0
    )

    out_v = jax.vmap(f_c)(tvec)
    assert out_v.shape == tvec.shape
    assert np.all(np.isfinite(np.asarray(out_v)))

    out_jv = jax.jit(jax.vmap(f_ls))(tvec)
    assert out_jv.shape == tvec.shape
    assert np.all(np.isfinite(np.asarray(out_jv)))

    g1 = jax.grad(f_ls)(t0)
    assert np.isfinite(np.asarray(g1))

    g2 = jax.grad(lambda x: jax.vmap(f_ls)(x).sum())(tvec)
    assert g2.shape == tvec.shape
    assert np.all(np.isfinite(np.asarray(g2)))


@pytest.mark.slow
def test_ccr_iwm_torture_50_per_deme_raises_cleanly():
    m = stdpopsim.IsolationWithMigration(
        NA=5000, N1=4000, N2=1000, T=1000, M12=0.01, M21=0.001
    )
    g = m.model.to_demes()
    # 50 from each deme => k=100. The current CCR implementation enumerates the
    # full (k+1)^(2d) colored state tensor, which is intractable for this size.
    with pytest.raises(ValueError, match="CCR state space too large"):
        CCRCurve(g, k=100).curve(
            num_samples={"pop1": (25, 25), "pop2": (25, 25)},
            params={},
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "demo,pops",
    [
        (demo, Counter([pop.name for pop in pops]))
        for demo in enumerate_stdpopsim_models(max_pops=3)
        for pops in it.combinations_with_replacement(demo.populations, 2)
    ],
    ids=_idfun,
)
def test_ccr_equals_iicr_k2_stdpopsim(demo, pops):
    if "EarlyWolfAdmixture" in demo.id:
        pytest.skip("Known problematic migration/admixture model.")

    g = demo.model.to_demes()
    if len(g.migrations) > 500:
        pytest.skip("Skipping demographies with many migration rate changes.")
    if not _supported_for_ccr(g):
        pytest.skip("CCR currently only supports piecewise-constant population sizes.")

    t = _time_grid_for_pair(g, pops)
    t = jnp.asarray(t)
    iicr_out = IICRCurve(g, k=2)(t=t, num_samples=dict(pops), params={})
    ccr_out = CCRCurve(g, k=2)(t=t, num_samples=_as_ccr_samples(pops), params={})
    np.testing.assert_allclose(
        np.asarray(ccr_out["c"]), np.asarray(iicr_out["c"]), rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(ccr_out["log_s"]),
        np.asarray(iicr_out["log_s"]),
        rtol=5e-4,
        atol=5e-7,
    )
