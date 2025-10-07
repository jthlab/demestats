import itertools as it
from collections import Counter
from functools import partial

import demes
import jax
import jax.numpy as jnp
import msprime as msp
import numpy as np
import pytest
import scipy
import stdpopsim

from demesinfer.iicr import IICRCurve

from .demos import SingleDeme


def _msp_iicr(msp_demo: msp.Demography, t: np.array, lineages: dict):
    model_times = np.array([e.time for e in msp_demo.events])
    t = np.sort(np.unique(np.concatenate([t, model_times])))
    c, p = msp_demo.debug().coalescence_rate_trajectory(steps=t, lineages=lineages)
    return dict(t=t, c=c, p=p)


def _idfun(x):
    if isinstance(x, stdpopsim.DemographicModel):
        return x.id
    else:
        return "".join(x)


@pytest.mark.parametrize(
    "demo,pops",
    [
        (demo, dict(Counter([pop.name for pop in pops])))
        for demo in stdpopsim.all_demographic_models()
        for pops in it.combinations_with_replacement(demo.populations, 2)
    ],
    ids=_idfun,
)
def test_stdpopsim(request, pytestconfig, demo, pops):
    key = request.node.name
    if "EarlyWolfAdmixture" in key:
        pytest.skip("Test fails due to high migration")

    g = demo.model.to_demes()
    start_times = [d.start_time for d in g.demes if d.name in pops]
    end_times = [d.end_time for d in g.demes if d.name in pops]
    t0 = max(end_times)
    t1 = min(start_times)
    if np.isinf(t1):
        t1 = 5 * t0
    t = np.linspace(t0, t1, 1234)
    model_times = np.array([e.time for e in demo.model.events]).clip(t0, t1)
    t = np.sort(np.unique(np.concatenate([t, model_times])))
    ii = IICRCurve(demo.model.to_demes(), 2)
    d = ii(params={}, t=jnp.array(t), num_samples=pops)
    c2 = d["c"]
    p2 = np.exp(d["log_s"])
    val = pytestconfig.cache.get(key, None)
    try:
        c1 = val[0].tolist()
        p1 = val[1].tolist()
        assert c1.shape == c2.shape
        assert p1.shape == p2.shape
    except:
        c1, p1 = demo.model.debug().coalescence_rate_trajectory(steps=t, lineages=pops)
        val = (c1.tolist(), p1.tolist())
        pytestconfig.cache.set(key, val)
    c1, p1 = map(np.array, val)
    # thle iicr can jump discontinuously at the model times, so the value depends on
    # whether the function is defined to be left- or right-continuous. afaict theres's
    # not really a convention for this so we just ignore the values at the model times
    tm = np.isclose(t[:, None], model_times[None, :]).any(1)
    np.testing.assert_allclose(c1[~tm], c2[~tm], rtol=1e-6, atol=1e-6)
    # FIXME this does not always match, appears to be due to numerical inaccuracy in msprime
    if t0 == 0.0:
        # msprime debugger does not grok the fact that a population can end. so if
        # a populations ends at a time t>0 then the survival probability is undefined
        # up to that time, whereas it computes it as 1.
        np.testing.assert_allclose(p1, p2, atol=1e-4, rtol=1e-4)


#
#
# @pytest.mark.parametrize("pops",
#     map(Counter, it.combinations_with_replacement(["ArchaicAFR", "YRI", "CHB", "CEU", "Neanderthal"], 2))
# )
# def test_ooa_const(request, pytestconfig, nd, pops):
#     demo = stdpopsim.get_species("HomSap").get_demographic_model("OutOfAfricaArchaicAdmixture_5R19")
#     g = demo.model.to_demes()
#     # modify each epoch to have constant size
#     d = g.asdict()
#     for deme in d['demes']:
#         for epoch in deme['epochs']:
#             epoch['end_size'] = epoch['start_size']
#             epoch["size_function"] = "constant"
#     d['migrations'] = []
#     g = demes.Graph.fromdict(d)
#     m3 = momi3.momi.Momi3(g)
#     t = np.linspace(0.0, 1.1e4, 123456)
#     c2, p2 = m3.coalescence_rate_trajectory(t, pops, _nd=True, _jit=False)
#     key = request.node.name
#     val = pytestconfig.cache.get(key, None)
#     if not val:
#         c1, p1 = msprime.Demography.from_demes(g).debug().coalescence_rate_trajectory(steps=t, lineages=pops)
#         val = (c1.tolist(), p1.tolist())
#         pytestconfig.cache.set(key, val)
#     c1, p1 = map(np.array, val)
#     np.testing.assert_allclose(c1, c2, rtol=1e-6, atol=1e-6)
#
#
# @pytest.mark.skip
# def test_iicr_grad_vmap():
#     demo = stdpopsim.get_species("HomSap").get_demographic_model("OutOfAfrica_3G09")
#     g = demo.model.to_demes()
#     m3 = momi3.momi.Momi3(g)
#     i = m3.iicr_nd(2)
#
#     @jax.vmap
#     @eqx.filter_grad
#     def f(pd):
#         t = jnp.linspace(1., 1e4, 16)
#         c, s = jax.vmap(i, (None, 0, None))({"YRI": 1, "CHB": 1}, t, pd)
#         return c.mean()
#
#     pd = {('migrations', 0, 'rate'): jnp.array([1e-6, 1e-5, 1e-4])}
#     res = f(pd)
#     breakpoint()
#     res = f(pd)
#
# def test_iicr_star(nd):
#     demo = msprime.Demography()
#     pops = "ABCDE"
#     for p in pops:
#         demo.add_population(name=p, initial_size=1e4)
#     demo.add_population(name="anc", initial_size=1e5)
#     demo.add_population_split(time=1e3, derived=pops, ancestral="anc")
#     m3 = momi3.momi.Momi3(demo.to_demes())
#     print(m3.iicr({"A": 2}).constraints)
#     t = np.append(np.linspace(0.0, 1.1e4, 12345), 1e3)
#     t.sort()
#     for d in map(Counter, it.combinations_with_replacement(pops, 2)):
#         c, p = m3.coalescence_rate_trajectory(t, d, _nd=nd)
#         if len(d) == 1:
#             np.testing.assert_allclose(c[t < 1e3], 1 / 2 / 1e4)
#         else:
#             np.testing.assert_allclose(c[t < 1e3], 0.0)
#         np.testing.assert_allclose(c[t >= 1e3], 1 / 2 / 1e5)
#
#     # test some other one-off cases
#     d = {"A": 2, "B": 1}
#     c, p = m3.coalescence_rate_trajectory(t, d, _nd=nd)
#     np.testing.assert_allclose(c[t < 1e3], 1 / 2 / 1e4)
#     np.testing.assert_allclose(c[t >= 1e3], 3 / 2 / 1e5)
#
#     d = {"A": 2, "B": 2}
#     c, p = m3.coalescence_rate_trajectory(t, d, _nd=nd)
#     np.testing.assert_allclose(c[t < 1e3], 2 / 2 / 1e4)
#     np.testing.assert_allclose(c[t >= 1e3], 6 / 2 / 1e5)
#
#     d = {"A": 2, "B": 2, "C": 1}
#     c, p = m3.coalescence_rate_trajectory(t, d)
#     np.testing.assert_allclose(c[t < 1e3], 2 / 2 / 1e4)
#     np.testing.assert_allclose(c[t >= 1e3], 10 / 2 / 1e5)


@pytest.mark.parametrize("k", [2, 5, 10])
def test_iicr_simple(k):
    N = 1e4
    demo = stdpopsim.PiecewiseConstantSize(N0=N).model.to_demes()
    # FIXME: rate at changepoints is not handled correctly because of autodiff
    t = np.linspace(0.0, 1.1e4, 10)
    ii = IICRCurve(demo, k)
    d = ii(params={}, t=jnp.array(t), num_samples={"pop_0": k})
    np.testing.assert_allclose(d["log_s"], -t * k * (k - 1) / 4 / N)
    np.testing.assert_allclose(d["c"], k * (k - 1) / 4 / N)


@pytest.mark.parametrize("k", [2, 5, 20])
def test_iicr_growth(k):
    N = 1e4
    b = demes.Builder()
    t = jnp.linspace(0, 2.1e4, 123)
    b.add_deme(
        name="A",
        epochs=[
            dict(start_size=N, end_size=N, end_time=1000),
            dict(start_size=N, end_size=N * np.exp(0.01 * 1000), end_time=0),
        ],
    )
    demo = b.resolve()
    N_t = np.array([demo.demes[0].size_at(tt) for tt in t])
    ii = IICRCurve(demo, k)
    d = ii(params={}, t=jnp.array(t), num_samples={"A": k})
    np.testing.assert_allclose(d["c"], k * (k - 1) / 4 / N_t, atol=1e-6, rtol=1e-6)


def test_iicr_grad():
    N = 1e4
    demo = stdpopsim.IsolationWithMigration(
        NA=5000, N1=4000, N2=1000, T=1000, M12=0.01, M21=0.001
    )
    g = demo.model.to_demes()
    t = np.linspace(0.0, 1.1e4, 123)
    k = 2
    ii = IICRCurve(g, k)
    lineages = {"pop1": 1, "pop2": 1}
    params = {("migrations", 0, "rate"): 0.01, ("migrations", 1, "rate"): 0.001}

    @jax.jit
    @jax.grad
    def f(params, t):
        return ii(params=params, t=jnp.atleast_1d(t), num_samples=lineages)["c"]

    dp = f(params, 100.0)


@pytest.mark.parametrize("demes", it.combinations_with_replacement(["pop1", "pop2"], 2))
def test_iicr_iwm(iwm, demes):
    g = iwm.model.to_demes()
    t = np.linspace(0.0, 1.1e4, 123)
    k = 2
    ii = IICRCurve(g, k)
    lineages = dict(Counter(demes))
    d1 = _msp_iicr(iwm.model, t, lineages)
    d2 = ii(params={}, t=d1["t"], num_samples=lineages)
    np.testing.assert_allclose(d1["c"], d2["c"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(d1["log_s"], d2["log_s"], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("pops", it.combinations_with_replacement("AB", 2))
def test_iicr_pulse(pops):
    N = 1e4
    b = demes.Builder()
    t = jnp.linspace(0, 2.1e4, 123)
    b.add_deme(
        name="anc",
        epochs=[
            dict(start_size=N, end_size=N, end_time=1000),
        ],
    )
    b.add_deme(
        name="A",
        ancestors=["anc"],
        start_time=1000,
        epochs=[
            dict(start_size=N, end_size=N, end_time=0),
        ],
    )
    b.add_deme(
        name="B",
        ancestors=["anc"],
        start_time=1000,
        epochs=[
            dict(start_size=N / 2, end_size=N, end_time=0),
        ],
    )
    b.add_pulse(sources=["A"], dest="B", time=250, proportions=[0.5])
    demo = b.resolve()
    t = np.append(np.linspace(0.0, 1.1e4, 123), 250.0)
    t.sort()
    k = 2
    ii = IICRCurve(demo, k)
    lineages = dict(Counter(pops))
    d1 = _msp_iicr(msp.Demography.from_demes(demo), t, lineages)
    d2 = ii(params={}, t=d1["t"], num_samples=lineages)
    np.testing.assert_allclose(d1["c"], d2["c"], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "pops", it.combinations_with_replacement(["A", "B", "admix"], 2)
)
def test_iicr_multiple_ancestry(rng, pops):
    N = 1e4
    b = demes.Builder()
    t = jnp.linspace(0, 2.1e4, 123)
    b.add_deme(
        name="anc",
        epochs=[
            dict(start_size=N, end_size=N, end_time=1000),
        ],
    )
    b.add_deme(
        name="A",
        ancestors=["anc"],
        start_time=1000,
        epochs=[
            dict(start_size=N, end_size=N, end_time=0),
        ],
    )
    b.add_deme(
        name="B",
        ancestors=["anc"],
        start_time=1000,
        epochs=[
            dict(start_size=N / 2, end_size=N, end_time=0),
        ],
    )
    alpha = list(rng.dirichlet([1.0, 1.0]))
    b.add_deme(
        name="admix",
        ancestors=["A", "B"],
        proportions=alpha,
        start_time=500,
        epochs=[
            dict(start_size=N / 2, end_size=N, end_time=0),
        ],
    )
    beta = rng.uniform(0.1, 0.5)
    b.add_pulse(sources=["A"], dest="B", time=750, proportions=[beta])
    demo = b.resolve()
    t = np.append(np.linspace(0.0, 1.1e4, 123), 250.0)
    t.sort()
    k = 2
    ii = IICRCurve(demo, k)
    lineages = dict(Counter(pops))
    d1 = _msp_iicr(msp.Demography.from_demes(demo), t, lineages)
    d2 = ii(params={}, t=d1["t"], num_samples=lineages)
    np.testing.assert_allclose(d1["c"], d2["c"], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "pops", it.combinations_with_replacement(["A", "B", "admix"], 2)
)
def test_iicr_mig0_vs_msp(pops, rng):
    N = 1e4
    b = demes.Builder()
    b.add_deme(
        name="anc",
        epochs=[
            dict(start_size=N, end_size=N, end_time=800),
        ],
    )
    b.add_deme(
        name="A",
        ancestors=["anc"],
        start_time=900,
        epochs=[
            dict(start_size=N, end_size=N * 2, end_time=0),
        ],
    )
    b.add_deme(
        name="B",
        ancestors=["anc"],
        start_time=1000,
        epochs=[
            dict(start_size=N / 2, end_size=N, end_time=0),
        ],
    )
    alpha = list(rng.dirichlet([1.0, 1.0]))
    b.add_deme(
        name="admix",
        ancestors=["A", "B"],
        proportions=alpha,
        start_time=500,
        epochs=[
            dict(start_size=N / 2, end_size=N, end_time=0),
        ],
    )
    b.add_migration(source="A", dest="B", start_time=1e2, end_time=0.0, rate=0.0)
    beta = rng.uniform(0.1, 0.5)
    b.add_pulse(sources=["A"], dest="B", time=50, proportions=[beta])
    demo = b.resolve()
    lineages = dict(Counter(pops))
    ii = IICRCurve(demo, 2)
    d = ii(params={}, t=jnp.linspace(0.0, 1.1e4, 123), num_samples=lineages)


@pytest.mark.parametrize("n", [2, 5, 10])
def test_larger_n(n):
    demo, _ = SingleDeme.Constant(size=1e4).base()
    t = np.linspace(0.0, 1.1e4, 123456)
    ii = IICRCurve(demo, n)
    d = ii(t, {"A": n})
    c = d["c"]
    p = np.exp(d["log_s"])
    nC2 = n * (n - 1) / 2
    np.testing.assert_allclose(c, nC2 / 2 / 1e4)
    np.testing.assert_allclose(p, np.exp(-t * nC2 / 2 / 1e4))


def test_iicr_call_vmap():
    demo = msp.Demography()
    pops = [demo.add_population(initial_size=5000, name=f"P{i}") for i in range(5)]
    demo.add_population(initial_size=5000, name="anc")
    for i in range(4):
        demo.set_symmetric_migration_rate(populations=(f"P{i}", f"P{i + 1}"), rate=1e-4)
    demo.add_population_split(
        time=1000, derived=[f"P{i}" for i in range(5)], ancestral="anc"
    )

    paths = {
        ("migrations", 0, "rate"): 0.0001,
    }
    t_breaks = jax.numpy.geomspace(1e-4, 1e5, 2000)
    deme_names = {
        "P2": "value2",
        "P0": "value0",
        "P3": "value3",
        "P4": "value4",
        "P1": "value1",
    }.keys()
    unique_cfg = np.array(
        [
            [0, 0, 0, 0, 2],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 2, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 2, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 2, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [2, 0, 0, 0, 0],
        ],
        np.int32,
    )
    iicr = IICRCurve(demo=demo.to_demes(), k=2)
    iicr_call = jax.jit(iicr.__call__)
    with jax.debug_nans(True):
        c_map = jax.vmap(
            lambda cfg: iicr_call(
                params=paths, t=t_breaks, num_samples=dict(zip(deme_names, cfg))
            )["c"]
        )(jnp.array(unique_cfg))
    np.testing.assert_equal(np.isnan(c_map).any(), False)


@pytest.mark.parametrize("demes", it.combinations_with_replacement(["pop1", "pop2"], 2))
def test_curve(iwm, demes):
    g = iwm.model.to_demes()
    t = np.linspace(0.0, 1.1e4, 123)
    k = 2
    ii = IICRCurve(g, k)
    num_samples = dict(Counter(demes))
    c = ii.curve(num_samples)
    s1 = set(c.jump_ts.tolist())
    s2 = set([0.0, g.demes[0].epochs[0].end_time, np.inf])
    assert s1 == s2


@pytest.mark.parametrize("demes", it.combinations_with_replacement(["pop1", "pop2"], 2))
def test_integral_identity(iwm, demes):
    g = iwm.model.to_demes()
    t = np.linspace(0.0, 1.1e4, 123)
    k = 2
    ii = IICRCurve(g, k)
    num_samples = dict(Counter(demes))
    c = ii.curve(num_samples)

    @jax.jit
    def f(t):
        return c(t)["c"]

    for t in np.geomspace(1, 1e4, 10):
        Lambda, err = scipy.integrate.quad(f, 0, t, points=c.jump_ts)
        # tolerances are a bit loose due to numerical integration error
        np.testing.assert_allclose(-Lambda, c(t)["log_s"], atol=1e-4, rtol=1e-4)
