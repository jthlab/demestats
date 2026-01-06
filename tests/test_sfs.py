from collections import OrderedDict

import demes
import jax
import jax.numpy as jnp
import moments
import momi as momi2
import msprime as msp
import numpy as np
import pytest
import stdpopsim
from penzai import pz
from pytest import fixture

from demestats.sfs import ExpectedSFS
from demestats.sfs.events.sample import Downsample
from demestats.sfs.events.state import SetupState, State

from .demos import MultiAnc, SingleDeme, ThreeDemes, TwoDemes


@fixture(
    params=[
        "constant_1d",
        "exponential_1d",
        "constant_2d",
        "exponential_2d",
        "exponential_2d_pulse",
        "exponential_2d_two_pulse",
        "exponential_2d_migration",
        "constant_2d_migration_two_phase",
        "constant_2d_migration_sym_pulse",
        "exponential_3d_migrations",
        "multianc",
    ],
)
def toy_demo_name(request):
    return request.param


@fixture
def toy_demes(toy_demo_name):
    if toy_demo_name == "multianc":
        return list("ABC")
    else:
        # extract the integer part of the name
        n = int(next(x for x in toy_demo_name if x.isdigit()))
        assert 1 <= n <= 3
        return list("ABC")[:n]


@fixture
def toy_samples(rng, toy_demes):
    return dict(zip(toy_demes, rng.integers(1, 20, size=len(toy_demes))))


def _demo_for(name):
    match name:
        case "constant_1d":
            return SingleDeme.Constant().base()
        case "exponential_1d":
            return SingleDeme.Exponential().base()
        case "constant_2d":
            return TwoDemes.Constant().base()
        case "exponential_2d":
            return TwoDemes.Exponential().base()
        case "exponential_2d_pulse":
            return TwoDemes.Exponential().pulse()
        case "exponential_2d_two_pulse":
            return TwoDemes.Exponential().two_pulses()
        case "exponential_2d_migration":
            return TwoDemes.Exponential().migration()
        case "constant_2d_migration_two_phase":
            return TwoDemes.Constant().migration_twophase()
        case "constant_2d_migration_sym_pulse":
            return TwoDemes.Constant().migration_sym_pulse()
        case "exponential_3d_migrations":
            return ThreeDemes.Exponential().migrations()
        case "multianc":
            return MultiAnc().base()
    raise ValueError(f"Unknown demo name: {name}")


def moments_sfs(g: demes.Graph, num_samples: int):
    esfs = moments.Spectrum.from_demes(
        g,
        sampled_demes=list(num_samples.keys()),
        sample_sizes=list(num_samples.values()),
    )
    esfs = np.array(esfs)
    return esfs * 4 * g.demes[0].epochs[0].start_size


def momi2_sfs(momi2_model, sample_sizes):
    assert set(momi2_model.leafs) == sample_sizes.keys()
    P = len(sample_sizes)
    momi2sfs_array = []
    Mutant_sizes = np.indices([n + 1 for n in sample_sizes.values()])
    Mutant_sizes = np.moveaxis(Mutant_sizes, 0, -1).reshape(-1, P)[1:-1]
    for bs in Mutant_sizes:
        x = []
        for i in range(P):
            n = list(sample_sizes.values())[i]
            b = bs[i]
            x.append([int(n - b), int(b)])
        momi2sfs_array.append([x])
    s = momi2.site_freq_spectrum(momi2_model.leafs, momi2sfs_array)
    momi2_model.set_data(s, length=1)
    momi2_model.set_mut_rate(1.0)
    ret = momi2_model.expected_sfs()
    esfs = np.zeros([n + 1 for n in sample_sizes.values()])
    for key in ret:
        ind = [i[1] for i in key]
        esfs[tuple(ind)] = ret[key]
    return esfs


def assert_close_sfs(s1, s2, atol=1e-5):
    s1, s2 = [s.flatten()[1:-1] for s in (s1, s2)]
    e1 = s1 / s1.sum()
    e2 = s2 / s2.sum()
    # np.testing.assert_allclose(
    #     s1 / s1.sum(),
    #     s2 / s2.sum(),
    #     atol=atol
    # )
    # check that the two distributions are close in TV distance
    tv_distance = np.sum(np.abs(e1 - e2)) / 2.0
    assert tv_distance < atol


def test_missing_demes():
    m = stdpopsim.IsolationWithMigration(
        NA=5000, N1=4000, N2=1000, T=1000, M12=0.01, M21=0.003
    )
    g = m.model.to_demes()
    with pytest.raises(ValueError):
        ExpectedSFS(g, num_samples={"X": 10})


def test_sfs_basic():
    b = demes.Builder()
    b.add_deme("A", epochs=[{"start_size": 10, "end_size": 10, "end_time": 0}])
    g = b.resolve()
    esfs = ExpectedSFS(g, num_samples={"A": 10})
    e1 = esfs()[1:-1]
    e2 = 1 / np.arange(1, 10)
    np.testing.assert_allclose(e1 / e1.sum(), e2 / e2.sum(), rtol=1e-5)


def test_prune_leaf_noop():
    g, _ = SingleDeme.Constant().base()
    n = 6
    base = ExpectedSFS(g, num_samples={"A": n})()
    pruned = ExpectedSFS(g, num_samples={"A": n}, prune={"A": n})()
    np.testing.assert_allclose(base, pruned)


def test_prune_leaf_changes_output():
    g, _ = SingleDeme.Constant().base()
    n = 6
    m = 3
    base = ExpectedSFS(g, num_samples={"A": n})()
    pruned = ExpectedSFS(g, num_samples={"A": n}, prune={"A": m})()
    assert base.shape == pruned.shape
    assert not np.allclose(base, pruned)


def test_prune_path_ambiguous_raises():
    g, _ = SingleDeme.Constant().base()
    path = ("demes", 0, "epochs", 0, "end_time")
    with pytest.raises(ValueError):
        ExpectedSFS(g, num_samples={"A": 6}, prune=[("A", 3, path)])


def test_prune_zero_samples_zero_sfs():
    g, _ = SingleDeme.Constant().base()
    pruned = ExpectedSFS(g, num_samples={"A": 4}, prune={"A": 0})()
    assert np.allclose(pruned, 0.0)


def test_downsample_event_hypergeom():
    n = 4
    m = 2
    ev = Downsample(pop="A", m=m)
    setup = SetupState(
        migrations=frozenset(),
        axes=OrderedDict({"A": n + 1}),
        ns={"A": {"A": n}},
    )
    _, aux = ev.setup(demo={}, aux={}, child_state=setup)
    vec = np.arange(n + 1, dtype=float)
    st = State(pl=pz.nx.wrap(vec, "A"), phi=0.0, l0=vec[0])
    out, _ = ev(demo={}, aux=aux, child_state=st)
    expected = aux["B"] @ vec
    np.testing.assert_allclose(out.pl.unwrap("A"), expected)


def test_prune_deep_ancestry_no_effect():
    b = demes.Builder()
    b.add_deme("ANC", epochs=[{"end_time": 50.0, "start_size": 1.0}])
    b.add_deme("A", ancestors=["ANC"], epochs=[{"start_size": 1.0}])
    g = b.resolve()
    ns = {"A": 4}
    path = ("demes", 1, "start_time")
    base = ExpectedSFS(g, num_samples=ns)()
    pruned = ExpectedSFS(g, num_samples=ns, prune=[("A", 1, path)])()
    np.testing.assert_allclose(base, pruned)


def test_prune_path_requires_pop():
    g, _ = TwoDemes.Constant().base()
    demo = g.asdict()
    b_idx = next(i for i, d in enumerate(demo["demes"]) if d["name"] == "B")
    path = ("demes", b_idx, "epochs", 0, "end_time")
    with pytest.raises(ValueError):
        ExpectedSFS(g, num_samples={"A": 5, "B": 5}, prune=[("A", 3, path)])


def test_sfs_iwm():
    m = stdpopsim.IsolationWithMigration(
        NA=5000, N1=4000, N2=1000, T=1000, M12=0.01, M21=0.021
    )
    g = m.model.to_demes()
    ns = {"pop1": 10, "pop2": 10}
    esfs = ExpectedSFS(g, num_samples=ns)
    e2 = moments_sfs(g, ns)
    e1 = esfs()
    assert_close_sfs(e1, e2, atol=1e-4)


@pytest.mark.parametrize("vs", ["momi2", "moments"])
def test_toy_demos(toy_demo_name, toy_samples, vs):
    g, momi2_model = _demo_for(toy_demo_name)
    if vs == "momi2" and momi2_model is None:
        pytest.skip("momi2 model not available for this demo")
    ns = toy_samples
    esfs = ExpectedSFS(g, num_samples=ns)
    e1 = esfs()
    if vs == "moments":
        try:
            e2 = moments_sfs(g, ns)
        except Exception:
            pytest.skip("moments SFS calculation failed")
    elif vs == "momi2":
        e2 = momi2_sfs(momi2_model, ns)
    assert_close_sfs(e1, e2)


def test_stdpopsim(sp_demo, rng, monkeypatch):
    g = sp_demo.model.to_demes()
    # randomly sample to max 5 demes
    demes = rng.choice(g.demes, size=min(5, len(g.demes)), replace=False)
    ns = {deme.name: rng.integers(1, 5) for deme in demes}
    esfs = ExpectedSFS(g, num_samples=ns)
    e1 = esfs()
    e2 = moments_sfs(g, ns)
    assert_close_sfs(e1, e2, atol=1e-4)


def test_tensor_prod(rng):
    m = stdpopsim.IsolationWithMigration(
        NA=5000, N1=4000, N2=1000, T=1000, M12=0.01, M21=0.0
    )
    g = m.model.to_demes()
    ns = {"pop1": 10, "pop2": 11}
    proj = {"pop1": rng.normal(size=(1, 11)), "pop2": rng.normal(size=(1, 12))}
    m = ExpectedSFS(g, num_samples=ns)
    p1 = m.tensor_prod(proj)
    esfs = m()
    esfs = np.array(esfs)
    esfs[0, 0] = esfs[-1, -1] = 0.0
    p2 = np.einsum("i,j,ij->", proj["pop1"][0], proj["pop2"][0], esfs)
    np.testing.assert_allclose(p1, p2, rtol=1e-5)


def test_equal_migration_twophase():
    g1, _ = TwoDemes.Constant().migration(tstart=1.0, tend=0.0, rate=0.05)
    g2, _ = TwoDemes.Constant().migration_twophase(
        rate1=0.05, rate2=0.05, tstart=1.0, tend=0.0
    )
    ns = {"A": 10, "B": 13}
    e1, e2 = [ExpectedSFS(g, num_samples=ns)() for g in (g1, g2)]
    np.testing.assert_allclose(e1, e2)


def test_equal_empty_pulse():
    g1, _ = TwoDemes.Constant().migration_sym_pulse(p=0.0, tstart=1.0, tend=0.0)
    g2, _ = TwoDemes.Constant().migration_sym(tstart=1.0, tend=0.0)
    ns = {"A": 10, "B": 13}
    e1, e2 = [ExpectedSFS(g, num_samples=ns)() for g in (g1, g2)]
    assert_close_sfs(e1, e2)


def test_small_migration():
    g, _ = TwoDemes.Exponential().migration(tstart=1.0, tend=0.0, rate=0.05)
    ns = {"A": 2, "B": 3}
    esfs = ExpectedSFS(g, num_samples=ns)
    e1 = esfs()
    e2 = moments_sfs(g, ns)
    assert_close_sfs(e1, e2)


def test_rescaling():
    g1, _ = TwoDemes.Exponential(t=1.0, size=1.0, g=1.0).migration(
        tstart=1.0, tend=0.5, rate=0.05
    )
    g2, _ = TwoDemes.Exponential(t=1e3, size=1e3, g=1e-3).migration(
        tstart=1.0 * 1e3, tend=0.5 * 1e3, rate=0.05 / 1e3
    )
    ns = {"A": 7, "B": 10}
    esfs1 = ExpectedSFS(g1, num_samples=ns)
    esfs2 = ExpectedSFS(g2, num_samples=ns)
    e1 = esfs1()
    e2 = esfs2()
    np.testing.assert_allclose(e1 * 1e3, e2, rtol=1e-5)
    assert_close_sfs(e1, e2)


def test_bind():
    g, _ = SingleDeme.Constant().base()
    esfs = ExpectedSFS(g, num_samples={"A": 10})
    params = {
        ("demes", 0, "epochs", 0, "start_size"): 1.0,
        ("demes", 0, "epochs", 0, "end_size"): 2.0,
    }
    with pytest.raises(ValueError):
        esfs(params=params)
    esfs(params=params)


def test_random_projection_normal():
    demo = msp.Demography()
    demo.add_population(initial_size=5000, name="anc")
    [demo.add_population(initial_size=5000, name=f"P{i}") for i in range(2)]
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
    demo.add_population_split(
        time=1000, derived=[f"P{i}" for i in range(2)], ancestral="anc"
    )
    afs_samples = {f"P{i}": 10 * 2 for i in range(2)}
    esfs = ExpectedSFS(demo.to_demes(), num_samples=afs_samples)
    rng = np.random.default_rng(5)
    params = {
        frozenset(
            {
                ("demes", 0, "epochs", 0, "end_size"),
                ("demes", 0, "epochs", 0, "start_size"),
            }
        ): 1000.0,
    }
    e1 = esfs(params)
    e1 = e1.at[0, 0].set(0.0)
    e1 = e1.at[-1, -1].set(0.0)
    tensor = []
    truth = []
    tp = jax.jit(lambda X: esfs.tensor_prod(X, params))
    for _ in range(5):
        proj = {"P0": rng.normal(size=(1, 21)), "P1": rng.normal(size=(1, 21))}
        tensor.append(tp(proj))
        truth.append(jnp.einsum("i,j,ij->", proj["P0"][0], proj["P1"][0], e1))
    np.testing.assert_allclose(np.squeeze(truth), np.squeeze(tensor), rtol=1e-4)


def test_2ndary_contact():
    demo = demes.Builder()
    demo.add_deme(name="mainland", epochs=[dict(start_size=1000, end_time=0)])
    demo.add_deme(
        name="island",
        ancestors=["mainland"],
        start_time=500,
        epochs=[dict(start_size=500, end_time=0)],
    )
    demo.add_migration(
        source="mainland", dest="island", rate=1e-4, start_time=300, end_time=0
    )
    demo.add_migration(
        source="island",
        dest="mainland",
        rate=1e-4,  # same or different rate
        start_time=300,
        end_time=0,
    )
    g = demo.resolve()
    ns = {"mainland": 10, "island": 10}
    esfs = ExpectedSFS(g, num_samples=ns)
    e1 = esfs()
    e2 = moments_sfs(g, ns)
    assert_close_sfs(e1, e2, atol=1e-4)
