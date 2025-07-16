import demes
import numpy as np
import stdpopsim
from pytest import fixture


def _idfun(x):
    if isinstance(x, stdpopsim.DemographicModel):
        return x.id
    else:
        return "".join(x)


@fixture(
    params=list(stdpopsim.all_demographic_models()),
    ids=_idfun,
)
def sp_demo(request):
    return request.param


@fixture
def demo(sp_demo) -> demes.Graph:
    return sp_demo.model.to_demes()


@fixture
def iwm():
    return stdpopsim.IsolationWithMigration(
        NA=5000, N1=4000, N2=1000, T=1000, M12=0.01, M21=0.001
    )


@fixture(params=range(5))
def rng(request):
    """Return a random number generator with the specified seed."""
    seed = request.param
    return np.random.default_rng(seed)
