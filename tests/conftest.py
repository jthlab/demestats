def enumerate_stdpopsim_models(max_pops=5):
    import stdpopsim
    models = []
    for mdl in stdpopsim.all_demographic_models():
        if mdl.num_populations > max_pops:
            continue
        models.append(mdl)
    return models
import os

import demes
import numpy as np
import stdpopsim
from pytest import fixture

from .demos import MultiAnc, SingleDeme, ThreeDemes, TwoDemes


def _idfun(x):
    if isinstance(x, stdpopsim.DemographicModel):
        return x.id
    else:
        return "".join(x)


@fixture(
    params=[
        mdl for mdl in stdpopsim.all_demographic_models() if mdl.num_populations <= 5
    ],
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
def seed(request):
    """Return a random number generator with the specified seed."""
    if os.getenv("GITHUB_ACTIONS") == "true":
        return 42  # A single, hard-coded seed for CI
    else:
        return request.param


@fixture
def rng(seed):
    return np.random.default_rng(seed)
