from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp
from penzai import pz

import demesinfer.events as base
from demesinfer.util import log_hypergeom

from .state import *


@dataclass(kw_only=True)
class Upsample(base.Event):
    "upsample to at least m lineages (backwards in time)"

    pop: str
    m: int

    def setup(self, demo: dict, aux: dict, child_state: SetupState) -> SetupReturn:
        ns = child_state.ns
        n = child_state.axes[self.pop] - 1
        if self.m <= n:
            # no upsampling needed
            return child_state, {}
        i, j = np.ogrid[: self.m + 1, : n + 1]
        B = np.exp(log_hypergeom(M=self.m, N=n, n=i, k=j))

        # is this neccesary?
        assert ns[self.pop] == {self.pop: n}
        nsp = dict(child_state.ns)
        nsp[self.pop] = {self.pop: self.m}

        out_axes = OrderedDict(child_state.axes)
        assert out_axes[self.pop] == n + 1
        out_axes[self.pop] = self.m + 1
        return child_state._replace(axes=out_axes, ns=nsp), {"B": B}

    def __call__(self, demo: dict, aux: dict, child_state: State) -> StateReturn:
        plp = child_state.pl
        if aux:  # if aux is empty, n > m, no downsampling was needed
            plp = plp.untag(self.pop)
            plp = pz.nx.nmap(jnp.array(aux["B"]).__matmul__)(plp)
            plp = plp.tag(self.pop)
        return child_state._replace(pl=plp), {}


@dataclass(kw_only=True)
class Downsample(base.Event):
    "Downsample to m lineages (backwards in time)"

    pop: str
    m: int

    def setup(self, demo: dict, aux: dict, child_state: SetupState) -> SetupReturn:
        n = child_state.axes[self.pop] - 1
        assert self.m <= n
        if self.m == n:
            return child_state, {}
        i, j = np.ogrid[: n + 1, : self.m + 1]
        B = np.exp(log_hypergeom(M=n, N=self.m, n=i, k=j))
        Bplus = np.linalg.pinv(B, rcond=1e-5)
        nsp = dict(ns)
        nsp[self.pop] = {self.pop: self.m}
        out_axes = OrderedDict(in_axes)
        assert out_axes[self.pop] == n + 1
        out_axes[self.pop] = self.m + 1
        return child_state._replace(axes=out_axes, ns=nsp), {"B": B, "Bplus": Bplus}

    def execute(self, st: State, params: dict, aux: dict) -> StateReturn:
        if aux is None:
            # no bounding was possible/necessary, so setup set aux to None.
            return st
        ((i, (B, Bplus)),) = aux.items()
        d = st.pl.ndim
        pl_inds = list(range(d))
        out_inds = list(pl_inds)
        assert d not in out_inds
        out_inds[i] = d
        plp = oe_einsum(st.pl, tuple(pl_inds), Bplus, (d, i), tuple(out_inds))
        return st._replace(pl=plp)
