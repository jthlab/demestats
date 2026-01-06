from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp
from penzai import pz

import demestats.events as base
from demestats.util import log_hypergeom

from .state import SetupReturn, SetupState, State, StateReturn


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
        if self.m < 0:
            raise ValueError(f"downsample size must be non-negative, got {self.m}")
        if self.m >= n:
            return child_state, {}
        if n == 0:
            raise ValueError(f"cannot downsample {self.pop} from 0 to {self.m}")
        nsp = deepcopy(child_state.ns)
        self_n = nsp[self.pop].get(self.pop, 0)
        total_n = sum(nsp[self.pop].values())
        if self.m == self_n and self_n < total_n:
            nsp[self.pop] = {self.pop: self.m}
            out_axes = OrderedDict(child_state.axes)
            assert out_axes[self.pop] == n + 1
            out_axes[self.pop] = self.m + 1
            return child_state._replace(axes=out_axes, ns=nsp), {"truncate": True}
        i, j = np.ogrid[: self.m + 1, : n + 1]
        B = np.exp(log_hypergeom(M=n, N=self.m, n=j, k=i))
        nsp[self.pop] = _rescale_ns(nsp[self.pop], self.m)
        out_axes = OrderedDict(child_state.axes)
        assert out_axes[self.pop] == n + 1
        out_axes[self.pop] = self.m + 1
        return child_state._replace(axes=out_axes, ns=nsp), {"B": B}

    def __call__(self, demo: dict, aux: dict, child_state: State) -> StateReturn:
        plp = child_state.pl
        if aux.get("truncate"):
            plp = plp.untag(self.pop)[: self.m + 1]
            plp = plp.tag(self.pop)
        elif aux:
            plp = plp.untag(self.pop)
            plp = pz.nx.nmap(jnp.array(aux["B"]).__matmul__)(plp)
            plp = plp.tag(self.pop)
        return child_state._replace(pl=plp), {}


def _rescale_ns(ns: dict[str, int], m: int) -> dict[str, int]:
    n = sum(ns.values())
    if n == 0:
        return {k: 0 for k in ns}
    if m == n:
        return dict(ns)
    if m == 0:
        return {k: 0 for k in ns}
    scale = m / n
    keys = list(ns.keys())
    raw = np.array([ns[k] * scale for k in keys], dtype=float)
    base = np.floor(raw).astype(int)
    remainder = int(m - base.sum())
    if remainder:
        frac = raw - base
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            base[idx] += 1
    return {k: int(v) for k, v in zip(keys, base)}
