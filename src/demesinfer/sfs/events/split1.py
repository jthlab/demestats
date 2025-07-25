"""Merge two populations in the same event block"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from penzai import pz

import demesinfer.events as base
from demesinfer.util import log_hypergeom

from .state import *


@dataclass(kw_only=True)
class Split1(base.Split1):
    """Merge two populations in the same event block.

    Attributes:
        donor: first population which gets merged into recipient (backwards in time)
        recipient: population which receives lineages from donor

    Notes:
        After the merge, the donor population is removed from the partial likelihood.
    """

    def __post_init__(self):
        assert self.donor != self.recipient

    def setup(self, demo: dict, aux: dict, child_state: SetupState) -> SetupReturn:
        """Setup for the split1 lemma.

        Args:
            in_axes: axes of the child partial likelihood.
            ns: number of samples subtended by each population.

        Returns:
            axes: dimensions of the pl after the split event
            aux: auxiliary data

        Notes:
            pop2 is merged into pop1. The merged population resides in pop1.
        """
        nw1 = child_state.axes[self.donor] - 1
        nw2 = child_state.axes[self.recipient] - 1
        i, j, k = np.ogrid[: nw1 + nw2 + 1, : nw1 + 1, : nw2 + 1]
        # update population counts
        nsp = deepcopy(child_state.ns)
        nsp[self.recipient] |= nsp[self.donor]
        del nsp[self.donor]
        n = sum(nsp[self.recipient].values())
        # resulting axes, same thing
        out_axes = OrderedDict(child_state.axes)
        del out_axes[self.donor]
        # deleting and then re-adding ensures that recipient is the last axis. this enables us to easily apply
        # the hypergeom inverse in execute() by treating the leading axes as batch dimensions.
        del out_axes[self.recipient]
        out_axes[self.recipient] = n + 1
        # compute aux
        H = np.exp(log_hypergeom(M=nw1 + nw2, N=i, n=nw1, k=j)) * (
            i == j + k
        )  # [nw1+nw2+1, nw1+1, nw2+1]
        aux = {"H": H}
        # downsample if necessary
        assert n <= nw1 + nw2
        if n < nw1 + nw2:
            B = np.exp(
                log_hypergeom(
                    M=nw1 + nw2, N=i[..., 0], n=n, k=jnp.arange(n + 1)[None, :]
                )
            )  # [nw1+nw2+1, n + 1]
            aux["B"] = B
            aux["Bplus"] = np.linalg.pinv(B)
            Q, R = np.linalg.qr(B, mode="reduced")
            aux["Q"] = Q
            aux["R"] = R

        return child_state._replace(axes=out_axes, ns=nsp), aux

    def __call__(self, demo: dict, aux: dict, child_state: State) -> StateReturn:
        """Merge pop2 into pop1 when they are both in the same event block."""
        if "Bplus" in aux:
            # B = QR so B+X = solve(R, Q.T B)
            # hypergeometrically upsample (forwards in time) to go from n to nw1 + nw2
            # Bplus_inds = [a, b]
            # H_inds = [b, self.donor, self.recipient]
            # plp = oe_einsum(
            #     aux["Bplus"], Bplus_inds, aux["H"], H_inds, st.pl, pl_inds, out_inds
            # )
            def f(pl):
                new_recip = 0
                temp = 1
                donor = 2
                recip = 3
                plp = jnp.einsum(
                    aux["Q"].T,
                    [new_recip, temp],
                    aux["H"],
                    [temp, donor, recip],
                    pl,
                    [donor, recip],
                    [0],
                )
                return jax.scipy.linalg.solve_triangular(aux["R"], plp)
        else:

            def f(pl):
                new_recip = 0
                donor = 1
                recip = 2
                return jnp.einsum(
                    aux["H"], [new_recip, donor, recip], pl, [donor, recip], [new_recip]
                )

        pl = child_state.pl.untag(self.donor, self.recipient)
        plp = pz.nx.nmap(f)(pl).tag(self.recipient)
        return child_state._replace(pl=plp), {}
