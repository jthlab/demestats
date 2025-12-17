import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Int, ScalarLike
from penzai import pz

from ..iicr.state import SetupState, State


class StateCcr(State):
    # p is an (k+1,)*2*d array whose axes are (deme, "red") and (deme, "blue").
    p: pz.nx.NamedArray

    @property
    def axes(self) -> tuple[tuple[str, str], ...]:
        axes = tuple(self.p.named_axes.keys())
        if not all(isinstance(a, tuple) and len(a) == 2 for a in axes):
            raise ValueError(
                "CCR state axes must be 2-tuples (deme, color); got "
                f"{self.p.named_axes.keys()}"
            )
        return axes

    @property
    def pops(self) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for pop, _color in self.axes:
            if pop not in seen:
                seen.add(pop)
                out.append(pop)
        return tuple(out)

    @property
    def n(self) -> int:
        ret = next(iter(self.p.named_axes.values()))
        assert all(v == ret for v in self.p.named_axes.values()), (
            f"Expected all axes to have the same size, got {self.p.named_axes}"
        )
        return ret - 1

    @property
    def d(self) -> int:
        return len(self.pops)

    def coal_rate(self, t, demo):
        # Cross-coalescence hazard for each microstate: sum_d (n_red[d] * n_blue[d]) / (2 Ne_d(t)).
        import demesinfer.util as util

        etas = util.coalescent_rates(demo)
        k = self.n
        ndim = len(self.axes)

        rate = jnp.zeros((k + 1,) * ndim)
        for deme_idx, pop in enumerate(self.pops):
            red_axis = 2 * deme_idx
            blue_axis = red_axis + 1
            red_idx = jnp.arange(k + 1).reshape(
                (1,) * red_axis + (k + 1,) + (1,) * (ndim - red_axis - 1)
            )
            blue_idx = jnp.arange(k + 1).reshape(
                (1,) * blue_axis + (k + 1,) + (1,) * (ndim - blue_axis - 1)
            )
            rate = rate + (red_idx * blue_idx) * (1.0 / (2.0 * etas[pop](t)))
        return pz.nx.wrap(rate, *self.axes)

    @classmethod
    def setup(
        cls,
        num_samples: dict[str, tuple[Int[ScalarLike, ""], Int[ScalarLike, ""]]],
        k: int,
    ) -> dict[str, "StateCcr"]:
        ret: dict[str, StateCcr] = {}
        for pop, (nr, nb) in num_samples.items():
            nr = int(nr)
            nb = int(nb)
            if nr < 0 or nb < 0 or nr + nb > k:
                raise ValueError(
                    f"Invalid CCR sample counts for deme {pop}: (red={nr}, blue={nb}) "
                    f"with k={k}"
                )
            arr = jnp.zeros((k + 1, k + 1))
            arr = arr.at[nr, nb].set(1.0)
            p = pz.nx.wrap(arr, (pop, "red"), (pop, "blue"))
            ret[pop] = cls(p=p, log_s=jnp.array(0.0))
        return ret


# Backwards-compatible alias used by other CCR modules.
State = StateCcr
