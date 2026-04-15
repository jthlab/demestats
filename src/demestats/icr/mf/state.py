import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar, ScalarLike

from ..state import State


class StateMf(State):
    pops: tuple[str, ...] = eqx.field(static=True)
    group_sizes: Float[Array, "g"]
    u: Float[Array, "g d"]

    @property
    def d(self) -> int:
        return len(self.pops)

    @property
    def g(self) -> int:
        return int(self.group_sizes.shape[0])

    @classmethod
    def setup(
        cls, num_samples: dict[str, Int[ScalarLike, ""]], k: int
    ) -> dict[str, "StateMf"]:
        # `k` is unused in the mean-field representation, but kept for signature
        # compatibility with the exact ICR states.
        del k
        ret: dict[str, StateMf] = {}
        for pop, n in num_samples.items():
            try:
                n_int = int(n)
            except Exception:
                n_int = None

            if n_int is not None:
                if n_int < 0:
                    raise ValueError(f"Invalid sample count for {pop}: {n_int}")
                if n_int == 0:
                    group_sizes = jnp.zeros((0,), dtype=jnp.float32)
                    u = jnp.zeros((0, 1), dtype=jnp.float32)
                    ret[pop] = cls(
                        pops=(pop,), group_sizes=group_sizes, u=u, log_s=jnp.array(0.0)
                    )
                    continue
                n = n_int

            n_arr = jnp.asarray(n, dtype=jnp.float32)
            group_sizes = n_arr.reshape((1,))
            u = n_arr.reshape((1, 1))
            ret[pop] = cls(
                pops=(pop,), group_sizes=group_sizes, u=u, log_s=jnp.array(0.0)
            )
        return ret
