import jax.numpy as jnp
from interpax import interp1d
from jaxtyping import Array, Float, ScalarLike

from ..iicr import interp


class ExpmCcrInterp(interp.Interpolator):
    """Scalar interpolator for a CCR expm segment.

    We precompute `c(t)` and `log P(no-cross by t)` on a fixed time grid during lifting,
    then interpolate these scalars. This avoids storing/interpolating the full colored
    probability tensor (which is huge) and keeps evaluation vmap/jit-friendly.

    For terminal segments (t1=inf), we clamp evaluation to the last grid point,
    effectively assuming the curve is constant beyond that point.
    """

    ts: Float[Array, "t"]
    cs: Float[Array, "t"]
    log_pnc: Float[Array, "t"]

    def jumps(self, demo):
        return jnp.array([])

    def __call__(self, t: ScalarLike, demo: dict) -> dict[str, ScalarLike]:
        # Clamp to the precomputed grid to avoid extrapolation artifacts.
        t = jnp.clip(t, self.t0, self.ts[-1])
        c = interp1d(t, self.ts, self.cs, method="linear", extrap=False)
        lp = interp1d(t, self.ts, self.log_pnc, method="linear", extrap=False)
        return dict(c=c, log_s=self.state.log_s + lp)


__all__ = ["ExpmCcrInterp"]
