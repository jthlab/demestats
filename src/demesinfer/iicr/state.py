from typing import NamedTuple

from jaxtyping import Array, Float


class State(NamedTuple):
    # p is an (d,)*n  array denoting the joint probability that each of N lineages is
    # in each of D demes.
    # c is the probability that the first coalescence event has not occured by time t.
    p: Float[Array, "..."]  # (d+1,)*n
    pops: tuple[str]
    log_s: Float[Array, ""]

    def check_shape(self):
        assert p.shape[0] == 1 + len(self.pops)


SetupState = None
StateReturn = tuple[State, dict]
SetupReturn = tuple[SetupState, dict]
