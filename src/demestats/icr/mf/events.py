from dataclasses import dataclass

import jax.numpy as jnp

from .. import events as base
from . import lift
from .state import StateMf as State

NoOp = base.NoOp
Epoch = base.Epoch
State = State


@dataclass(kw_only=True)
class PopulationStart(base.PopulationStart):
    pass


def _merge_pops_into_recipient(
    state: State, *, donor: str, recipient: str
) -> tuple[tuple[str, ...], jnp.ndarray]:
    pops = list(state.pops)
    donor_idx = pops.index(donor)
    recip_idx = pops.index(recipient)
    u = state.u
    u = u.at[:, recip_idx].add(u[:, donor_idx])
    keep = jnp.array([i for i in range(len(pops)) if i != donor_idx], dtype=jnp.int32)
    pops.pop(donor_idx)
    return tuple(pops), u[:, keep]


@dataclass(kw_only=True)
class Split1(base.Split1):
    def __call__(self, demo: dict, aux: dict, child_state: State) -> tuple[State, dict]:
        del demo, aux
        pops, u = _merge_pops_into_recipient(
            child_state, donor=self.donor, recipient=self.recipient
        )
        return (
            State(
                pops=pops,
                group_sizes=child_state.group_sizes,
                u=u,
                log_s=child_state.log_s,
            ),
            {},
        )


@dataclass(kw_only=True)
class Split2(base.Split2):
    def __call__(
        self, demo: dict, aux: dict, donor_state: State, recipient_state: State
    ) -> tuple[State, dict]:
        del demo, aux
        assert set(donor_state.pops).isdisjoint(set(recipient_state.pops))
        pops = recipient_state.pops + donor_state.pops

        d1 = recipient_state.d
        d2 = donor_state.d
        g1 = recipient_state.g
        g2 = donor_state.g
        u1 = jnp.concatenate(
            [recipient_state.u, jnp.zeros((g1, d2), dtype=recipient_state.u.dtype)],
            axis=1,
        )
        u2 = jnp.concatenate(
            [jnp.zeros((g2, d1), dtype=donor_state.u.dtype), donor_state.u], axis=1
        )
        u = jnp.concatenate([u1, u2], axis=0)
        group_sizes = jnp.concatenate(
            [recipient_state.group_sizes, donor_state.group_sizes], axis=0
        )
        merged = State(
            pops=pops,
            group_sizes=group_sizes,
            u=u,
            log_s=recipient_state.log_s + donor_state.log_s,
        )
        pops2, u2 = _merge_pops_into_recipient(
            merged, donor=self.donor, recipient=self.recipient
        )
        return (
            State(
                pops=pops2,
                group_sizes=merged.group_sizes,
                u=u2,
                log_s=merged.log_s,
            ),
            {},
        )


@dataclass(kw_only=True)
class Merge(base.Merge):
    def __call__(
        self, demo: dict, aux: dict, pop1_state: State, pop2_state: State
    ) -> tuple[State, dict]:
        del demo, aux
        assert set(pop1_state.pops).isdisjoint(set(pop2_state.pops))
        pops = pop1_state.pops + pop2_state.pops

        d1 = pop1_state.d
        d2 = pop2_state.d
        g1 = pop1_state.g
        g2 = pop2_state.g
        u1 = jnp.concatenate(
            [pop1_state.u, jnp.zeros((g1, d2), dtype=pop1_state.u.dtype)], axis=1
        )
        u2 = jnp.concatenate(
            [jnp.zeros((g2, d1), dtype=pop2_state.u.dtype), pop2_state.u], axis=1
        )
        u = jnp.concatenate([u1, u2], axis=0)
        group_sizes = jnp.concatenate([pop1_state.group_sizes, pop2_state.group_sizes])
        return (
            State(
                pops=pops,
                group_sizes=group_sizes,
                u=u,
                log_s=pop1_state.log_s + pop2_state.log_s,
            ),
            {},
        )


@dataclass(kw_only=True)
class MigrationStart(base.MigrationStart):
    def __call__(self, demo: dict, aux: dict, child_state: State) -> tuple[State, dict]:
        return child_state, {}


@dataclass(kw_only=True)
class MigrationEnd(base.MigrationEnd):
    def __call__(self, demo: dict, aux: dict, child_state: State) -> tuple[State, dict]:
        return child_state, {}


@dataclass(kw_only=True)
class Pulse(base.Pulse):
    def __call__(self, demo: dict, aux: dict, child_state: State) -> tuple[State, dict]:
        del aux
        p = self.prop_fun(demo)
        pops = list(child_state.pops)
        src = pops.index(self.source)
        dst = pops.index(self.dest)

        u = child_state.u
        moved = p * u[:, dst]
        u = u.at[:, dst].add(-moved).at[:, src].add(moved)
        return (
            State(
                pops=child_state.pops,
                group_sizes=child_state.group_sizes,
                u=u,
                log_s=child_state.log_s,
            ),
            {},
        )


@dataclass(kw_only=True)
class Admix(base.Admix):
    def __call__(self, demo: dict, aux: dict, child_state: State) -> tuple[State, dict]:
        del aux
        p = self.prop_fun(demo)
        pops = list(child_state.pops)
        child = pops.index(self.child)

        u = child_state.u
        if self.parent1 in pops:
            p1 = pops.index(self.parent1)
        else:
            pops.append(self.parent1)
            u = jnp.concatenate([u, jnp.zeros((u.shape[0], 1), dtype=u.dtype)], axis=1)
            p1 = len(pops) - 1
        if self.parent2 in pops:
            p2 = pops.index(self.parent2)
        else:
            pops.append(self.parent2)
            u = jnp.concatenate([u, jnp.zeros((u.shape[0], 1), dtype=u.dtype)], axis=1)
            p2 = len(pops) - 1

        u_child = u[:, child]
        u = u.at[:, p1].add((1 - p) * u_child).at[:, p2].add(p * u_child)
        keep = jnp.array([i for i in range(len(pops)) if i != child], dtype=jnp.int32)
        pops.pop(child)
        return (
            State(
                pops=tuple(pops),
                group_sizes=child_state.group_sizes,
                u=u[:, keep],
                log_s=child_state.log_s,
            ),
            {},
        )


State = State
