"""Admixture events."""

from dataclasses import dataclass

import demestats.events as base

from .state import SetupReturn, SetupState, State, StateReturn


@dataclass(kw_only=True)
class MigrationStart(base.MigrationStart):
    def setup(self, demo: dict, aux: dict, child_state: SetupState) -> SetupReturn:
        mig = child_state.migrations | {(self.source, self.dest)}
        return child_state._replace(migrations=mig), {}

    def __call__(self, demo: dict, aux: dict, child_state: State) -> StateReturn:
        return child_state, {}


@dataclass(kw_only=True)
class MigrationEnd(base.MigrationEnd):
    def setup(self, demo: dict, aux: dict, child_state: SetupState) -> SetupReturn:
        mig = child_state.migrations - {(self.source, self.dest)}
        return child_state._replace(migrations=mig), {}

    def __call__(self, demo: dict, aux: dict, child_state: State) -> StateReturn:
        return child_state, {}
