"""Admixture events."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Int, ScalarLike
from penzai import pz

import demesinfer.events as base
from demesinfer.util import log_hypergeom

from .state import *


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
