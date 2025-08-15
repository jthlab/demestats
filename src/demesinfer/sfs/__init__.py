import operator
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial, reduce

import demes
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Sequence
from jaxtyping import Array, ArrayLike, Float, Int, PyTree, Scalar, ScalarLike, Shaped
from loguru import logger
from penzai import pz

import demesinfer.event_tree as event_tree
import demesinfer.sfs.events as events
from demesinfer.path import Path, bind
from demesinfer.traverse import traverse

from .events.state import *


@dataclass
class ExpectedSFS:
    demo: demes.Graph
    num_samples: dict[str, Int[ScalarLike, ""]]
    et: event_tree.EventTree = field(init=False)

    def __post_init__(self):
        if not (self.num_samples.keys() <= {pop.name for pop in self.demo.demes}):
            raise ValueError(
                "num_samples must contain only deme names from the demo, found {} which is not in {}".format(
                    self.num_samples.keys() - {pop.name for pop in self.demo.demes},
                    {pop.name for pop in self.demo.demes},
                )
            )

        self.et = et = event_tree.EventTree(self.demo, events=events)
        # increase migration sample sizes, we need >= 4
        # for continuous migration, we require that there are at least four nodes.
        # so for now we just enforce this globally. slightly wasteful if there is
        # not any cm 🤷.
        leaves = et.leaves
        for j, deme in enumerate(self.demo.demes):
            pop = deme.name
            if self.num_samples.get(pop, 0) >= 4:
                continue
            if (
                len(
                    [m for m in self.demo.migrations if deme.name in [m.source, m.dest]]
                )
                == 0
            ):
                continue
            logger.debug("Upsampling {} to 4 samples", pop)
            node = leaves[pop]
            # deme participates in migrations and has fewer than 4 samples, so we add an upsampling
            # event right above the leaf
            kw = {k: et.nodes[node][k] for k in ["t", "block"]}
            kw["event"] = events.Upsample(pop=pop, m=4)
            v = et._add_node(**kw)
            (parent,) = et._T.successors(node)
            label = et.edges[node, parent].get("label")
            et._remove_edge(node, parent)
            et._add_edge(node, v)
            et._add_edge(v, parent)
            # reconstruct the label if necessary
            if label is not None:
                et.edges[v, parent]["label"] = label
            et._check()

        self._aux = self._setup()

    def bind(self, params: dict[event_tree.Variable, ScalarLike]) -> dict:
        """
        Bind the parameters to the event tree's demo.
        """
        return self.et.bind(params, rescale=True)

    def variables(self) -> Sequence[event_tree.Variable]:
        """
        Return the parameters that can be optimized.
        """
        return self.et.variables()

    def _setup(self) -> dict[tuple[event_tree.Node, ...], dict]:
        setup_state = {}
        for pop, leaf in self.et.leaves.items():
            n = self.num_samples.get(pop, 0)
            setup_state[(leaf,)] = SetupState(
                migrations=frozenset(),
                axes=OrderedDict({pop: n + 1}),
                ns={pop: {pop: n}},
            )
        _, aux = traverse(
            self.et,
            setup_state,
            node_callback=lambda node, node_attrs, **kw: node_attrs["event"].setup(
                demo=self.et.demodict,
                **kw,
            ),
            lift_callback=partial(events.setup_lift, demo=self.et.demodict),
            aux=None,
        )
        return aux

    def __call__(
        self, params: dict[event_tree.Variable, ScalarLike] = {}
    ) -> Float[Array, "*T"]:
        bs = [n + 1 for n in self.num_samples.values()]
        num_derived = jnp.indices(bs)
        num_derived = jnp.rollaxis(num_derived, 0, num_derived.ndim).reshape(
            -1, len(bs)
        )

        def f(nd):
            nd = dict(zip(self.num_samples, nd))
            ret = {}
            for pop, leaf in self.et.leaves.items():
                # some ghost populations may not be sampled. then they have trivial partial leaf likelihood.
                n = self.num_samples.get(pop, 0)
                d = nd.get(pop, 0)
                ret[pop] = jax.nn.one_hot(jnp.array([d]), n + 1)[0]
            return ret

        X = jax.vmap(f)(num_derived)
        res = self.dp(params, X)
        return res.at[jnp.array([0, -1])].set(0.0).reshape(bs)

    def tensor_prod(
        self,
        X: PyTree[Shaped[ArrayLike, "B ?D"], "T"],
        params: dict[Path, ScalarLike] = {},
    ) -> Float[Array, "B"]:
        demo = self.bind(params)

        for pop in X:
            n = self.num_samples.get(pop, 0)
            assert X[pop].shape[1] == n + 1

        def f(v):
            u = jnp.eye(v.shape[1])[jnp.array([0, -1])]
            return jnp.concatenate([u, v])

        Xa = jax.tree.map(f, X)
        states = _call(
            Xa,
            self.et,
            demo,
            self._aux,
        )
        Pi = reduce(
            operator.mul, jax.tree.map(lambda a: a[:, jnp.array([0, -1])], X).values()
        )
        ret = states.phi[2:]
        ret -= Pi[:, 0] * states.phi[0]
        ret -= Pi[:, 1] * states.phi[1]
        return ret * self.et.scaling_factor

    def dp(
        self,
        params: dict[event_tree.Variable, ScalarLike],
        X: dict[str, Float[ArrayLike, "batch *T"]],
    ) -> Float[Array, "batch"]:
        demo = self.bind(params)
        pops = {pop.name for pop in self.demo.demes}
        state = _call(
            X,
            self.et,
            demo,
            self._aux,
        )
        return state.phi * self.et.scaling_factor


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0,) + (None,) * 3)
def _call(
    X: dict[str, Float[Array, "T"]],
    et: event_tree.EventTree,
    demo: dict,
    aux: dict,
) -> State:
    states = {}
    for pop, node in et.leaves.items():
        Xp = X.get(pop, jnp.ones(1))
        states[node,] = State(
            pl=pz.nx.wrap(Xp, pop),
            phi=0.0,
            l0=Xp[0],
        )

    def node_callback(node, node_attrs, **kw):
        kw["demo"] = demo
        return node_attrs["event"](**kw)

    def lift_callback(
        state: State, t0: Path, t1: Path, terminal: bool, aux: dict
    ) -> StateReturn:
        return events.lift(
            state=state, t0=t0, t1=t1, terminal=terminal, demo=demo, aux=aux
        )

    states, _ = traverse(
        et, states, node_callback, lift_callback, aux=aux, _fuse_lifts=True
    )
    return states[et.root,]
