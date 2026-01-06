import operator
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial, reduce

import demes
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Sequence
from jaxtyping import Array, ArrayLike, Float, Int, PyTree, ScalarLike, Shaped
from loguru import logger
from penzai import pz

import demestats.event_tree as event_tree
import demestats.sfs.events as events
from demestats.path import Path
from demestats.traverse import traverse

from .events.state import SetupState, State

Params = dict[event_tree.Variable, ScalarLike]
PruneSpec = tuple[str, int, Path | event_tree.Node | None]
PruneInput = Mapping[str, int] | Sequence[PruneSpec] | None


@dataclass
class ExpectedSFS:
    """
    Build an ExpectedSFS object that can be later used to compute the full expected
    site frequency spectrum or the projected site frequency spectrum.

    Parameters
    ----------
        demo : demes.Graph
            A ``demes`` graph
        num_samples : dict
            A dictionary specifying how many haploids per population to use to compute
            the expected SFS. The name of the
            populations must match the exact names use in ``demo``.
        prune : mapping or sequence, optional
            Optional manual downsampling events. Provide either a mapping
            ``{deme_name: m}`` to downsample directly above leaves, or a sequence of
            ``(deme_name, m[, at])`` tuples where ``at`` is a node id or a demes time
            path to insert the downsample event above that node.

    Returns:
        ExpectedSFS: an ExpectedSFS object used to compute expected site frequency spectrum

    Notes
    -----
    From a user perspective, understanding the underlying structure of an ExpectedSFS object
    is not necessary. The only functions that a user would use is ``ExpectedSFS.__call__``
    which computes the full expected site frequency spectrum and ``ExpectedSFS.tensor_prod`` which
    computes the projected site frequency spectrum.

    Example:
    ::
       ESFS = ExpectedSFS(demo.to_demes(), num_samples=afs_samples)

    Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.

    See Also
    --------
    demestats.sfs.ExpectedSFS.__call__
    demestats.sfs.ExpectedSFS.tensor_prod
    """

    demo: demes.Graph
    num_samples: dict[str, Int[ScalarLike, ""]]
    prune: PruneInput = None
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
            kw = {k: et.nodes[node][k] for k in ["t", "block", "ti"]}
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

        if self.prune:
            self._apply_pruning(self.prune)

        self._aux = self._setup()

    def _apply_pruning(self, prune: PruneInput) -> None:
        specs = self._normalize_prune(prune)
        if not specs:
            return
        et = self.et
        for pop, m, at in specs:
            if pop not in {d.name for d in self.demo.demes}:
                raise ValueError(f"unknown deme {pop} in pruning spec")
            if isinstance(at, tuple):
                target = self._resolve_prune_target(pop, at)
                self._insert_downsample(
                    target["node"],
                    pop,
                    m,
                    t_override=target.get("t_override"),
                    ti_override=target.get("ti_override"),
                )
            else:
                node = self._resolve_prune_node(pop, at)
                self._insert_downsample(node, pop, m)
        et.__dict__.pop("T_reduced", None)

    def _normalize_prune(self, prune: PruneInput) -> list[PruneSpec]:
        if prune is None:
            return []
        if isinstance(prune, Mapping):
            return [(pop, int(m), None) for pop, m in prune.items()]
        specs = []
        for item in prune:
            if len(item) == 2:
                pop, m = item
                at = None
            elif len(item) == 3:
                pop, m, at = item
            else:
                raise ValueError("prune entries must be (pop, m) or (pop, m, at)")
            specs.append((pop, int(m), at))
        return specs

    def _resolve_prune_node(
        self, pop: str, at: Path | event_tree.Node | None
    ) -> event_tree.Node:
        et = self.et
        if at is None:
            return et.leaves[pop]
        if isinstance(at, int):
            if at not in et.nodes:
                raise ValueError(f"prune node {at} not in event tree")
            if pop not in et.nodes[at]["block"]:
                raise ValueError(f"deme {pop} not in node {at} block")
            return at
        raise ValueError(f"invalid prune locator {at!r}")

    def _resolve_prune_target(self, pop: str, at: Path) -> dict:
        et = self.et
        matches = [
            n
            for n in et.nodes
            if et.nodes[n]["t"] == at and pop in et.nodes[n]["block"]
        ]
        if len(matches) == 1:
            return {"node": matches[0]}
        if len(matches) > 1:
            raise ValueError(
                f"prune locator {at} matched {len(matches)} nodes for {pop}: {matches}"
            )
        edge_matches = []
        for child, parent in et.edges:
            if et.nodes[parent]["t"] != at:
                continue
            if pop not in et.nodes[child]["block"]:
                continue
            edge_matches.append((child, parent))
        if len(edge_matches) != 1:
            raise ValueError(
                f"prune locator {at} matched {len(edge_matches)} edges for {pop}: {edge_matches}"
            )
        child, parent = edge_matches[0]
        return {
            "node": child,
            "t_override": at,
            "ti_override": et.nodes[parent].get("ti"),
        }

    def _insert_downsample(
        self,
        node: event_tree.Node,
        pop: str,
        m: int,
        t_override: Path | None = None,
        ti_override: int | None = None,
    ) -> None:
        et = self.et
        parents = list(et._T.successors(node))
        if len(parents) != 1:
            raise ValueError(f"node {node} has no parent to attach pruning")
        (parent,) = parents
        kw = {k: et.nodes[node][k] for k in ["t", "block", "ti"]}
        if t_override is not None:
            kw["t"] = t_override
        if ti_override is not None:
            kw["ti"] = ti_override
        kw["event"] = events.Downsample(pop=pop, m=m)
        v = et._add_node(**kw)
        label = et.edges[node, parent].get("label")
        et._remove_edge(node, parent)
        et._add_edge(node, v)
        et._add_edge(v, parent)
        if label is not None:
            et.edges[v, parent]["label"] = label
        et._check()

    def bind(self, params: Params) -> dict:
        """
        Bind the parameters to the event tree's demo.
        """
        return self.et.bind(params, rescale=True)

    def variable_for(self, path: Path) -> event_tree.Variable:
        """Return the variable associated with a given path."""
        return self.et.variable_for(path)

    @property
    def variables(self) -> Sequence[event_tree.Variable]:
        """
        Return the parameters that can be optimized.
        """
        return self.et.variables

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
            scan_over_lifts=False,
        )
        return aux

    def __call__(self, params: Params = {}) -> Float[Array, "*T"]:
        """
        Computes the full expected site frequency spectrum under a given set of model parameters and values

        Parameters
        ----------
            params : dict
                A dictionary of model parameters and their value
        Returns:
            Float[Array]: An array of float values that represent the full expected site frequency spectrum

        Notes
        -----
        You must first construct an ExpectedSFS object. See the ExpectedSFS API.

        Example:
        ::
            ESFS = ExpectedSFS(demo.to_demes(), num_samples=afs_samples)
            params = {param_key: val}
            esfs = ESFS(params)

        Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.

        See Also
        --------
        demestats.sfs.ExpectedSFS
        """
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
        params: Params = {},
    ) -> Float[Array, "B"]:
        """
        Computes the projected expected site frequency spectrum under a given set random
        projection vectors and model parameters. A tensor product operation between the random projections and
        the expected site frequency spectrum is applied to obtain the projected
        SFS. To obtain the appropriate projection vectors, one can
        use the function ``demestats.loglik.sfs_loglik.prepare_projection``.

        Parameters
        ----------
            X: dict
                A dictionary of random projection vectors
            params : dict
                A dictionary of model parameters and their value
        Returns:
            Float[Array]: An array of float values that represent the projected expected site frequency spectrum

        Notes
        -----
        You must first construct an ExpectedSFS object. See the ExpectedSFS API.

        Example:
        ::
            proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
            esfs_obj = ExpectedSFS(demo.to_demes(), num_samples=afs_samples)
            lowdim_esfs = esfs_obj.tensor_prod(proj_dict, paths)

        Please refer to ``Random Projection`` section for a specific example, the above provided codes are just outlines of how to call on the functions.

        See Also
        --------
        demestats.loglik.sfs_loglik.prepare_projection
        """
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
        Pi = reduce(operator.mul, jax.tree.map(lambda a: a[:, [0, -1]], X).values())
        ret = states.phi[2:]
        ret -= Pi[:, 0] * states.phi[0]
        ret -= Pi[:, 1] * states.phi[1]
        return ret * self.et.scaling_factor

    def dp(
        self,
        params: Params,
        X: dict[str, Float[ArrayLike, "batch *T"]],
    ) -> Float[Array, "batch"]:
        demo = self.bind(params)
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

    lift_callback = partial(events.lift, demo=demo)

    states, _ = traverse(et, states, node_callback, lift_callback, aux=aux)
    return states[et.root,]
