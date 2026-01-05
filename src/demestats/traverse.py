"event tree traversals"

import equinox as eqx
import jax
import jax.numpy as jnp
import networkx as nx
from beartype.typing import Callable, TypeVar
from loguru import logger

from . import events, util
from .event_tree import EventTree, Node

T = TypeVar("T")


def traverse(
    et: EventTree,
    init_state: dict[tuple[Node], T],
    # FIXME i am too lazy to figure out what are the corrct type annotations
    node_callback: Callable,
    lift_callback: Callable,
    aux=None,
    scan_over_lifts: bool = True,
) -> tuple[dict[tuple[Node, ...], T], dict]:
    """Traverse the event tree from the leaves upward and apply
    callbacks to nodes and edges.

    Args:
        et: The event tree to traverse.
        init_state: Initial state for each leaf node.
        node_callback: Callback function for nodes.
        lift_callback: Callback function for lifting between nodes.

    Returns:
        dict: A dictionary mapping nodes to their final states after traversal.
    """
    if aux is None:
        aux = {}

    states = dict(init_state)

    T = et.T_reduced

    def get_parent(node):
        parents = list(T.successors(node))
        assert len(parents) <= 1, "Node should have at most one parent."
        if parents == []:
            return None
        return parents[0]

    def get_children(node):
        children = list(T.predecessors(node))
        assert len(children) <= 2, "Node should have at most two children."
        return children

    out_aux = {}
    nodes_to_process = list(nx.topological_sort(T))
    while nodes_to_process:
        node = nodes_to_process.pop(0)
        # process children, transition upwards, and add the parent to the queue
        node_attrs = T.nodes[node]
        logger.trace("node={} node_attrs={}", node, node_attrs)
        children = get_children(node)
        match len(children):
            case 0:
                # leaf node, no event to process
                state = states[node,]
                node_aux = {}
            case 1:
                # single child, just update the state
                child = children[0]
                state, node_aux = node_callback(
                    node,
                    node_attrs,
                    child_state=states[child, node],
                    aux=aux.get((node,), {}),
                )
                logger.trace(
                    "Processing node={node} with child={child} state {child_state}",
                    node=node,
                    child=child,
                    child_state=states[child, node],
                )
            case 2:
                # multiple children, need to aggregate states
                kw = {}
                for child in children:
                    # this label is guaranteed to exist for nodes that have multiple children
                    label = T.edges[child, node]["label"]
                    kw[label + "_state"] = states[child, node]
                logger.trace(
                    "Processing node={node} with children={children} states={kw}",
                    node=node,
                    children=dict(zip(kw, children)),
                    kw=kw,
                )
                state, node_aux = node_callback(
                    node, node_attrs, **kw, aux=aux.get((node,), {})
                )
            case _:
                raise ValueError("Node has more than two children, cannot process.")

        logger.trace(
            "Resulting state={state}",
            state=state,
        )

        # update the state for the just processed node
        states[node,] = state
        out_aux[node,] = node_aux

        # now lift the node to just before its parent node
        parent = get_parent(node)
        if parent is None:
            # reached the root node, nothing to do
            break

        if get_parent(parent) is None:
            t0i = T.nodes[node]["ti"]
            t1i = T.nodes[parent]["ti"]
            state, edge_aux = lift_callback(
                state=state,
                t0i=t0i,
                t1i=t1i,
                terminal=True,
                constant=True,
                migrations=[],
                aux=aux.get((node, parent), {}),
            )
            states[node, parent] = state
            out_aux[node, parent] = edge_aux
            continue

        def is_bypassable(p):
            ev = T.nodes[p].get("event", None)
            return isinstance(
                ev, (events.MigrationStart, events.MigrationEnd, events.Epoch)
            )

        parent = get_parent(node)
        t0i = T.nodes[node]["ti"]
        t1i = T.nodes[parent]["ti"]
        pairs = [(t0i, t1i, node, parent)]
        while is_bypassable(parent):
            nodes_to_process.remove(parent)
            node = parent
            parent = get_parent(node)
            t0i = T.nodes[node]["ti"]
            t1i = T.nodes[parent]["ti"]
            pairs.append((t0i, t1i, node, parent))
        logger.debug("halted at parent={}", T.nodes[parent])

        # we can either scan across the lifts, or go from t0 to t1 using the ode solver
        # prefer the scan if all the epochs are constant size because then we can use lift_const

        logger.trace(
            "Lifting node {node} to parent {parent} with state {state} and aux {node_aux}",
            node=node,
            parent=parent,
            state=state,
            node_aux=node_aux,
        )

        epochs = []
        const = []
        edges = []
        edge_in_auxs = []
        block = T.nodes[node]["block"]
        for t0i, t1i, n, p in pairs:
            if t0i == t1i:
                continue
            assert T.nodes[n]["block"] == block
            c = et.constant_growth_in(block, t0i, t1i)
            const.append(c)
            epochs.append((t0i, t1i))
            edges.append((n, p))
            edge_in_auxs.append(aux.get((n, p), {}))

        edge_out_aux = {}
        if epochs:
            t0i = epochs[0][0]
            t1i = epochs[-1][1]
            t0p, t1p = [next(iter(et.times[ti])) for ti in [t0i, t1i]]
            migr = []
            for tup in util.migrations_in(et.demodict, t0p, t1p):
                if not set(tup) & block:
                    continue
                assert set(tup).issubset(block)
                migr.append(tup)
            constant = all(const)
            if constant:
                times = jnp.array(epochs)
                state_a, state_na = state.partition()

                def body(state_a, tup):
                    (t0i, t1i), auxi = tup
                    state = eqx.combine(state_a, state_na)
                    state, aux = lift_callback(
                        state=state,
                        t0i=t0i,
                        t1i=t1i,
                        constant=True,
                        migrations=migr,
                        terminal=False,
                        aux=auxi,
                    )
                    return state.partition()[0], aux

                if scan_over_lifts:
                    edge_in_auxs = util.tree_stack(edge_in_auxs)
                    state_a, edge_out_aux = jax.lax.scan(
                        body, state_a, (times, edge_in_auxs)
                    )
                    state = eqx.combine(state_a, state_na)
                    edge_out_aux = util.tree_unstack(edge_out_aux)
                else:  # manually unroll
                    edge_out_aux = []
                    for i in range(times.shape[0]):
                        t0i, t1i = times[i]
                        auxi = edge_in_auxs[i]
                        state, aux = lift_callback(
                            state=state,
                            t0i=t0i,
                            t1i=t1i,
                            constant=True,
                            migrations=migr,
                            terminal=False,
                            aux=auxi,
                        )
                        edge_out_aux.append(aux)
                for (n, p), edge_out_auxi in zip(edges, edge_out_aux):
                    out_aux[n, p] = edge_out_auxi
            else:
                state, edge_out_aux = lift_callback(
                    state=state,
                    t0i=t0i,
                    t1i=t1i,
                    constant=False,
                    migrations=migr,
                    terminal=False,
                    aux=aux.get((node, parent), {}),
                )
                out_aux[node, parent] = edge_out_aux

        states[node, parent] = state

        logger.trace(
            "Got new_state={state}",
            state=state,
        )

    return states, out_aux
