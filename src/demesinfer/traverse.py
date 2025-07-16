"event tree traversals"

import networkx as nx
from beartype import beartype
from beartype.typing import Callable, TypeVar
from jaxtyping import jaxtyped
from loguru import logger

from .event_tree import EventTree, Node

T = TypeVar("T")


def traverse(
    et: EventTree,
    init_state: dict[tuple[Node], T],
    # FIXME i am too lazy to figure out what are the corrct type annotations
    node_callback: Callable,
    lift_callback: Callable,
    aux=None,
    _fuse_lifts: bool = True,
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

    T = et.T

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

    # fifo queue for processing nodes
    q = list(et.leaves)

    out_aux = {}

    for node in nx.topological_sort(T):
        # process children, transition upwards, and add the parent to the queue
        node_attrs = T.nodes[node]
        logger.trace("node={} node_attrs={}", node, node_attrs)
        if node is None:
            # reached the root node, nothing to do
            assert len(q) == 0
            continue
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
        # if _fuse_lifts, then we don't stop until we meet a node with
        # multiple children. this prevents us from doing multiple lifts,
        # which are expensive, when we can instead just do e.g. a single ode solve
        t0 = T.nodes[node]["t"]
        parent = get_parent(node)
        if parent is None:
            # reached the root node, nothing to do
            break
        t1 = T.nodes[parent]["t"]
        terminal = get_parent(parent) is None

        # short-circuit the lift in cases where the time is the same.
        if abs(et.get_path(t0) - et.get_path(t1)) < 1e-8:
            state, node_aux = state, {}
        else:
            node_attrs = T.nodes[node]
            logger.trace(
                "Lifting node {node} to parent {parent} with state {state} and aux {node_aux}",
                node=node,
                parent=parent,
                state=state,
                node_aux=node_aux,
            )
            state, node_aux = lift_callback(
                state=state,
                t0=t0,
                t1=t1,
                terminal=terminal,
                aux=aux.get((node, parent), {}),
            )
            logger.trace(
                "Got new_state={state}",
                state=state,
            )
        states[node, parent] = state
        out_aux[node, parent] = node_aux

    return states, out_aux
