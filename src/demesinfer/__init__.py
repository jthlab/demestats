import jax

jax.config.update("jax_enable_x64", True)

from jaxtyping import install_import_hook

with install_import_hook("demesinfer", "beartype.beartype"):
    import demesinfer.constr
    import demesinfer.event_tree
    import demesinfer.iicr
    import demesinfer.loglik
    import demesinfer.path
    import demesinfer.pexp
    import demesinfer.sfs
    import demesinfer.sfs.events

# import sys
# from loguru import logger
# logger.remove()  # remove the old handler. Else, the old one will work along with the new one you've added below'
# logger.add(sys.stderr, level="INFO")

import demes

jax.tree_util.register_pytree_node(
    demes.Graph,
    lambda graph: ((), graph.asdict()),
    lambda aux, data: demes.Graph.fromdict(aux),
)
