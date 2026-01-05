import demes
import jax
from jaxtyping import install_import_hook

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

with install_import_hook(
    [
        "demestats." + x
        for x in ["constr", "event_tree", "iicr", "loglik", "path", "pexp", "sfs"]
    ],
    "beartype.beartype",
):
    import demestats.constr
    import demestats.event_tree
    import demestats.iicr
    import demestats.loglik
    import demestats.path
    import demestats.pexp
    import demestats.sfs  # noqa: F401

# import sys
# from loguru import logger
# logger.remove()  # remove the old handler. Else, the old one will work along with the new one you've added below'
# logger.add(sys.stderr, level="INFO")

jax.tree_util.register_pytree_node(
    demes.Graph,
    lambda graph: ((), graph.asdict()),
    lambda aux, data: demes.Graph.fromdict(aux),
)
