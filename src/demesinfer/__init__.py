import jax

jax.config.update("jax_enable_x64", True)

from jaxtyping import install_import_hook

# Plus any one of the following:

with install_import_hook("demesinfer", "beartype.beartype"):
    import demesinfer.constr
    import demesinfer.drivers.iicr
    import demesinfer.path
