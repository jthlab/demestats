from jaxtyping import ScalarLike

from .path import Path


def rescale_demo(demo: dict, scaling_factor: ScalarLike = 1.0) -> dict:
    """Rescale the demo according to the scaling factor."""

    def f(node, path):
        if isinstance(node, dict):
            return {k: f(v, path + (k,)) for k, v in node.items()}
        elif isinstance(node, list):
            return [f(v, path + (i,)) for i, v in enumerate(node)]
        # leaf reached, rescaling logic
        value = node
        match path:
            case ("demes", i, "start_time" | "end_time"):
                value /= scaling_factor
            case (
                "demes",
                i,
                "epochs",
                j,
                "start_size" | "end_size" | "start_time" | "end_time",
            ):
                value /= scaling_factor
            case ("demes", i, "epochs", j, "selfing_rate" | "cloning_rate"):
                value *= scaling_factor
            case ("pulses", i, "time"):
                value /= scaling_factor
            case ("migrations", i, "start_time" | "end_time"):
                value /= scaling_factor
            case ("migrations", i, "rate"):
                value *= scaling_factor
            case ("pulses", i, "time"):
                value /= scaling_factor
            case _:
                pass
        return value

    return f(demo, ())
