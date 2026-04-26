# ICR

momi (short for MOran Models for Inference) is a Python package for computing the
expected sample frequency spectrum (SFS) and using it to infer demographic history.

Within `demestats` package, all ICR related functions are implemented under `demestats.icr.ICRCurve` and meanfield approximations
implemented under `demestats.icr.ICRMeanFieldCurve`.

The method for ICR is described in the following preprint:

Liang, J., & Terhorst, J. (2026, April 10). Computing coalescence rates for complex demographies and sampling configurations [Preprint]. bioRxiv.
https://doi.org/10.64898/2026.04.09.717519

## Content

- [``ICR Tutorial``](icr_tutorial.md) introduces all of the core functions of ``ICR``
- [``Model Constraints``](../momi3/model_constraints.md) shows how to modify model constraints
- [``ICR Optimization``](icr_optimization.md) demonstrates how to construct custom inference pipelines using ``scipy.minimize`` and the SFS
- [``Special Examples``](../momi3/special_examples.md) shows examples of constructing complex models (e.g. exponential growth, admixture, bottleneck)

```{toctree}
:hidden:
:maxdepth: 1

ICR Tutorial <icr_tutorial>
Model Constraints <../momi3/model_constraints>
ICR Optimization <icr_optimization>
Special Examples <../momi3/special_examples>
```
