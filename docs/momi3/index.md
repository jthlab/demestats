# momi3

momi (short for MOran Models for Inference) is a Python package for computing the
expected sample frequency spectrum (SFS) and using it to infer demographic history.

This third version (momi3) is a complete rewrite of the original momi and momi2 packages. It
introduces greater flexibility in model specification and improved performance.

momi3 is implemented as a **component of** the broader `demestats` package (primarily in
`demestats.sfs`).

The method for momi3 is described in the following preprint:

Dilber, E., & Terhorst, J. (2024, March 29). Faster inference of complex demographic
models from large allele frequency spectra [Preprint]. bioRxiv.
https://doi.org/10.1101/2024.03.26.586844

## Content

- [``momi3 Tutorial``](momi3_tutorial.md) introduces all of the core functions of ``momi3``
- [``Model Constraints``](model_constraints.md) shows how to modify model constraints
- [``Random Projection``](random_projection.md) introduces an approximation of the full expected SFS
- [``SFS Optimization``](sfs_optimization.md) demonstrates how to construct custom inference pipelines using ``scipy.minimize`` and the SFS
- [``Special Examples``](special_examples.md) shows examples of constructing complex models (e.g. exponential growth, admixture, bottleneck)

```{toctree}
:hidden:
:maxdepth: 1

momi3 Tutorial <momi3_tutorial>
Model Constraints <model_constraints>
Random Projection <random_projection>
SFS Optimization <sfs_optimization>
Pruning <pruning>
Special Examples <special_examples>
```
