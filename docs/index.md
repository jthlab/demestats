# Welcome to demestats' documentation
## momi3 (within demestats)

momi (short for MOran Models for Inference) is a Python package for computing the
expected sample frequency spectrum (SFS) and using it to infer demographic history.

This third version (momi3) is a complete rewrite of the original momi and momi2 packages. It
introduces greater flexibility in model specification and improved performance.

momi3 is implemented as a **component of** the broader `demestats` package (primarily in
`demestats.sfs`). For installation instructions, tutorials, and API reference, see the
documentation pages in this site.

The method for momi3 is described in the following preprint:

Dilber, E., & Terhorst, J. (2024, March 29). Faster inference of complex demographic
models from large allele frequency spectra [Preprint]. bioRxiv.
https://doi.org/10.1101/2024.03.26.586844

## Content
To navigate the documentation, please refer to the [``Notation``](https://demestats.readthedocs.io/en/latest/notation.html) section *first* to understand the representation of parameters within a demographic model. 

- [``momi3 Tutorial``](https://demestats.readthedocs.io/en/latest/momi3_tutorial.html) introduces all of the core functions of ``momi3``
- [``Model Constraints``](https://demestats.readthedocs.io/en/latest/model_constraints.html) shows how to modify model constraints
- [``Random Projection``](https://demestats.readthedocs.io/en/latest/random_projection.html) introduces an approximation of the full expected SFS
- [``SFS Optimization``](https://demestats.readthedocs.io/en/latest/sfs_optimization.html) demonstrates how to construct custom inference pipelines using ``scipy.minimize`` and the SFS
- [``Special Examples``](https://demestats.readthedocs.io/en/latest/special_examples.html) shows examples of constructing complex models (e.g. exponential growth, admixture, bottleneck)
- [``Loading Data``](https://demestats.readthedocs.io/en/latest/loading_data.html) shows how to import empirical or externally generated datasets for use in SFS and IICR analyses

```{toctree}
:hidden:
:maxdepth: 1

Installation <installation>
Notation <notation>
momi3 Tutorial <momi3_tutorial>
Model Constraints <model_constraints>
Random Projection <random_projection>
SFS Optimization <sfs_optimization>
Special Examples <special_examples>
Loading Data <loading_data>
API <api/index>
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

## License

The MIT License (MIT)

Copyright (c) 2023 Enes Dilber, Jiatong (Kevin) Liang, Junyan (Samuel) Tan, and Jonathan Terhorst

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
