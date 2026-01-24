# Welcome to demestats

`demestats` is a Python package for computing population genetic statistics from demographic models and tree sequences. It includes efficient implementations for:

- **SFS** (momi3): Expected Sample Frequency Spectrum calculation and demographic inference.
- **CCR**: Cross-Coalescence Rates.
- **IICR**: Inverse Instantaneous Coalescence Rate curves.

## Installation
`demestats` will eventually be available on PyPI. Until then, please install from GitHub:

```bash
pip install git+https://github.com/jthlab/demestats
```

We recommend using a virtual environment (e.g., `venv` or `conda`) to avoid conflicts.

## Getting Started

- [Notation](notation.md)
- [Loading Data](loading_data.md)

## Submodules

- [momi3 (SFS)](momi3/index.md)
- [CCR (Cross-Coalescence Rate)](ccr.md)

```{toctree}
:hidden:
:maxdepth: 2

Notation <notation>
Loading Data <loading_data>
momi3 (SFS) <momi3/index>
CCR <ccr>
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
