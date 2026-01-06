# Installation

We recommend using a virtual environment to manage dependencies for `demestats`. This
helps avoid conflicts with other Python packages and ensures a clean installation.

Using `venv`:

```bash
python -m venv demestats-env
source demestats-env/bin/activate  # On Windows use `demestats-env\Scripts\activate`
```

Using `conda`:

```bash
conda create -n demestats-env
conda activate demestats-env
```

Install from GitHub (example):

```bash
pip install git+https://github.com/jthlab/momi3.git@iicr
```

Requirements
------------

Before getting started, ensure your environment includes:

- Python ≥ 3.11
- Core scientific stack: `numpy`, `scipy`, `jax`, `jaxlib`
- Demography tooling: `demes`, `msprime`, `demesdraw`

For a full dependency list, see `pyproject.toml`.
