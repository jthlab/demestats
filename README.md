# demestats

**demestats** is a Python package for computing population genetic statistics from demographic models and tree sequences. It serves as a unified framework for:

- **momi3**: A complete rewrite of the momi/momi2 packages for expected Sample Frequency Spectrum (SFS) calculation and demographic inference.
- **CCR**: Cross-Coalescence Rates for analyzing population divergence and gene flow.
- **IICR**: Inverse Instantaneous Coalescence Rate curves.

## Documentation

Full documentation is available at **[demestats.readthedocs.org](https://demestats.readthedocs.org)**.

## Installation

`demestats` will eventually be available on PyPI. Until then, please install directly from GitHub:

```bash
pip install git+https://github.com/jthlab/demestats
```

We recommend using a virtual environment (e.g., `venv` or `conda`) to avoid conflicts with system packages.

## Submodules

### momi3 (SFS)
momi3 computes the expected SFS for flexible demographic models (using `demes`) and provides tools for likelihood-based inference. It introduces random projection methods for faster evaluation of complex models.

### CCR & IICR
These modules provide statistics for analyzing coalescence rates across time, useful for inferring changes in population size and migration history.

## License

MIT License. See `docs/index.md` or `LICENSE` for details.
