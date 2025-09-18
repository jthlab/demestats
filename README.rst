.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/momi3.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/momi3
    .. image:: https://readthedocs.org/projects/momi3/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://momi3.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/momi3/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/momi3
    .. image:: https://img.shields.io/pypi/v/momi3.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/momi3/
    .. image:: https://img.shields.io/conda/vn/conda-forge/momi3.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/momi3
    .. image:: https://pepy.tech/badge/momi3/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/momi3
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/momi3

    .. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
        :alt: Project generated with PyScaffold
        :target: https://pyscaffold.org/


=====
momi3
=====

momi (short for MOran Models for Inference) is a Python package for computing the expected sample frequency spectrum (SFS)—a key summary statistic in population genetics—and using it to infer demographic history.

This third version is a complete rewrite of the original momi and momi2 packages. It introduces several major improvements, including greater flexibility in model specification and enhanced performance and scalability.

Additionally, momi3 is now being built as a core component of the broader ``demesinfer`` package, providing a more comprehensive framework for demographic inference.

For installation instructions, a tutorial, and API reference, please refer to the documentation.

The method is described in the following preprint:

Dilber, E., & Terhorst, J. (2024, March 29). Faster inference of complex demographic models from large allele frequency spectra [Preprint]. bioRxiv. https://doi.org/10.1101/2024.03.26.586844