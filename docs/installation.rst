============
Installation
============

We recommend using a virtual environment to manage dependencies for the ``momi3`` package. This helps avoid conflicts with other Python packages and ensures a clean installation.

To set up a virtual environment, you can use `venv` or `conda`. Here’s how to do it with `venv`:

.. code:: bash

   python -m venv momi3-env
   source momi3-env/bin/activate  # On Windows use `momi3-env\Scripts\activate`

Alternatively, if you prefer `conda`, you can create an environment with:

.. code:: bash

   conda create -n momi3-env
   conda activate momi3-env

You can then install ``momi3`` and all its dependencies using pip from the GitHub repository:

.. code:: bash

   pip install git+https://github.com/jthlab/momi3.git@iicr

This will fetch the latest version of the ``iicr`` branch along with all necessary dependencies.


Before getting started, ensure your environment includes the following:

- Python ≥ 3.11
- The following Python packages:
  - ``momi3``
  - ``diffrax``
  - ``frozendict``
  - ``jax``
  - ``jaxlib``
  - ``numpy``
  - ``opt_einsum``
  - ``scipy``
  - ``sympy``
  - ``seaborn``
  - ``networkx``
  - ``pandas``
  - ``tqdm``
  - ``demes``
  - ``msprime``
  - ``demesdraw``
  - ``joblib``
  - ``sparse``
  - ``gmpy2``
  - ``optax``
  - ``cvxpy``
  - ``jaxopt``