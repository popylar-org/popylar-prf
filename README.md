
[![github license badge](https://img.shields.io/github/license/popylar-org/popylar-prf)](https://github.com/popylar-org/popylar-prf/blob/main/LICENSE)
[![build](https://github.com/popylar-org/popylar-prf/actions/workflows/build.yml/badge.svg)](https://github.com/popylar-org/popylar-prf/actions/workflows/build.yml)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## How to use popylar_prf

A modern Python implementation for population receptive field model fitting.

## Installation

To install the development version of popylar_prf from GitHub, run:

```console
git clone git@github.com:popylar-org/popylar-prf.git
cd popylar-prf
python -m pip install .
```

The package relies on [Keras](https://keras.io/) for multi-backend model fitting. To install popylar_prf with
the Tensorflow backend, run:

```console
python -m pip install .[tensorflow]
```

To install the PyTorch backend, run:

```console
python -m pip install .[torch]
```

To install the JAX backend, run:

```console
python -m pip install .[jax]
```

## Documentation

The local documentation can be build as HTML files with:

```console
# In popylar-prf directory
cd docs/
make html
```

The documentation can then be opened in the browser from `_build/html/index.html`.

## Development

The project setup for developers is documented in [project_setup.md](project_setup.md). To make an editable install
with development dependencies, run:

```console
python -m pip install -e .[dev]
```

The test suite can be run with:

```console
python -m pytest
```

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the
[NLeSC/python-template](https://github.com/NLeSC/python-template).

## Copyright

2025, Netherlands eScience Center, Vrije Universiteit Amsterdam, Netherlands Institute for Neuroscience
