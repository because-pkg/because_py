# because-py

**Python NumPyro backend for the R [`because`](https://because-pkg.github.io/because/) package.**

[![Docs](https://img.shields.io/badge/docs-because--py-0076BA)](https://because-pkg.github.io/because_py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python ≥ 3.8](https://img.shields.io/badge/python-≥3.8-blue)](https://www.python.org/)

because-py translates structural equation models (SEMs) specified as R-style formulas into fully probabilistic [NumPyro](https://num.pyro.ai/) models and samples them with HMC/NUTS via JAX.

---

## Installation

### From R (recommended)

because-py is designed to be used from R via the `because` package. A one-step helper installs all required Python dependencies automatically:

```r
library(because)
install_because_numpyro()
```

This installs `numpyro`, `jax`, `jaxlib`, `networkx`, `funsor`, and `because_py` into your active Python environment.

See the [Installation & Setup](https://because-pkg.github.io/because_py/articles/installation.html) article for full details including Miniconda setup and troubleshooting.

### Directly in Python

```bash
pip install git+https://github.com/because-pkg/because_py.git
```

---

## Quick start

### From R

```r
library(because)

set.seed(42)
n   <- 200
x   <- rnorm(n)
y   <- 0.7 * x + rnorm(n)
dat <- data.frame(x = x, y = y)

fit <- because(
  equations = list(y ~ x),
  data      = dat,
  engine    = "numpyro",
  n.iter    = 1000,
  n.burnin  = 500
)
summary(fit)
```

### Directly in Python

```python
import because
import numpy as np

rng = np.random.default_rng(42)
n   = 200
x   = rng.normal(size=n)
y   = 0.7 * x + rng.normal(size=n)

result = because.fit(
    equations   = ["y ~ x"],
    data        = {"x": x, "y": y},
    num_samples = 1000,
    num_warmup  = 500
)
```

---

## Documentation

Full documentation including articles and API reference:
**[because-pkg.github.io/because_py/](https://because-pkg.github.io/because_py/)**

For the R package documentation:
**[because-pkg.github.io](https://because-pkg.github.io)**

---

## How it works

because-py parses SEM formulas into a directed acyclic graph (DAG), topologically sorts the nodes, and generates a NumPyro model closure on the fly. The model is sampled with NUTS (No-U-Turn Sampler) via JAX — supporting CPU, GPU, and TPU backends.

---

## Requirements

- Python ≥ 3.8
- [JAX](https://jax.readthedocs.io/) + jaxlib
- [NumPyro](https://num.pyro.ai/)
- NumPy
- networkx
- funsor

---

## Related packages

| Package | Description |
|---|---|
| [`because`](https://github.com/because-pkg/because) | Main R package — JAGS, Nimble, and NumPyro engines |
| [`because.phybase`](https://github.com/because-pkg/because.phybase) | Phylogenetic extension for `because` |
| `because_py` | This package — Python/NumPyro backend |
