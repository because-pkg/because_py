# Changelog

All notable changes to `because_py` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-11-11

### Performance & Memory Optimizations
* **Batched, Memory-Capped WAIC for MultiPhylo Ensembles**:
  * Implemented batched log-likelihood evaluation (`batch_size = 50`) and systematic sample thinning (100 draws/tree across 30 trees = 3,000 ensemble draws) in `calculate_waic_ensemble()`.
  * Reduces peak memory usage from $> 24\text{ GB}$ to $< 250\text{ MB}$, eliminating vector memory limit crashes on comparative datasets with large covariance matrices.
  * Added robust exception handling with `warnings.warn` so that diagnostic calculation never discards completed MCMC sampling results.
* **Liability-Scale Pagel's Lambda for Binomial Phylogenetic Models**:
  * Automatically evaluates Pagel's $\lambda$ on the continuous latent liability scale for binary traits with phylogenetic covariance structures, incorporating the $\pi^2/3 \approx 3.2899$ disturbance variance.
* **Distributional Parsimony Rule for d-Separation**:
  * In `CausalGraph`, automatically prioritizes continuous variables as responses in auxiliary d-separation tests against discrete/categorical variables, avoiding slow link functions while preserving exact d-separation topology.
* **Ensemble Postprocessing and Parameter Constraints in `jax.pmap`**:
  * Applied `postprocess_fn` to pmapped chain states, ensuring sampled parameters (such as positive variance components and Cholesky correlations) are mapped to their constrained support and deterministic sites are evaluated.
* **Ensemble D-Separation Diagnostics & Subsampling Safeguards**:
  * Added `Rhat_max` and ensemble-summed ESS to d-separation summary tables for multiPhylo models.
  * Disabled `dsep_max_obs` subsampling when structured covariance matrices (`cor_matrices`) are present to ensure complete tree topology matching.
* **Lazy JAX/NumPyro Imports**:
  * Deferred imports in model compiler to prevent premature JAX threadpool initialization before `XLA_FLAGS` are set.

### Major Architecture Refactoring
* **Modular Model Compiler**:
  * Factored the monolithic `compiler.py` into dedicated, single-responsibility components under `because.builder.components`:
    * `variances.py`: Universal variance component partitioning (Dirichlet/Beta) and MAG induced residual correlations (LKJ Cholesky).
    * `multiscale.py`: Deterministic node evaluation (`I(...)`, interactions) and cross-scale hierarchical bridging (coarse-to-fine broadcasting, fine-to-coarse resolution locking via `segment_sum`).
    * `random_effects.py`: Spectral Gaussian marginalization in phylogenetic eigenbases and structural random intercept/slope plates with Cholesky factors.
    * `likelihoods.py`: Distribution mapping for all 8 supported families (Gaussian, Poisson, Bernoulli/Binomial, NegativeBinomial2, ZIP, ZINB, Categorical, OrderedLogistic) and missing data (NaN) marginalization/imputation.
  * Decoupled self-contained Python source string generation into `because.builder.codegen`.
* **Decoupled Engine & Public API**:
  * Slimmed `api.py` into a lightweight facade (~240 lines, down from 896), delegating to dedicated submodules:
    * `because.engine.env`: JAX threadpool limits (`OMP_NUM_THREADS`, `MKL`, `XLA_FLAGS`), cgroup/SLURM/PBS/LSF core auto-detection, and lazy JAX/NumPyro initialization.
    * `because.engine.runner`: MCMC kernel configuration (NUTS, DiscreteHMCGibbs for discrete sites) and parallel/sequential execution.
    * `because.engine.ensemble`: MultiPhylo wave ensemble runner leveraging `jax.pmap` across available devices.
    * `because.diagnostics.waic`: WAIC calculation and pointwise log-likelihood accumulation across standard and ensemble models.
    * `because.dsep.runner`: D-Separation and M-Separation testing loop with automated subsampling and convergence diagnostics.
    * `because.summary`: Tabular posterior and d-separation summaries (`summary()`, `dsep_summary()`).
* **Standardized Pytest Suite & Hygiene**:
  * Eliminated 24 root-level test scripts and obsolete debug scratch files.
  * Established a formal test suite under `tests/` with `pytest.ini` and `tests/conftest.py`:
    * `tests/unit/`: Fast unit tests (< 1.5s) covering parser, DAG/MAG topology, and compiler codegen without MCMC overhead.
    * `tests/integration/`: End-to-end NUTS MCMC tests covering DAG/MAG d-separation, deterministic nodes, correlated errors, multiscale resolution locking, WAIC, and summaries.
* **R / Reticulate Compatibility**:
  * Full backward compatibility for R's `reticulate` caller, preserving function signatures, return structures, and internal helper symbols. Exposes `__version__ = "1.0.0"`.

---

## [0.3.0] - 2026-11-08

### Added
* **MultiPhylo Wave Ensemble Sampling**: Parallel MCMC sampling across phylogenetic tree ensembles using `jax.pmap` waves.
* **Tree Index Parameter Monitoring**: Injected 1-based `K_tree_<struct>` parameter into posterior samples for trace diagnostics.
* **ROPE Sample Preservation**: Preserved raw posterior sample draws in `dsep_results` for region of practical equivalence (ROPE) analysis in R.
* **Level Disambiguation**: Resolved variable and structure level disambiguation when hierarchical scales share equal sample sizes.

### Fixed
* Omitted redundant `sigma_tot` and `sigma_struct` for latent indicator variables.
* Excluded latent indicator variables from Pagel's $\lambda$ calculation.

---

## [0.2.10] - 2026-06-25

### Improvements
* Refined latent indicator exclusions in induced correlation sets.
* Enhanced MCMC parameter alias mapping and kwargs forwarding.

---

## [0.2.9] - 2026-10-29

### Added
* Mean-centered group random effects in NumPyro builder to prevent confounding with global intercepts.
* Support for parameter aliases, kwargs forwarding, and `fit_numpyro_model` alias.
* Added `max_treedepth` and `adapt_delta` control arguments to `fit()`.

### Fixed
* Fixed NetworkX cycle and string iteration bugs when latents are passed as strings.
* Graceful handling of evaluation errors for unused computed variables in hierarchical DAGs.

---

## [0.2.8] - 2026-06-20

### Added
* Universal Variance Component Partitioning: Dirichlet / Beta prior partitioning for variables with multiple competing error components (residual, spatial GP, induced correlation).
* Non-Centered Parameterization (NCP) for multivariate latent errors in MAG models.
* Lazy JAX/NumPyro initialization with automatic CPU core detection (SLURM, PBS, cgroups) and thread limits.

---

## [0.2.0] - 2026-06-08

### Added
* **MAG Induced Correlations**: Support for unmeasured latent confounders using 2×2 LKJ Cholesky correlation priors.
* **DiscreteHMCGibbs & MultiPhylo**: Analytical spectral Gaussian marginalization for multiPhylo trees in `dcat` mode.
* **WAIC Diagnostics**: Rao-Blackwellized WAIC calculations for models with discrete tree mixtures.
* **Summary APIs**: Added `summary()` and `dsep_summary()` returning formatted pandas DataFrames.

---

## [0.1.0] - 2026-06-03

### Initial Release
* Formula-driven SEM compilation to NumPyro models.
* Topological DAG sorting and minimal d-separation basis set generation.
* NUTS sampling via JAX.
* Support for Gaussian, Poisson, Binomial, and Negative Binomial likelihoods.
* Random intercepts and spatial/phylogenetic covariance matrix support via Cholesky decomposition.
