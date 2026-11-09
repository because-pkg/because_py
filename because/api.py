"""
Public API for because-py: high-level interface for model fitting, diagnostics,
and d-separation tests.
"""

import warnings
import numpy as np

from .builder import FormulaParser, CausalGraph, NumPyroBuilder
from .engine.env import (
    _detect_available_cores,
    _init_jax,
    get_jax_primitives,
)
from .engine.runner import run_mcmc
from .engine.ensemble import run_ensemble_chains, wrap_ensemble_model
from .diagnostics.waic import (
    calculate_waic,
    calculate_waic_internal,
    calculate_waic_ensemble,
    accumulate_waic_from_log_lik,
)
from .dsep.runner import run_dsep_tests
from .summary import summary, dsep_summary

__version__ = "1.0.0"

# Backwards compatibility aliases for private functions
_accumulate_waic_from_log_lik = accumulate_waic_from_log_lik
_calculate_waic_ensemble = calculate_waic_ensemble
_calculate_waic_internal = calculate_waic_internal


_JAX_EXPORTS = {
    "jax", "jnp", "numpyro", "dist", "diag",
    "MCMC", "NUTS", "Predictive", "DiscreteHMCGibbs"
}

def __getattr__(name):
    """
    Expose JAX and NumPyro module attributes lazily on because.api
    for backwards compatibility with existing R/reticulate callers.
    """
    if name in _JAX_EXPORTS:
        return get_jax_primitives()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_dsep_equations(equations, latent=None):
    """
    Helper to return implied d-sep claims without fitting the model.

    :param equations: List of formula strings.
    :param latent: Optional list of unmeasured/latent variable names.
    :return: List of claim dictionaries.
    """
    if isinstance(latent, str):
        latent = [latent]
    test_parser = FormulaParser(equations)
    test_parsed = test_parser.parse()
    graph = CausalGraph(test_parsed)
    graph.build()
    return graph.generate_dsep_equations(latent=latent)


def fit(
    equations,
    data,
    family=None,
    latent=None,
    cor_matrices=None,
    induced_correlations=None,
    dsep=False,
    dsep_only=False,
    calculate_waic=False,
    num_samples=1000,
    num_warmup=500,
    num_chains=1,
    thinning=1,
    n_cores=1,
    seed=0,
    dsep_max_obs=2000,
    quiet=False,
    dsep_equations_to_run=None,
    adapt_delta=0.95,
    max_treedepth=10,
    fix_latent="loading",
    prior_scale_fixed=None,
    levels=None,
    **kwargs
):
    """
    Fit a causal graphical model using NumPyro (JAX backend).

    :param equations: List of R-style formulas as strings (e.g., ["y ~ x", "z ~ y"])
    :param data: Dictionary mapping variable names to 1D numpy arrays
    :param family: Optional dictionary mapping response variable names to families
    :param latent: Optional list of unmeasured/latent variable names
    :param cor_matrices: Optional dictionary mapping group names to covariance matrices
    :param induced_correlations: Optional list of variable pairs with induced correlations
    :param dsep: Boolean, if True, tests implied conditional independencies
    :param dsep_only: Boolean, if True, only runs d-sep tests without fitting the main model
    :param calculate_waic: Boolean, if True, computes WAIC for the model
    :param num_samples: Number of post-warmup MCMC samples per chain
    :param num_warmup: Number of warmup/burn-in MCMC samples per chain
    :param num_chains: Number of MCMC chains to sample
    :param thinning: Thinning interval for MCMC samples
    :param n_cores: Number of parallel CPU cores to use
    :param seed: Random seed for PRNG
    :param dsep_max_obs: Maximum sample size for d-sep testing (subsamples if data is larger)
    :param quiet: Boolean, if True, suppresses MCMC progress bars and messages
    :param dsep_equations_to_run: Optional subset of d-sep test claims to evaluate
    :param adapt_delta: NUTS target acceptance probability (default 0.95)
    :param max_treedepth: NUTS maximum tree depth (default 10)
    :param fix_latent: Method to fix scale of latent variables ("loading" or "variance")
    :param prior_scale_fixed: Optional fixed scale for regression coefficients
    :param levels: Optional dictionary mapping level names to variable lists
    :return: Dictionary containing samples, mcmc object, waic, dsep_results, and model_code
    """
    _init_jax(n_cores)
    prim = get_jax_primitives()
    jax = prim["jax"]
    jnp = prim["jnp"]

    # Parse equations and build causal graph
    parser = FormulaParser(equations)
    parsed = parser.parse()
    deterministic_terms = parser.deterministic_terms

    # Auto-detect latents: variables in equations but not in data or deterministic terms
    if isinstance(latent, str):
        latent = [latent]

    if latent is None:
        vars_in_eqs = set()
        for eq in parsed:
            if eq["response"]:
                vars_in_eqs.add(eq["response"])
            vars_in_eqs.update(eq["fixed"])
        latent = list(vars_in_eqs - set(data.keys()) - set(deterministic_terms.keys()))

    graph = CausalGraph(parsed, deterministic_terms=deterministic_terms)
    graph.build()

    if not quiet:
        print("Graph information:")
        print(f"   Observed stochastic nodes: {len(data)}")
        unobserved = len(latent) if latent else 0
        print(f"   Unobserved stochastic nodes: {unobserved}")
        print("\nInitializing model")

    compiler = NumPyroBuilder(
        graph,
        family_dict=family,
        deterministic_terms=deterministic_terms,
        cor_matrices=cor_matrices,
        fix_latent=fix_latent,
        induced_correlations=induced_correlations,
        prior_scale_fixed=prior_scale_fixed,
        levels=levels,
    )
    model_func = compiler.generate_model_function(data_for_compilation=data)

    try:
        model_code_str = compiler.generate_model_code_string()
    except Exception:
        model_code_str = None

    # Convert data to JAX arrays
    jax_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            jax_data[k] = {vk: jnp.array(vv) for vk, vv in v.items()}
        else:
            jax_data[k] = jnp.array(v)

    results = {}

    ensemble_structs = [
        v for v in (cor_matrices or {}).values()
        if isinstance(v, dict) and v.get("type") == "multiPhylo_ensemble"
    ]
    has_multiPhylo_ensemble = len(ensemble_structs) > 0

    if not dsep_only:
        rng_key = jax.random.PRNGKey(seed)

        if has_multiPhylo_ensemble:
            n_trees = int(ensemble_structs[0].get("n_trees", jax_data.get("Ntree", 10)))
            ens_n_chains = int(
                ensemble_structs[0].get(
                    "n_chains",
                    ensemble_structs[0].get("chains_per_tree", num_chains if num_chains > 1 else 2)
                )
            )
            ens_model_func = wrap_ensemble_model(model_func, cor_matrices, jax_data)
            rng_key, mcmc, tree_samples_list, numpy_samples = run_ensemble_chains(
                rng_key,
                ens_model_func,
                jax_data,
                n_trees,
                ens_n_chains,
                n_cores,
                num_warmup=num_warmup,
                num_samples=num_samples,
                thinning=thinning,
                adapt_delta=adapt_delta,
                max_treedepth=max_treedepth,
                cor_matrices=cor_matrices,
                show_progress=not quiet,
            )
        else:
            rng_key, mcmc, numpy_samples = run_mcmc(
                model_func,
                jax_data,
                rng_key,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                thinning=thinning,
                n_cores=n_cores,
                adapt_delta=adapt_delta,
                max_treedepth=max_treedepth,
                cor_matrices=cor_matrices,
                quiet=quiet,
            )

        results["samples"] = numpy_samples
        results["mcmc"] = mcmc
        results["parameter_map"] = None
        results["model_string"] = "NumPyro Causal Hierarchical Model"
        results["model_code"] = model_code_str

        if calculate_waic:
            if not quiet:
                print("Calculating WAIC...")

            has_multiPhylo = cor_matrices and any(
                isinstance(v, dict) and v.get("type") == "multiPhylo" for v in cor_matrices.values()
            )
            if has_multiPhylo_ensemble:
                results["waic"] = calculate_waic_ensemble(ens_model_func, tree_samples_list, jax_data)
            elif has_multiPhylo:
                waic_model_func = compiler.generate_model_function(data_for_compilation=data, force_plate_obs=True)
                results["waic"] = calculate_waic_internal(waic_model_func, mcmc, jax_data)
            else:
                results["waic"] = calculate_waic_internal(model_func, mcmc, jax_data)
    else:
        rng_key = jax.random.PRNGKey(seed)

    if not dsep:
        return results

    # D-Separation / M-Separation testing
    results["dsep_results"] = run_dsep_tests(
        graph=graph,
        data=data,
        latent=latent,
        dsep_equations_to_run=dsep_equations_to_run,
        deterministic_terms=deterministic_terms,
        family=family,
        cor_matrices=cor_matrices,
        fix_latent=fix_latent,
        prior_scale_fixed=prior_scale_fixed,
        levels=levels,
        dsep_max_obs=dsep_max_obs,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        thinning=thinning,
        n_cores=n_cores,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        rng_key=rng_key,
        quiet=quiet,
        has_multiPhylo_ensemble=has_multiPhylo_ensemble,
        ensemble_structs=ensemble_structs,
    )

    return results


# Alias for backwards compatibility
fit_numpyro_model = fit
