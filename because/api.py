import warnings
import os
import sys
import numpy as np
from .builder import FormulaParser, CausalGraph, NumPyroBuilder

# ---------------------------------------------------------------------------
# Lazy JAX / NumPyro initialisation
# ---------------------------------------------------------------------------
# JAX creates its C++ threadpools (tf_XLAEigen, tf_foreach, llvm-worker) the
# moment "import jax" executes.  The pool sizes are read from
# std::thread::hardware_concurrency(), which returns the *physical* CPU count
# of the host node — NOT the cgroup/container quota.  We therefore must set
# all thread-limiting env vars *before* any jax import statement runs.
#
# The sentinel _jax_initialized ensures we only call _init_jax() once per
# Python process even if fit() is called many times.
# ---------------------------------------------------------------------------
_jax_initialized: bool = False

# Module-level placeholders filled by _init_jax()
jax               = None
jnp               = None
numpyro           = None
dist              = None
MCMC              = None
NUTS              = None
Predictive        = None
DiscreteHMCGibbs  = None
diag              = None


def _init_jax(n_cores: int = 1) -> None:
    """
    Lazily initialise JAX and NumPyro with thread limits already set.

    Must be called at the top of every public function that uses JAX *before*
    any JAX symbol is referenced.  Subsequent calls after the first are
    no-ops (Python's import system caches modules, so the threadpools are
    created only once).

    Parameters
    ----------
    n_cores : int
        Number of parallel chains / devices requested by the caller.
        Used to size the XLA device-count flag and numpyro host devices.
    """
    global _jax_initialized
    global jax, jnp, numpyro, dist, MCMC, NUTS, Predictive, DiscreteHMCGibbs, diag

    if _jax_initialized:
        return

    # ------------------------------------------------------------------ #
    # Detect whether JAX was already imported by external code (e.g. the  #
    # user ran `import jax` before calling because.fit()).  In that case  #
    # the threadpools are already sized from hardware_concurrency() and   #
    # our env-var limits are too late for the C++ layer.  Warn clearly.   #
    # ------------------------------------------------------------------ #
    jax_already_loaded = "jax" in sys.modules

    if jax_already_loaded:
        warnings.warn(
            "JAX was imported before because.fit() was called.  "
            "Thread-pool limits (OMP_NUM_THREADS, XLA thread counts, etc.) "
            "cannot be applied retroactively to the JAX C++ backend.  "
            "To control thread usage on a cluster, set thread-limit env vars "
            "and ensure JAX is first imported inside because.fit() — i.e. do "
            "NOT call `import jax` at the top of your script before using because.",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        # -------------------------------------------------------------- #
        # Set ALL thread-limiting env vars before any jax import.        #
        # os.environ.setdefault() respects values already set by the     #
        # user (e.g. via Sys.setenv() in R or the Docker ENV block).     #
        # -------------------------------------------------------------- #
        _thread_env = {
            "OMP_NUM_THREADS":               "1",
            "OPENBLAS_NUM_THREADS":          "1",
            "GOTO_NUM_THREADS":              "1",
            "MKL_NUM_THREADS":               "1",
            "MKL_DOMAIN_NUM_THREADS":        "1",
            "NUMEXPR_NUM_THREADS":           "1",
            "LLVM_NUM_THREADS":              "1",   # llvm-worker-N threads
            "TF_NUM_INTEROP_THREADS":        str(n_cores),
            "TF_NUM_INTRAOP_THREADS":        str(n_cores),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        for k, v in _thread_env.items():
            os.environ.setdefault(k, v)

        # XLA_FLAGS: merge with any flags already present
        existing_xla = os.environ.get("XLA_FLAGS", "")
        new_flags = []
        if "--xla_force_host_platform_device_count" not in existing_xla:
            new_flags.append(f"--xla_force_host_platform_device_count={n_cores}")
        if "--xla_cpu_multi_thread_eigen" not in existing_xla:
            new_flags.append("--xla_cpu_multi_thread_eigen=false")
        if "intra_op_parallelism_threads" not in existing_xla:
            new_flags.append(f"intra_op_parallelism_threads={n_cores}")
        if "inter_op_parallelism_threads" not in existing_xla:
            new_flags.append(f"inter_op_parallelism_threads={n_cores}")
        if new_flags:
            os.environ["XLA_FLAGS"] = (existing_xla + " " + " ".join(new_flags)).strip()

    # ------------------------------------------------------------------ #
    # Now it is safe to import JAX and NumPyro.                           #
    # If jax_already_loaded is True the imports below are instant cache   #
    # hits — no new threadpools are created.                              #
    # ------------------------------------------------------------------ #
    import jax            as _jax
    import jax.numpy      as _jnp
    import numpyro        as _numpyro
    import numpyro.distributions as _dist
    import numpyro.diagnostics   as _diag
    from numpyro.infer import (
        MCMC              as _MCMC,
        NUTS              as _NUTS,
        Predictive        as _Predictive,
        DiscreteHMCGibbs  as _DiscreteHMCGibbs,
    )

    # Expose as module-level globals so the rest of api.py works unchanged
    jax              = _jax
    jnp              = _jnp
    numpyro          = _numpyro
    dist             = _dist
    diag             = _diag
    MCMC             = _MCMC
    NUTS             = _NUTS
    Predictive       = _Predictive
    DiscreteHMCGibbs = _DiscreteHMCGibbs

    # numpyro.set_host_device_count MUST be called before any JAX computation
    if n_cores > 1 and not jax_already_loaded:
        try:
            _numpyro.set_host_device_count(n_cores)
        except Exception as exc:
            warnings.warn(
                f"because: numpyro.set_host_device_count({n_cores}) failed: {exc}",
                RuntimeWarning,
            )

    _jax_initialized = True

def get_dsep_equations(equations, latent=None):
    """Helper to return implied d-sep claims without fitting the model."""
    test_parser = FormulaParser(equations)
    test_parsed = test_parser.parse()
    graph = CausalGraph(test_parsed)
    graph.build()
    return graph.generate_dsep_equations(latent=latent)

def fit(equations, data, family=None, latent=None, cor_matrices=None, induced_correlations=None, dsep=False, dsep_only=False, calculate_waic=False, num_samples=1000, num_warmup=500, num_chains=1, thinning=1, n_cores=1, seed=0, dsep_max_obs=2000, quiet=False, dsep_equations_to_run=None, adapt_delta=0.95, fix_latent="loading"):
    """
    High-level API for because-py. Fits a causal hierarchical model using NumPyro.
    
    :param equations: List of formula strings (e.g., ["y ~ x + z + (1|group)"]).
    :param data: Dictionary of numpy arrays for the data.
    :param family: Dictionary specifying families (e.g. {"y": "poisson"}). Default is "gaussian".
    :param latent: List of unmeasured/latent variables (e.g. ["L1"]). If None, auto-detected from equations.
    :param dsep: Boolean, if True, computes and tests the basis set of implied conditional independencies.
    :param num_samples: Number of post-warmup MCMC samples.
    :param num_warmup: Number of warmup MCMC samples.
    :param num_chains: Number of parallel MCMC chains.
    :param thinning: Thinning interval for MCMC samples.
    :param seed: Random seed for JAX PRNGKey.
    :param dsep_max_obs: Maximum number of observations to use for d-sep tests (random subsampling) for speed.
    :param quiet: Boolean, if True, suppresses MCMC progress bars.
    
    :return: A dictionary containing the fitted MCMC object and optional dsep results.
    """
    if isinstance(equations, str):
        equations = [equations]
        
    if not isinstance(family, dict):
        family = {}
        
    if not quiet:
        print("Compiling NumPyro model graph")
        print("   Resolving causal relationships")
        print("   Allocating nodes")
        
    parser = FormulaParser(equations)
    parsed = parser.parse()
    deterministic_terms = parser.deterministic_terms
    
    # Initialise JAX lazily with thread limits applied before first import.
    # This is the earliest safe point: after argument validation, before any
    # JAX symbol is referenced.
    _init_jax(n_cores)

    # Auto-detect latents: variables in equations but not in data
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
        
    compiler = NumPyroBuilder(graph, family_dict=family, deterministic_terms=deterministic_terms, cor_matrices=cor_matrices, fix_latent=fix_latent, induced_correlations=induced_correlations)
    model_func = compiler.generate_model_function(data_for_compilation=data)
    # Generate human-readable Python source for inspection / because_continue()
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
    
    if not dsep_only:
        rng_key = jax.random.PRNGKey(seed)
        rng_key, subkey = jax.random.split(rng_key)
        
        kernel_base = NUTS(model_func, target_accept_prob=adapt_delta, init_strategy=numpyro.infer.init_to_sample())
        if cor_matrices and any(isinstance(v, dict) and v.get("type") == "multiPhylo" for v in cor_matrices.values()):
            n_trees = int(jax_data.get("Ntree", 10))
            def rw_fn(rng_key, discrete_sites):
                new_sites = {}
                for k, v in discrete_sites.items():
                    if k.startswith("K_tree"):
                        new_sites[k] = jax.random.randint(rng_key, shape=jnp.shape(v), minval=0, maxval=n_trees)
                    else:
                        new_sites[k] = v
                return new_sites
            kernel_base = DiscreteHMCGibbs(kernel_base)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="There are not enough devices")
            mcmc = MCMC(kernel_base, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, thinning=thinning, progress_bar=False)
        
        mcmc.run(subkey, **jax_data)
        
        samples = mcmc.get_samples(group_by_chain=True)
        
        # Format samples dictionary (convert Jax arrays to pure numpy arrays for easy reticulate conversion)
        numpy_samples = {k: np.asarray(v) for k, v in samples.items()}
        
        # Add basic convergence diagnostics if possible (num_chains > 1)
        # We can implement basic rhat/ess here if requested, but typically standard packages do this better.
        
        results["samples"] = numpy_samples
        results["mcmc"] = mcmc
        results["parameter_map"] = None
        results["model_string"] = "NumPyro Causal Hierarchical Model"  # legacy placeholder
        results["model_code"] = model_code_str  # executable Python source string
        
        if calculate_waic:
            if not quiet:
                print("Calculating WAIC...")
            
            # For multiPhylo, the exact Gibbs kernel requires `.to_event(1)` on the observation sites,
            # which collapses the log-likelihood to a scalar per chain/sample, preventing pointwise WAIC calculation.
            # We must rebuild the model using standard `numpyro.plate` specifically for WAIC evaluation.
            has_multiPhylo = cor_matrices and any(isinstance(v, dict) and v.get("type") == "multiPhylo" for v in cor_matrices.values())
            if has_multiPhylo:
                waic_model_func = compiler.generate_model_function(data_for_compilation=data, force_plate_obs=True)
            else:
                waic_model_func = model_func
                
            results["waic"] = _calculate_waic_internal(waic_model_func, mcmc, jax_data)
            
    else:
        # We just need a dummy rng key
        rng_key = jax.random.PRNGKey(seed)
        
    if not dsep:
        return results
        
    # D-Separation Testing (and M-Separation)
    # ----------------------------------------------------
    # If dsep_equations_to_run is provided as a list of dicts, it comes pre-computed
    # from the R package (which uses dagitty's optimal MAG logic).
    if dsep_equations_to_run is not None and len(dsep_equations_to_run) > 0 and isinstance(dsep_equations_to_run[0], dict):
        dsep_equations = dsep_equations_to_run
    else:
        # Otherwise, fall back to Python's internal graph logic
        dsep_equations = graph.generate_dsep_equations(latent=latent)
        
        # Filter if incremental caching is used from R (old string-based format)
        if dsep_equations_to_run is not None:
            norm_to_run = [" ".join(e.split()) for e in dsep_equations_to_run]
            dsep_equations = [c for c in dsep_equations if " ".join(c["equation_string"].split()) in norm_to_run]

    if not dsep_equations:
        if not quiet:
            print("Graph is fully connected or no testable claims exist. No implied conditional independencies to test.")
        return results
        
    if not quiet:
        msg = f"\nRunning M-Separation tests (MAG)" if latent else f"\nRunning D-Separation tests"
        print(f"{msg} ({len(dsep_equations)} implied claims)...")
        
    dsep_results = []
    
    # Optional subsampling for d-sep speed
    # We use data.values() iter to find length, but we must ignore latents (not in data)
    N_total = len(next(iter(data.values())))
    dsep_data = data
    if N_total > dsep_max_obs:
        if not quiet:
            print(f"Subsampling data to {dsep_max_obs} rows for fast testing.")
        idx = np.random.choice(N_total, size=dsep_max_obs, replace=False)
        dsep_data = {k: v[idx] for k, v in data.items()}
        
    jax_dsep_data = {}
    for k, v in dsep_data.items():
        if isinstance(v, dict):
            jax_dsep_data[k] = {vk: jnp.array(vv) for vk, vv in v.items()}
        else:
            jax_dsep_data[k] = jnp.array(v)
    
    for i, claim in enumerate(dsep_equations):
        eq_str = claim["equation_string"]
        test_var = claim["test_node"]
        resp = claim["response"]
        claim_type = claim["type"]
        
        if not quiet:
            test_desc = "(Induced Correlation)" if claim_type == "correlation" else "(Conditional Independence)"
            print(f"\n[Test {i+1} / {len(dsep_equations)}]  {eq_str}  {test_desc}")
            
        compile_eq_str = eq_str
        if cor_matrices:
            for s_name in cor_matrices.keys():
                if f"(1|{s_name})" not in compile_eq_str.replace(" ", ""):
                    compile_eq_str += f" + (1|{s_name})"
            
        test_parser = FormulaParser([compile_eq_str])
        test_parsed = test_parser.parse()
        test_graph = CausalGraph(test_parsed, deterministic_terms=deterministic_terms)
        test_graph.build()
        
        test_compiler = NumPyroBuilder(test_graph, family_dict=family, deterministic_terms=deterministic_terms, cor_matrices=cor_matrices, fix_latent=fix_latent)
        test_model_func = test_compiler.generate_model_function(data_for_compilation=dsep_data)
        
        rng_key, subkey = jax.random.split(rng_key)
        
        # We run MCMC for the dsep test using the same parameters as the main model
        # To get valid Rhat, we enforce at least 2 chains if the user requested less than 2
        dsep_chains = max(2, num_chains)
        kernel_base = NUTS(test_model_func, target_accept_prob=adapt_delta, init_strategy=numpyro.infer.init_to_sample())
        if cor_matrices and any(isinstance(v, dict) and v.get("type") == "multiPhylo" for v in cor_matrices.values()):
            n_trees = int(jax_dsep_data.get("Ntree", 10))
            kernel_base = DiscreteHMCGibbs(kernel_base)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*There are not enough devices.*")
            test_mcmc = MCMC(kernel_base, num_warmup=num_warmup, num_samples=num_samples, num_chains=dsep_chains, thinning=thinning, progress_bar=False)
            test_mcmc.run(subkey, **jax_dsep_data)
        
        # We need grouped samples for Rhat and neff calculations
        samples_grouped = test_mcmc.get_samples(group_by_chain=True)
        # We also want flat samples for easy percentiles
        samples_flat = {k: np.asarray(v).flatten() for k, v in samples_grouped.items()}
        
        beta_name = f"beta_{resp}_{test_var}"
        
        if beta_name in samples_flat:
            posteriors = samples_flat[beta_name]
            mean_val = float(np.mean(posteriors))
            # 95% HPDI (using percentile for simplicity in this base implementation)
            ci_lower = float(np.percentile(posteriors, 2.5))
            ci_upper = float(np.percentile(posteriors, 97.5))
            
            # Compute Rhat and neff
            chain_samples = np.asarray(samples_grouped[beta_name])
            rhat = float(diag.gelman_rubin(chain_samples))
            n_eff = float(diag.effective_sample_size(chain_samples))
            
            # If the 95% CI includes 0, it is conditionally independent
            is_independent = (ci_lower < 0 < ci_upper)
            
            if not quiet:
                print(f" -> {beta_name} = {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
                
            dsep_results.append({
                "claim": f"{resp} _|_ {test_var} | {', '.join(claim['conditioning_set'])}",
                "equation": eq_str,
                "coefficient": beta_name,
                "mean": mean_val,
                "ci_2.5": ci_lower,
                "ci_97.5": ci_upper,
                "is_independent": is_independent,
                "rhat": rhat,
                "n_eff": n_eff
            })
            
    results["dsep_results"] = dsep_results
    return results

def _calculate_waic_internal(model_func, mcmc, jax_data):
    """
    Calculates WAIC (Widely Applicable Information Criterion) with standard errors
    using pointwise log-likelihoods.

    Handles multi-scale hierarchical models where different response variables
    may have different numbers of observations (e.g. Abundance N=4500, Body_Mass_s N=50).
    Each variable's pointwise log-likelihoods are kept separate and only combined
    at the total-WAIC level, so shapes never need to broadcast.

    For multiPhylo models, uses Rao-Blackwellization: for each MCMC sample of
    continuous parameters, the log-likelihood is averaged over ALL trees (with
    equal weight = 1/T). This removes the artificial variance in p_waic that would
    result from K_tree jumping between trees across samples.
    """
    from numpyro.infer import log_likelihood
    import numpy as np
    from scipy.special import logsumexp
    import jax.numpy as jnp

    flat_samples = mcmc.get_samples()

    # Detect multiPhylo: K_tree variables present in samples
    k_tree_keys = [k for k in flat_samples.keys() if k.startswith("K_tree")]
    has_multi_phylo = len(k_tree_keys) > 0

    if has_multi_phylo:
        # Determine number of trees from jax_data
        n_trees = int(jax_data.get("Ntree", max(int(v.max()) + 1
                                                 for v in (np.array(flat_samples[k])
                                                           for k in k_tree_keys))))

        # Rao-Blackwellized WAIC:
        # For each continuous parameter sample s, compute:
        #   LL_RB[i, s] = log(1/T * sum_k exp(log p(y_i | theta_s, z_s, K=k)))
        # This marginalizes the discrete tree index with uniform weights,
        # removing the artificial variance from K_tree jumping across samples.
        ll_per_tree = []  # list of dicts: {var: array(n_samples, n_obs)}
        for k in range(n_trees):
            fixed_samples = {
                key: (jnp.full_like(val, k) if key in k_tree_keys else val)
                for key, val in flat_samples.items()
            }
            ll_k = log_likelihood(model_func, fixed_samples, **jax_data)
            ll_per_tree.append({var: np.array(v) for var, v in ll_k.items()})

        # Stack and marginalize: log_mean_exp over trees (axis=0)
        log_lik_dict = {}
        for var_name in ll_per_tree[0].keys():
            # shape: (n_trees, n_samples, n_obs)
            ll_stack = np.stack([ll_per_tree[k][var_name] for k in range(n_trees)], axis=0)
            # Marginalize: log(1/T * sum_k exp(ll)): shape (n_samples, n_obs)
            log_lik_dict[var_name] = logsumexp(ll_stack, axis=0) - np.log(n_trees)
    else:
        # Standard (non-multiPhylo) WAIC: use all samples directly
        log_lik_dict = log_likelihood(model_func, flat_samples, **jax_data)

    if not log_lik_dict:
        return None

    # Accumulate pointwise WAIC components per variable, then sum across variables.
    # Each ll has shape (n_samples, n_obs_for_this_variable).
    # We keep them separate to avoid shape-mismatch in multi-scale models.
    total_elpd_waic = 0.0
    total_p_waic    = 0.0
    total_waic      = 0.0
    total_n_obs     = 0
    se_sq_elpd      = 0.0
    se_sq_p         = 0.0
    se_sq_waic      = 0.0

    n_samples = None
    for k, ll in log_lik_dict.items():
        ll_np = np.array(ll)          # shape: (n_samples, n_obs_k)
        n_samples, n_obs_k = ll_np.shape

        # lpd per observation
        lpd_i      = logsumexp(ll_np, axis=0) - np.log(n_samples)
        # effective params per observation
        p_waic_i   = np.var(ll_np, axis=0, ddof=1)
        # elpd and waic per observation
        elpd_i     = lpd_i - p_waic_i
        waic_i     = -2.0 * elpd_i

        total_elpd_waic += float(np.sum(elpd_i))
        total_p_waic    += float(np.sum(p_waic_i))
        total_waic      += float(np.sum(waic_i))

        # SE contributions (each variable contributes independently)
        se_sq_elpd  += n_obs_k * float(np.var(elpd_i,   ddof=1))
        se_sq_p     += n_obs_k * float(np.var(p_waic_i, ddof=1))
        se_sq_waic  += n_obs_k * float(np.var(waic_i,   ddof=1))
        total_n_obs += n_obs_k

    return {
        "elpd_waic": {"Estimate": total_elpd_waic,       "SE": float(np.sqrt(se_sq_elpd))},
        "p_waic":    {"Estimate": total_p_waic,           "SE": float(np.sqrt(se_sq_p))},
        "waic":      {"Estimate": total_waic,             "SE": float(np.sqrt(se_sq_waic))},
        "n_obs":     total_n_obs,
        "n_samples": n_samples,
    }


def summary(result, digits=3):
    """
    Print a posterior summary table for a because.fit() result.

    Computes mean, standard deviation, 2.5% and 97.5% quantiles, Rhat
    (Gelman-Rubin convergence diagnostic), and effective sample size (n_eff)
    for every sampled parameter.  Mirrors the output of ``summary(fit)`` in
    the R ``because`` package.

    :param result: Dictionary returned by :func:`fit`.
    :param digits: Number of decimal places to display (default 3).
    :return: A ``pandas.DataFrame`` with one row per parameter.

    Example::

        import because
        result = because.fit(equations=["y ~ x"], data={"x": x, "y": y})
        print(because.summary(result))
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for because.summary(). "
            "Install it with: pip install pandas"
        )

    samples = result.get("samples")
    if samples is None:
        raise ValueError(
            "result['samples'] not found. Make sure fit() was called without "
            "dsep_only=True."
        )

    rows = []
    for param, vals in samples.items():
        # vals shape: (n_chains, n_samples) when group_by_chain=True
        vals = np.asarray(vals)
        flat = vals.flatten()

        mean  = float(np.mean(flat))
        sd    = float(np.std(flat, ddof=1))
        q2_5  = float(np.percentile(flat, 2.5))
        q97_5 = float(np.percentile(flat, 97.5))

        # Rhat and n_eff require chain dimension; skip gracefully if only 1 chain
        if vals.ndim == 2 and vals.shape[0] > 1:
            rhat  = float(diag.gelman_rubin(vals))
            n_eff = float(diag.effective_sample_size(vals))
        else:
            rhat  = float("nan")
            n_eff = float(flat.size)

        rows.append({
            "parameter": param,
            "mean":    round(mean,  digits),
            "sd":      round(sd,    digits),
            "2.5%":    round(q2_5,  digits),
            "97.5%":   round(q97_5, digits),
            "Rhat":    round(rhat,  digits),
            "n_eff":   round(n_eff, 1),
        })

    df = pd.DataFrame(rows).set_index("parameter")
    return df


def dsep_summary(result, digits=3, verbose=False):
    """
    Print a summary table of d-separation test results from a because.fit() call.

    Each row represents one implied conditional independence claim from the
    minimum basis set.  A claim **passes** when the 95% posterior interval of
    the test coefficient includes zero (i.e. the two variables are conditionally
    independent given the conditioning set).

    :param result: Dictionary returned by :func:`fit` with ``dsep=True``.
    :param digits: Number of decimal places to display (default 3).
    :param verbose: If ``True``, prints a one-line pass/fail verdict above the
        table (default ``False``).
    :return: A ``pandas.DataFrame`` with one row per d-sep claim.

    Example::

        import because
        result = because.fit(
            equations   = ["Brain ~ Mass", "Lifespan ~ Brain"],
            data        = data_dict,
            dsep        = True,
            dsep_only   = True,
            num_samples = 1000
        )
        print(because.dsep_summary(result))                    # table only
        print(because.dsep_summary(result, verbose=True))     # verdict + table
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for because.dsep_summary(). "
            "Install it with: pip install pandas"
        )

    tests = result.get("dsep_results")
    if tests is None:
        raise ValueError(
            "result['dsep_results'] not found. "
            "Make sure fit() was called with dsep=True."
        )

    rows = []
    for t in tests:
        rows.append({
            "claim":       t["claim"],
            "coefficient": t["coefficient"],
            "mean":        round(t["mean"],    digits),
            "2.5%":        round(t["ci_2.5"],  digits),
            "97.5%":       round(t["ci_97.5"], digits),
            "result":      "PASS" if t["is_independent"] else "FAIL",
            "Rhat":        round(t["rhat"],    digits),
            "n_eff":       round(t["n_eff"],   1),
        })

    df = pd.DataFrame(rows).set_index("claim")

    if verbose:
        n_pass  = sum(1 for t in tests if t["is_independent"])
        n_total = len(tests)
        verdict = "Model is consistent with the data." if n_pass == n_total \
                  else "One or more independence claims FAILED — consider revising the DAG."
        print(f"D-separation tests: {n_pass}/{n_total} passed. {verdict}\n")

    return df
