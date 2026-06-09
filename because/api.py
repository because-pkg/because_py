import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, DiscreteHMCGibbs
import numpyro.diagnostics as diag
from .builder import FormulaParser, CausalGraph, NumPyroBuilder

def get_dsep_equations(equations, latent=None):
    """Helper to return implied d-sep claims without fitting the model."""
    test_parser = FormulaParser(equations)
    test_parsed = test_parser.parse()
    graph = CausalGraph(test_parsed)
    graph.build()
    return graph.generate_dsep_equations(latent=latent)

def fit(equations, data, family=None, latent=None, cor_matrices=None, dsep=False, dsep_only=False, calculate_waic=False, num_samples=1000, num_warmup=500, num_chains=1, thinning=1, n_cores=1, seed=0, dsep_max_obs=2000, quiet=False, dsep_equations_to_run=None, adapt_delta=0.95, fix_latent="loading"):
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
    
    # Auto-detect latents: variables in equations but not in data
    if hasattr(jax, "local_device_count") and jax.local_device_count() < n_cores:
        try:
            numpyro.set_host_device_count(n_cores)
        except Exception as e:
            if not quiet:
                print(f"Warning: Failed to set numpyro host device count to {n_cores}: {e}")
                
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
        
    compiler = NumPyroBuilder(graph, family_dict=family, deterministic_terms=deterministic_terms, cor_matrices=cor_matrices, fix_latent=fix_latent)
    model_func = compiler.generate_model_function(data_for_compilation=data)
    
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
        if cor_matrices and any(v.get("type") == "multiPhylo" for v in cor_matrices.values()):
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
        mcmc = MCMC(kernel_base, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, thinning=thinning, progress_bar=False)
        
        mcmc.run(subkey, **jax_data)
        
        samples = mcmc.get_samples(group_by_chain=True)
        
        # Format samples dictionary (convert Jax arrays to pure numpy arrays for easy reticulate conversion)
        numpy_samples = {k: np.asarray(v) for k, v in samples.items()}
        
        # Add basic convergence diagnostics if possible (num_chains > 1)
        # We can implement basic rhat/ess here if requested, but typically standard packages do this better.
        
        results["samples"] = numpy_samples
        results["parameter_map"] = None
        results["model_string"] = "NumPyro Causal Hierarchical Model" # To pass a generic string
        
        if calculate_waic:
            if not quiet:
                print("Calculating WAIC...")
            
            # For multiPhylo, the exact Gibbs kernel requires `.to_event(1)` on the observation sites,
            # which collapses the log-likelihood to a scalar per chain/sample, preventing pointwise WAIC calculation.
            # We must rebuild the model using standard `numpyro.plate` specifically for WAIC evaluation.
            has_multiPhylo = cor_matrices and any(v.get("type") == "multiPhylo" for v in cor_matrices.values())
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
    dsep_equations = graph.generate_dsep_equations(latent=latent)
    
    # Filter if incremental caching is used from R
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
        if cor_matrices and any(v.get("type") == "multiPhylo" for v in cor_matrices.values()):
            n_trees = int(jax_dsep_data.get("Ntree", 10))
            kernel_base = DiscreteHMCGibbs(kernel_base)
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

