import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS

from .builder import FormulaParser, CausalGraph, NumPyroBuilder

def fit(equations, data, family=None, latent=None, cor_matrices=None, dsep=False, dsep_only=False, calculate_waic=False, num_samples=1000, num_warmup=500, num_chains=1, thinning=1, seed=0, dsep_max_obs=2000, quiet=False):
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
    if not isinstance(family, dict):
        family = {}
        
    if not quiet:
        print("Parsing equations...")
    parser = FormulaParser(equations)
    parsed = parser.parse()
    deterministic_terms = parser.deterministic_terms
    
    # Auto-detect latents: variables in equations but not in data
    if latent is None:
        vars_in_eqs = set()
        for eq in parsed:
            if eq["response"]:
                vars_in_eqs.add(eq["response"])
            vars_in_eqs.update(eq["fixed"])
            
        latent = list(vars_in_eqs - set(data.keys()) - set(deterministic_terms.keys()))
        if latent and not quiet:
            print(f"Auto-detected latent variables: {latent}")
            
    graph = CausalGraph(parsed, deterministic_terms=deterministic_terms)
    graph.build()
    
    if not quiet:
        print(f"Graph topological order: {graph.get_topological_order()}")
        print("Compiling core NumPyro model...")
        
    compiler = NumPyroBuilder(graph, family_dict=family, deterministic_terms=deterministic_terms, cor_matrices=cor_matrices)
    model_func = compiler.generate_model_function(data_for_compilation=data)
    
    # Convert data to JAX arrays
    jax_data = {k: jnp.array(v) for k, v in data.items()}
    
    results = {}
    
    if not dsep_only:
        if not quiet:
            print(f"Compiling model via NumPyro...")
            
        rng_key = jax.random.PRNGKey(seed)
        rng_key, subkey = jax.random.split(rng_key)
        
        mcmc = MCMC(NUTS(model_func), num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, thinning=thinning, progress_bar=not quiet)
        
        if not quiet:
            print("Sampling...")
            
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
            results["waic"] = _calculate_waic_internal(model_func, mcmc, jax_data)
            
    else:
        # We just need a dummy rng key
        rng_key = jax.random.PRNGKey(seed)
        
    if not dsep:
        return results
        
    # ----------------------------------------------------
    # D-Separation Testing (and M-Separation)
    # ----------------------------------------------------
    dsep_equations = graph.generate_dsep_equations(latent=latent)
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
        
    jax_dsep_data = {k: jnp.array(v) for k, v in dsep_data.items()}
    
    for i, claim in enumerate(dsep_equations):
        eq_str = claim["equation_string"]
        test_var = claim["test_node"]
        resp = claim["response"]
        claim_type = claim["type"]
        
        if not quiet:
            test_desc = "(Induced Correlation)" if claim_type == "correlation" else "(Conditional Independence)"
            print(f"\n[Test {i+1}/{len(dsep_equations)}] {eq_str}  {test_desc}")
            
        test_parser = FormulaParser([eq_str])
        test_parsed = test_parser.parse()
        test_graph = CausalGraph(test_parsed, deterministic_terms=deterministic_terms)
        test_graph.build()
        
        test_compiler = NumPyroBuilder(test_graph, family_dict=family, deterministic_terms=deterministic_terms, cor_matrices=cor_matrices)
        test_model_func = test_compiler.generate_model_function(data_for_compilation=dsep_data)
        
        rng_key, subkey = jax.random.split(rng_key)
        
        # We run a faster MCMC for dsep (just enough to get confident intervals)
        test_mcmc = MCMC(NUTS(test_model_func), num_warmup=300, num_samples=500, num_chains=1, thinning=1, progress_bar=False)
        test_mcmc.run(subkey, **jax_dsep_data)
        
        samples = test_mcmc.get_samples()
        beta_name = f"beta_{resp}_{test_var}"
        
        if beta_name in samples:
            posteriors = np.asarray(samples[beta_name])
            mean_val = float(np.mean(posteriors))
            # 95% HPDI (using percentile for simplicity in this base implementation)
            ci_lower = float(np.percentile(posteriors, 2.5))
            ci_upper = float(np.percentile(posteriors, 97.5))
            
            # If the 95% CI includes 0, it is conditionally independent
            is_independent = (ci_lower < 0 < ci_upper)
            
            if not quiet:
                status = "PASS (Independent)" if is_independent else "FAIL (Dependent)"
                print(f" -> {beta_name} = {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] - {status}")
                
            dsep_results.append({
                "claim": f"{resp} _|_ {test_var} | {', '.join(claim['conditioning_set'])}",
                "equation": eq_str,
                "coefficient": beta_name,
                "mean": mean_val,
                "ci_2.5": ci_lower,
                "ci_97.5": ci_upper,
                "is_independent": is_independent
            })
            
    results["dsep_results"] = dsep_results
    return results

def _calculate_waic_internal(model_func, mcmc, jax_data):
    """
    Calculates WAIC (Widely Applicable Information Criterion) with standard errors
    using pointwise log-likelihoods.
    
    Ported from because_waic.R
    """
    from numpyro.infer import log_likelihood
    import numpy as np
    from scipy.special import logsumexp
    
    # Get log likelihoods for all observed sites
    log_lik_dict = log_likelihood(model_func, mcmc.get_samples(), **jax_data)
    
    # Flatten across all endogenous variables to get a joint WAIC
    joint_ll = None
    for k, ll in log_lik_dict.items():
        if joint_ll is None:
            joint_ll = np.array(ll)
        else:
            joint_ll += np.array(ll)
            
    if joint_ll is None:
        return None
        
    n_samples, n_obs = joint_ll.shape
    
    # 1. Compute lpd (log pointwise predictive density)
    lpd_i = logsumexp(joint_ll, axis=0) - np.log(n_samples)
    
    # 2. Compute p_waic (effective number of parameters)
    p_waic_i = np.var(joint_ll, axis=0, ddof=1)
    
    # 3. Compute pointwise WAIC
    elpd_waic_i = lpd_i - p_waic_i
    waic_i = -2 * elpd_waic_i
    
    # Totals
    elpd_waic = np.sum(elpd_waic_i)
    p_waic = np.sum(p_waic_i)
    waic = np.sum(waic_i)
    
    # Standard Errors
    se_elpd_waic = np.sqrt(n_obs * np.var(elpd_waic_i, ddof=1))
    se_p_waic = np.sqrt(n_obs * np.var(p_waic_i, ddof=1))
    se_waic = np.sqrt(n_obs * np.var(waic_i, ddof=1))
    
    return {
        "elpd_waic": {"Estimate": float(elpd_waic), "SE": float(se_elpd_waic)},
        "p_waic": {"Estimate": float(p_waic), "SE": float(se_p_waic)},
        "waic": {"Estimate": float(waic), "SE": float(se_waic)},
        "n_obs": n_obs,
        "n_samples": n_samples,
        "pointwise": {
            "elpd_waic_i": elpd_waic_i,
            "p_waic_i": p_waic_i,
            "waic_i": waic_i
        }
    }
