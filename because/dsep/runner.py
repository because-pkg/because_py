import warnings
import numpy as np
from ..builder import FormulaParser, CausalGraph, NumPyroBuilder
from ..engine.env import get_jax_primitives
from ..engine.ensemble import run_ensemble_chains, wrap_ensemble_model

def run_dsep_tests(
    graph,
    data,
    latent=None,
    dsep_equations_to_run=None,
    deterministic_terms=None,
    family=None,
    cor_matrices=None,
    fix_latent="loading",
    prior_scale_fixed=None,
    levels=None,
    dsep_max_obs=2000,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    thinning=1,
    n_cores=1,
    adapt_delta=0.95,
    max_treedepth=10,
    rng_key=None,
    quiet=False,
    has_multiPhylo_ensemble=False,
    ensemble_structs=None,
):
    """
    Executes D-Separation / M-Separation testing for implied conditional independence claims.
    """
    prim = get_jax_primitives()
    jax = prim["jax"]
    jnp = prim["jnp"]
    numpyro = prim["numpyro"]
    MCMC = prim["MCMC"]
    NUTS = prim["NUTS"]
    DiscreteHMCGibbs = prim["DiscreteHMCGibbs"]
    diag = prim["diag"]

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    if dsep_equations_to_run is not None and len(dsep_equations_to_run) > 0 and isinstance(dsep_equations_to_run[0], dict):
        dsep_equations = dsep_equations_to_run
    else:
        dsep_equations = graph.generate_dsep_equations(latent=latent)
        if dsep_equations_to_run is not None:
            norm_to_run = [" ".join(e.split()) for e in dsep_equations_to_run]
            dsep_equations = [c for c in dsep_equations if " ".join(c["equation_string"].split()) in norm_to_run]

    if not dsep_equations:
        if not quiet:
            print("Graph is fully connected or no testable claims exist. No implied conditional independencies to test.")
        return []

    if not quiet:
        msg = "\nRunning M-Separation tests (MAG)" if latent else "\nRunning D-Separation tests"
        print(f"{msg} ({len(dsep_equations)} implied claims)...")

    dsep_results = []

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

    ensemble_structs = ensemble_structs or []

    for i, claim in enumerate(dsep_equations):
        eq_str = claim["equation_string"]
        test_var = claim["test_node"]
        resp = claim["response"]
        claim_type = claim["type"]

        if not quiet:
            test_desc = "(Induced Correlation)" if claim_type == "correlation" else "(Conditional Independence)"
            print(f"\n[Test {i+1} / {len(dsep_equations)}]  {eq_str}  {test_desc}")

        compile_eq_str = claim.get("compile_equation_string", eq_str)

        test_parser = FormulaParser([compile_eq_str])
        test_parsed = test_parser.parse()
        test_det_terms = {**(deterministic_terms or {}), **(test_parser.deterministic_terms or {})}
        test_graph = CausalGraph(test_parsed, deterministic_terms=test_det_terms)
        test_graph.build()

        test_compiler = NumPyroBuilder(
            test_graph,
            family_dict=family,
            deterministic_terms=test_det_terms,
            cor_matrices=cor_matrices,
            fix_latent=fix_latent,
            prior_scale_fixed=prior_scale_fixed,
            levels=levels
        )
        test_model_func = test_compiler.generate_model_function(data_for_compilation=dsep_data)

        if has_multiPhylo_ensemble:
            n_trees = int(ensemble_structs[0].get("n_trees", jax_dsep_data.get("Ntree", 10)))
            ens_n_chains = max(2, int(ensemble_structs[0].get("n_chains", ensemble_structs[0].get("chains_per_tree", 2))))
            test_ens_func = wrap_ensemble_model(test_model_func, cor_matrices, jax_dsep_data)
            rng_key, test_mcmc, test_tree_samples, samples_grouped = run_ensemble_chains(
                rng_key, test_ens_func, jax_dsep_data, n_trees, ens_n_chains, n_cores,
                num_warmup=num_warmup, num_samples=num_samples, thinning=thinning,
                adapt_delta=adapt_delta, max_treedepth=max_treedepth, cor_matrices=cor_matrices,
                show_progress=False
            )
        else:
            rng_key, subkey = jax.random.split(rng_key)
            dsep_chains = max(2, num_chains)
            kernel_base = NUTS(
                test_model_func,
                target_accept_prob=adapt_delta,
                max_tree_depth=max_treedepth,
                init_strategy=numpyro.infer.init_to_median()
            )
            if cor_matrices and any(isinstance(v, dict) and v.get("type") == "multiPhylo" for v in cor_matrices.values()):
                n_trees = int(jax_dsep_data.get("Ntree", 10))
                kernel_base = DiscreteHMCGibbs(kernel_base)
            c_method = "parallel" if (int(n_cores) > 1 and int(dsep_chains) > 1) else "sequential"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*There are not enough devices.*")
                test_mcmc = MCMC(
                    kernel_base,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=dsep_chains,
                    thinning=thinning,
                    chain_method=c_method,
                    progress_bar=False
                )
                test_mcmc.run(subkey, **jax_dsep_data)
            samples_grouped = test_mcmc.get_samples(group_by_chain=True)
            test_tree_samples = []

        samples_flat = {k: np.asarray(v).flatten() for k, v in samples_grouped.items()}

        beta_name = f"beta_{resp}_{test_var}"

        if beta_name in samples_flat:
            posteriors = samples_flat[beta_name]
            mean_val = float(np.mean(posteriors))
            ci_lower = float(np.percentile(posteriors, 2.5))
            ci_upper = float(np.percentile(posteriors, 97.5))

            chain_samples = np.asarray(samples_grouped[beta_name])
            if has_multiPhylo_ensemble and len(test_tree_samples) > 0 and test_tree_samples[0][beta_name].shape[0] >= 2:
                rhat = float(np.max([diag.gelman_rubin(ts[beta_name]) for ts in test_tree_samples]))
                rhat_tree = float(diag.gelman_rubin(chain_samples))
            else:
                rhat = float(diag.gelman_rubin(chain_samples))
                rhat_tree = None
            n_eff = float(diag.effective_sample_size(chain_samples))

            is_independent = (ci_lower < 0 < ci_upper)

            if not quiet:
                rhat_tree_msg = f"  Rhat_tree={rhat_tree:.3f}" if rhat_tree is not None else ""
                print(f" -> {beta_name} = {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] (Rhat={rhat:.3f}{rhat_tree_msg})")

            dsep_results.append({
                "claim": f"{resp} _|_ {test_var} | {', '.join(claim['conditioning_set'])}",
                "equation": eq_str,
                "coefficient": beta_name,
                "mean": mean_val,
                "ci_2.5": ci_lower,
                "ci_97.5": ci_upper,
                "is_independent": is_independent,
                "rhat": rhat,
                "rhat_tree": rhat_tree,
                "n_eff": n_eff,
                "samples": np.asarray(posteriors).tolist()
            })

    return dsep_results
