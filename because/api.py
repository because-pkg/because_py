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


def _detect_available_cores() -> int:
    """Detect allocated CPU cores across SLURM/PBS/SGE/LSF, cgroups, or local OS."""
    for env_var in (
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "PBS_NP",
        "NSLOTS",
        "LSB_DJOB_NUMPROC",
        "MC_CORES",
    ):
        val = os.environ.get(env_var, "").strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
    try:
        if os.path.exists("/sys/fs/cgroup/cpu.max"):
            with open("/sys/fs/cgroup/cpu.max", "r") as f:
                parts = f.read().strip().split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    quota, period = int(parts[0]), int(parts[1])
                    if quota > 0 and period > 0:
                        return max(1, quota // period)
        elif os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") and os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as fq, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as fp:
                quota, period = int(fq.read().strip()), int(fp.read().strip())
                if quota > 0 and period > 0:
                    return max(1, quota // period)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) - 1)


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

    avail_cores = _detect_available_cores()
    thread_limit = max(1, int(n_cores), avail_cores)
    # Allocate sufficient logical XLA CPU devices so subsequent fit() calls in
    # the same session can vary n_cores freely without restarting Python/R.
    host_devices = max(int(n_cores), avail_cores, 32)

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
            "TF_NUM_INTEROP_THREADS":        str(thread_limit),
            "TF_NUM_INTRAOP_THREADS":        str(thread_limit),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        for k, v in _thread_env.items():
            os.environ.setdefault(k, v)

        # XLA_FLAGS: merge with any flags already present
        existing_xla = os.environ.get("XLA_FLAGS", "")
        new_flags = []
        if "--xla_force_host_platform_device_count" not in existing_xla:
            new_flags.append(f"--xla_force_host_platform_device_count={host_devices}")
        if "--xla_cpu_multi_thread_eigen" not in existing_xla:
            new_flags.append("--xla_cpu_multi_thread_eigen=false")
        if "intra_op_parallelism_threads" not in existing_xla:
            new_flags.append(f"intra_op_parallelism_threads={thread_limit}")
        if "inter_op_parallelism_threads" not in existing_xla:
            new_flags.append(f"inter_op_parallelism_threads={thread_limit}")
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
    if host_devices > 1 and not jax_already_loaded:
        try:
            _numpyro.set_host_device_count(host_devices)
        except Exception as exc:
            warnings.warn(
                f"because: numpyro.set_host_device_count({host_devices}) failed: {exc}",
                RuntimeWarning,
            )

    _jax_initialized = True

def get_dsep_equations(equations, latent=None):
    """Helper to return implied d-sep claims without fitting the model."""
    if isinstance(latent, str):
        latent = [latent]
    test_parser = FormulaParser(equations)
    test_parsed = test_parser.parse()
    graph = CausalGraph(test_parsed)
    graph.build()
    return graph.generate_dsep_equations(latent=latent)

def fit(equations, data, family=None, latent=None, cor_matrices=None, induced_correlations=None, dsep=False, dsep_only=False, calculate_waic=False, num_samples=1000, num_warmup=500, num_chains=1, thinning=1, n_cores=1, seed=0, dsep_max_obs=2000, quiet=False, dsep_equations_to_run=None, adapt_delta=0.95, max_treedepth=10, fix_latent="loading", prior_scale_fixed=None, levels=None, **kwargs):
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
    # Backward compatibility with alternate parameter names
    if "n_samples" in kwargs:
        num_samples = kwargs.pop("n_samples")
    if "n_warmup" in kwargs:
        num_warmup = kwargs.pop("n_warmup")
    if "n_chains" in kwargs:
        num_chains = kwargs.pop("n_chains")
    if "priors" in kwargs:
        kwargs.pop("priors")
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
        
    compiler = NumPyroBuilder(graph, family_dict=family, deterministic_terms=deterministic_terms, cor_matrices=cor_matrices, fix_latent=fix_latent, induced_correlations=induced_correlations, prior_scale_fixed=prior_scale_fixed, levels=levels)
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

    def _wrap_ensemble_model(orig_fn, cor_mats, static_data):
        def _ensemble_model(_tree_idx=None, **override_kwargs):
            if _tree_idx is not None and cor_mats:
                for v in cor_mats.values():
                    if isinstance(v, dict) and v.get("type") == "multiPhylo_ensemble":
                        v["matrix"]["_active_tree_idx"] = _tree_idx
            merged = dict(static_data)
            if override_kwargs:
                merged.update(override_kwargs)
            return orig_fn(**merged)
        return _ensemble_model

    def _run_ensemble_chains(rng_key, ens_model_func, data_dict, n_trees, n_chains, n_cores_req, show_progress=True):
        total_chains = int(n_trees * n_chains)
        avail_devices = int(jax.local_device_count())
        if show_progress and int(n_cores_req) > 1 and avail_devices < min(int(n_cores_req), total_chains):
            print(
                f"  [MultiPhylo Ensemble] WARNING: Only {avail_devices} XLA CPU device(s) active in this R session "
                f"(requested n.cores={n_cores_req}). Please restart R ('Session -> Restart R') once "
                f"so JAX can allocate all {n_cores_req} CPU cores in parallel."
            )
        W = int(min(max(1, int(n_cores_req)), avail_devices, total_chains))
        kernel_base = NUTS(
            ens_model_func,
            target_accept_prob=adapt_delta,
            max_tree_depth=max_treedepth,
            init_strategy=numpyro.infer.init_to_median(),
        )

        if W > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*There are not enough devices.*")
                mcmc_obj = MCMC(
                    kernel_base,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=1,
                    thinning=thinning,
                    chain_method="sequential",
                    jit_model_args=True,
                    progress_bar=False,
                )
            # Pre-initialize sampler state & cached functions on the host once before pmap
            rng_key, init_key = jax.random.split(rng_key)
            init_kw = {"_tree_idx": jnp.int32(0)}
            mcmc_obj._args = ()
            mcmc_obj._kwargs = init_kw
            mcmc_obj.sampler.init(init_key, mcmc_obj.num_warmup, init_params=None, model_args=(), model_kwargs=init_kw)
            mcmc_obj._get_cached_fns()
            collect_fields = (mcmc_obj._sample_field,) + tuple(mcmc_obj._default_fields)

            def _pmap_single_chain(chain_rng_key, tree_idx):
                states, _ = mcmc_obj._single_chain_mcmc(
                    (chain_rng_key, None, None),
                    args=(),
                    kwargs={"_tree_idx": tree_idx},
                    collect_fields=collect_fields,
                    remove_sites=(),
                )
                return states[mcmc_obj._sample_field]

            pmap_fn = jax.pmap(_pmap_single_chain, devices=jax.local_devices()[:W])
            all_tree_indices = np.repeat(np.arange(n_trees, dtype=np.int32), n_chains)
            n_waves = int(np.ceil(total_chains / W))
            collected_chunks = []

            for w in range(n_waves):
                start_i = w * W
                end_i = min(total_chains, (w + 1) * W)
                chunk_indices = all_tree_indices[start_i:end_i]
                chunk_len = len(chunk_indices)
                if show_progress:
                    t_min, t_max = int(chunk_indices[0]) + 1, int(chunk_indices[-1]) + 1
                    tree_span = f"Tree {t_min}" if t_min == t_max else f"Trees {t_min}..{t_max}"
                    print(f"  [MultiPhylo Ensemble: Wave {w+1}/{n_waves}] Running {chunk_len} chain(s) in parallel across {W} CPU cores ({tree_span})...")
                rng_key, wave_key = jax.random.split(rng_key)
                wave_keys = jax.random.split(wave_key, W)
                if chunk_len < W:
                    pad_indices = np.pad(chunk_indices, (0, W - chunk_len), mode="edge")
                else:
                    pad_indices = chunk_indices
                wave_out = pmap_fn(wave_keys, jnp.asarray(pad_indices, dtype=jnp.int32))
                collected_chunks.append({
                    param: np.asarray(val)[:chunk_len] for param, val in wave_out.items()
                })

            numpy_samples_all = {
                param: np.concatenate([ch[param] for ch in collected_chunks], axis=0)
                for param in collected_chunks[0].keys()
            }

            # Inject 1-based tree index parameter for multiPhylo ensemble monitoring
            k_tree_var = None
            if cor_matrices:
                for k_struct, v_struct in cor_matrices.items():
                    if isinstance(v_struct, dict) and v_struct.get("type") == "multiPhylo_ensemble":
                        k_tree_var = f"K_tree_{k_struct}"
                        break
            if k_tree_var is None:
                k_tree_var = "K_tree_multiPhylo"

            if len(numpy_samples_all) > 0:
                first_param_arr = next(iter(numpy_samples_all.values()))
                num_post_warmup = first_param_arr.shape[1]
                k_tree_matrix = np.tile((all_tree_indices + 1)[:, None], (1, num_post_warmup)).astype(np.float64)
                numpy_samples_all[k_tree_var] = k_tree_matrix

            tree_samples_list = []
            for k in range(n_trees):
                c_start = k * n_chains
                c_end = (k + 1) * n_chains
                tree_samples_list.append({
                    param: val[c_start:c_end] for param, val in numpy_samples_all.items()
                })
            return rng_key, mcmc_obj, tree_samples_list, numpy_samples_all
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*There are not enough devices.*")
                mcmc_obj = MCMC(
                    kernel_base,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=n_chains,
                    thinning=thinning,
                    chain_method="sequential",
                    jit_model_args=True,
                    progress_bar=show_progress,
                )
            tree_samples_list = []
            for k in range(n_trees):
                if show_progress:
                    print(f"  [MultiPhylo Ensemble: Tree {k+1}/{n_trees}] Sampling {n_chains} chain(s) sequentially...")
                rng_key, subkey = jax.random.split(rng_key)
                mcmc_obj.run(subkey, _tree_idx=jnp.int32(k))
                s_k = mcmc_obj.get_samples(group_by_chain=True)
                tree_samples_list.append({param: np.asarray(val) for param, val in s_k.items()})
            numpy_samples_all = {
                param: np.concatenate([ts[param] for ts in tree_samples_list], axis=0)
                for param in tree_samples_list[0].keys()
            }

            k_tree_var = None
            if cor_matrices:
                for k_struct, v_struct in cor_matrices.items():
                    if isinstance(v_struct, dict) and v_struct.get("type") == "multiPhylo_ensemble":
                        k_tree_var = f"K_tree_{k_struct}"
                        break
            if k_tree_var is None:
                k_tree_var = "K_tree_multiPhylo"

            all_tree_indices = np.repeat(np.arange(n_trees, dtype=np.int32), n_chains)
            if len(numpy_samples_all) > 0:
                first_param_arr = next(iter(numpy_samples_all.values()))
                num_post_warmup = first_param_arr.shape[1]
                k_tree_matrix = np.tile((all_tree_indices + 1)[:, None], (1, num_post_warmup)).astype(np.float64)
                numpy_samples_all[k_tree_var] = k_tree_matrix
                for k in range(n_trees):
                    tree_samples_list[k][k_tree_var] = np.full((n_chains, num_post_warmup), k + 1, dtype=np.float64)

            return rng_key, mcmc_obj, tree_samples_list, numpy_samples_all

    ensemble_structs = [
        v for v in (cor_matrices or {}).values()
        if isinstance(v, dict) and v.get("type") == "multiPhylo_ensemble"
    ]
    has_multiPhylo_ensemble = len(ensemble_structs) > 0

    if not dsep_only:
        rng_key = jax.random.PRNGKey(seed)

        if has_multiPhylo_ensemble:
            n_trees = int(ensemble_structs[0].get("n_trees", jax_data.get("Ntree", 10)))
            ens_n_chains = int(ensemble_structs[0].get("n_chains", ensemble_structs[0].get("chains_per_tree", num_chains if num_chains > 1 else 2)))
            ens_model_func = _wrap_ensemble_model(model_func, cor_matrices, jax_data)
            rng_key, mcmc, tree_samples_list, numpy_samples = _run_ensemble_chains(
                rng_key, ens_model_func, jax_data, n_trees, ens_n_chains, n_cores, show_progress=not quiet
            )
        else:
            rng_key, subkey = jax.random.split(rng_key)
            kernel_base = NUTS(model_func, target_accept_prob=adapt_delta, max_tree_depth=max_treedepth, init_strategy=numpyro.infer.init_to_median())
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
            avail_devices = int(jax.local_device_count())
            c_method = "parallel" if (int(n_cores) > 1 and int(num_chains) > 1 and avail_devices >= int(num_chains)) else "sequential"
            if not quiet and int(num_chains) > 1:
                if c_method == "parallel":
                    print(f"  [NumPyro MCMC] Running {num_chains} chains in parallel across {num_chains} CPU cores...")
                elif int(n_cores) > 1 and avail_devices < int(num_chains):
                    print(
                        f"  [NumPyro MCMC] WARNING: Only {avail_devices} XLA CPU device(s) active in this R session "
                        f"(requested n.cores={n_cores} for {num_chains} chains). "
                        f"Falling back to sequential chains. Please restart R ('Session -> Restart R') once "
                        f"so JAX can allocate multiple CPU cores in parallel."
                    )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*There are not enough devices.*")
                mcmc = MCMC(kernel_base, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, thinning=thinning, chain_method=c_method, progress_bar=not quiet)

            mcmc.run(subkey, **jax_data)

            samples = mcmc.get_samples(group_by_chain=True)
            numpy_samples = {k: np.asarray(v) for k, v in samples.items()}

        results["samples"] = numpy_samples
        results["mcmc"] = mcmc
        results["parameter_map"] = None
        results["model_string"] = "NumPyro Causal Hierarchical Model"  # legacy placeholder
        results["model_code"] = model_code_str  # executable Python source string

        if calculate_waic:
            if not quiet:
                print("Calculating WAIC...")

            has_multiPhylo = cor_matrices and any(isinstance(v, dict) and v.get("type") == "multiPhylo" for v in cor_matrices.values())
            if has_multiPhylo_ensemble:
                results["waic"] = _calculate_waic_ensemble(ens_model_func, tree_samples_list, jax_data)
            elif has_multiPhylo:
                waic_model_func = compiler.generate_model_function(data_for_compilation=data, force_plate_obs=True)
                results["waic"] = _calculate_waic_internal(waic_model_func, mcmc, jax_data)
            else:
                results["waic"] = _calculate_waic_internal(model_func, mcmc, jax_data)

    else:
        # We just need a dummy rng key
        rng_key = jax.random.PRNGKey(seed)

    if not dsep:
        return results

    # D-Separation Testing (and M-Separation)
    # ----------------------------------------------------
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
        return results

    if not quiet:
        msg = f"\nRunning M-Separation tests (MAG)" if latent else f"\nRunning D-Separation tests"
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

        test_compiler = NumPyroBuilder(test_graph, family_dict=family, deterministic_terms=test_det_terms, cor_matrices=cor_matrices, fix_latent=fix_latent, prior_scale_fixed=prior_scale_fixed, levels=levels)
        test_model_func = test_compiler.generate_model_function(data_for_compilation=dsep_data)

        if has_multiPhylo_ensemble:
            n_trees = int(ensemble_structs[0].get("n_trees", jax_dsep_data.get("Ntree", 10)))
            ens_n_chains = max(2, int(ensemble_structs[0].get("n_chains", ensemble_structs[0].get("chains_per_tree", 2))))
            test_ens_func = _wrap_ensemble_model(test_model_func, cor_matrices, jax_dsep_data)
            rng_key, test_mcmc, test_tree_samples, samples_grouped = _run_ensemble_chains(
                rng_key, test_ens_func, jax_dsep_data, n_trees, ens_n_chains, n_cores, show_progress=False
            )
        else:
            rng_key, subkey = jax.random.split(rng_key)
            dsep_chains = max(2, num_chains)
            kernel_base = NUTS(test_model_func, target_accept_prob=adapt_delta, max_tree_depth=max_treedepth, init_strategy=numpyro.infer.init_to_median())
            if cor_matrices and any(isinstance(v, dict) and v.get("type") == "multiPhylo" for v in cor_matrices.values()):
                n_trees = int(jax_dsep_data.get("Ntree", 10))
                kernel_base = DiscreteHMCGibbs(kernel_base)
            c_method = "parallel" if (int(n_cores) > 1 and int(dsep_chains) > 1) else "sequential"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*There are not enough devices.*")
                test_mcmc = MCMC(kernel_base, num_warmup=num_warmup, num_samples=num_samples, num_chains=dsep_chains, thinning=thinning, chain_method=c_method, progress_bar=False)
                test_mcmc.run(subkey, **jax_dsep_data)
            samples_grouped = test_mcmc.get_samples(group_by_chain=True)

        samples_flat = {k: np.asarray(v).flatten() for k, v in samples_grouped.items()}

        beta_name = f"beta_{resp}_{test_var}"

        if beta_name in samples_flat:
            posteriors = samples_flat[beta_name]
            mean_val = float(np.mean(posteriors))
            ci_lower = float(np.percentile(posteriors, 2.5))
            ci_upper = float(np.percentile(posteriors, 97.5))

            chain_samples = np.asarray(samples_grouped[beta_name])
            if has_multiPhylo_ensemble and len(test_tree_samples) > 0 and test_tree_samples[0][beta_name].shape[0] >= 2:
                # Within-tree Rhat for MCMC convergence
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

    results["dsep_results"] = dsep_results
    return results


def _accumulate_waic_from_log_lik(log_lik_dict):
    import numpy as np
    from scipy.special import logsumexp

    if not log_lik_dict:
        return None

    total_elpd_waic = 0.0
    total_p_waic    = 0.0
    total_waic      = 0.0
    total_n_obs     = 0
    se_sq_elpd      = 0.0
    se_sq_p         = 0.0
    se_sq_waic      = 0.0
    all_elpd_i      = []
    all_p_waic_i    = []
    all_waic_i      = []

    n_samples = None
    for k, ll in log_lik_dict.items():
        ll_np = np.array(ll)          # shape: (n_samples, n_obs_k)
        n_samples, n_obs_k = ll_np.shape

        lpd_i      = logsumexp(ll_np, axis=0) - np.log(n_samples)
        p_waic_i   = np.var(ll_np, axis=0, ddof=1)
        elpd_i     = lpd_i - p_waic_i
        waic_i     = -2.0 * elpd_i

        all_elpd_i.append(elpd_i)
        all_p_waic_i.append(p_waic_i)
        all_waic_i.append(waic_i)

        total_elpd_waic += float(np.sum(elpd_i))
        total_p_waic    += float(np.sum(p_waic_i))
        total_waic      += float(np.sum(waic_i))

        se_sq_elpd  += n_obs_k * float(np.var(elpd_i,   ddof=1))
        se_sq_p     += n_obs_k * float(np.var(p_waic_i, ddof=1))
        se_sq_waic  += n_obs_k * float(np.var(waic_i,   ddof=1))
        total_n_obs += n_obs_k

    return {
        "elpd_waic": {"Estimate": total_elpd_waic, "SE": float(np.sqrt(se_sq_elpd))},
        "p_waic":    {"Estimate": total_p_waic,    "SE": float(np.sqrt(se_sq_p))},
        "waic":      {"Estimate": total_waic,      "SE": float(np.sqrt(se_sq_waic))},
        "pointwise": {
            "elpd_waic_i": np.concatenate(all_elpd_i).tolist(),
            "p_waic_i":    np.concatenate(all_p_waic_i).tolist(),
            "waic_i":      np.concatenate(all_waic_i).tolist(),
        },
        "n_obs":     total_n_obs,
        "n_samples": n_samples,
    }


def _calculate_waic_ensemble(ens_model_func, tree_samples_list, jax_data):
    from numpyro.infer import log_likelihood
    import numpy as np
    import jax.numpy as jnp

    ll_per_tree = []
    for k, ts in enumerate(tree_samples_list):
        flat_k = {
            param: jnp.array(val.reshape(-1, *val.shape[2:]))
            for param, val in ts.items()
        }
        ll_k = log_likelihood(ens_model_func, flat_k, _tree_idx=jnp.int32(k), **jax_data)
        ll_per_tree.append({var: np.asarray(v) for var, v in ll_k.items()})

    if not ll_per_tree or not ll_per_tree[0]:
        return None

    log_lik_dict = {
        var_name: np.concatenate([ll_per_tree[k][var_name] for k in range(len(tree_samples_list))], axis=0)
        for var_name in ll_per_tree[0].keys()
    }
    return _accumulate_waic_from_log_lik(log_lik_dict)


def _calculate_waic_internal(model_func, mcmc, jax_data):
    from numpyro.infer import log_likelihood
    import numpy as np
    from scipy.special import logsumexp
    import jax.numpy as jnp

    flat_samples = mcmc.get_samples()

    k_tree_keys = [k for k in flat_samples.keys() if k.startswith("K_tree")]
    has_multi_phylo = len(k_tree_keys) > 0

    if has_multi_phylo:
        n_trees = int(jax_data.get("Ntree", max(int(v.max()) + 1
                                                 for v in (np.array(flat_samples[k])
                                                           for k in k_tree_keys))))
        ll_per_tree = []
        for k in range(n_trees):
            fixed_samples = {
                key: (jnp.full_like(val, k) if key in k_tree_keys else val)
                for key, val in flat_samples.items()
            }
            ll_k = log_likelihood(model_func, fixed_samples, **jax_data)
            ll_per_tree.append({var: np.array(v) for var, v in ll_k.items()})

        log_lik_dict = {}
        for var_name in ll_per_tree[0].keys():
            ll_stack = np.stack([ll_per_tree[k][var_name] for k in range(n_trees)], axis=0)
            log_lik_dict[var_name] = logsumexp(ll_stack, axis=0) - np.log(n_trees)
    else:
        log_lik_dict = log_likelihood(model_func, flat_samples, **jax_data)

    return _accumulate_waic_from_log_lik(log_lik_dict)


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
    has_rhat_tree = any(t.get("rhat_tree") is not None for t in tests)
    for t in tests:
        row = {
            "claim":       t["claim"],
            "coefficient": t["coefficient"],
            "mean":        round(t["mean"],    digits),
            "2.5%":        round(t["ci_2.5"],  digits),
            "97.5%":       round(t["ci_97.5"], digits),
            "result":      "PASS" if t["is_independent"] else "FAIL",
            "Rhat":        round(t["rhat"],    digits),
        }
        if has_rhat_tree:
            row["Rhat_tree"] = round(t["rhat_tree"], digits) if t.get("rhat_tree") is not None else None
        row["n_eff"] = round(t["n_eff"],   1)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("claim")

    if verbose:
        n_pass  = sum(1 for t in tests if t["is_independent"])
        n_total = len(tests)
        verdict = "Model is consistent with the data." if n_pass == n_total \
                  else "One or more independence claims FAILED — consider revising the DAG."
        print(f"D-separation tests: {n_pass}/{n_total} passed. {verdict}\n")

    return df

# Alias for backwards compatibility with R package wrappers
fit_numpyro_model = fit
