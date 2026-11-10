import warnings
import numpy as np
from .env import get_jax_primitives

def wrap_ensemble_model(orig_fn, cor_mats, static_data):
    """
    Wraps a NumPyro model function so that tree indices are routed to the active
    multiPhylo correlation structure before calling the model.
    """
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


def run_ensemble_chains(
    rng_key,
    ens_model_func,
    data_dict,
    n_trees,
    n_chains,
    n_cores_req,
    num_warmup,
    num_samples,
    thinning=1,
    adapt_delta=0.95,
    max_treedepth=10,
    cor_matrices=None,
    show_progress=True,
):
    """
    Executes multiPhylo ensemble chains across available CPU cores using jax.pmap
    when parallel devices are available, or sequentially across trees otherwise.
    """
    prim = get_jax_primitives()
    jax = prim["jax"]
    jnp = prim["jnp"]
    numpyro = prim["numpyro"]
    MCMC = prim["MCMC"]
    NUTS = prim["NUTS"]

    total_chains = int(n_trees * n_chains)
    avail_devices = int(jax.local_device_count())
    if show_progress and int(n_cores_req) > 1 and avail_devices < min(int(n_cores_req), total_chains):
        print(
            f"  [MultiPhylo Ensemble] WARNING: Only {avail_devices} XLA CPU device(s) active in this session "
            f"(requested n.cores={n_cores_req}). Please restart R / Python once "
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
            raw_z = states[mcmc_obj._sample_field]
            post_fn = mcmc_obj.sampler.postprocess_fn((), {"_tree_idx": tree_idx})
            return jax.vmap(post_fn)(raw_z)

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
