import numpy as np
from scipy.special import logsumexp
from ..engine.env import get_jax_primitives

def accumulate_waic_from_log_lik(log_lik_dict):
    """
    Computes elpd_waic, p_waic, and waic along with standard errors from
    a dictionary of pointwise log-likelihood arrays.
    """
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


def calculate_waic_ensemble(ens_model_func, tree_samples_list, jax_data, max_samples_per_tree=100, batch_size=50):
    """
    Computes WAIC across multiPhylo ensemble chains with memory-efficient batching.
    """
    import gc
    from numpyro.infer import log_likelihood
    prim = get_jax_primitives()
    jnp = prim["jnp"]

    ll_per_tree = []
    for k, ts in enumerate(tree_samples_list):
        first_val = next(iter(ts.values()))
        n_chains_k = first_val.shape[0]
        n_iters_k = first_val.shape[1]
        total_k = n_chains_k * n_iters_k

        if total_k > max_samples_per_tree:
            sub_idx = np.linspace(0, total_k - 1, max_samples_per_tree, dtype=int)
        else:
            sub_idx = np.arange(total_k)

        n_sub = len(sub_idx)
        tree_var_lls = {}

        for b_start in range(0, n_sub, batch_size):
            b_end = min(n_sub, b_start + batch_size)
            b_indices = sub_idx[b_start:b_end]

            batch_k = {}
            for param, val in ts.items():
                v_flat = val.reshape(-1, *val.shape[2:])
                batch_k[param] = jnp.array(v_flat[b_indices])

            ll_batch = log_likelihood(ens_model_func, batch_k, _tree_idx=jnp.int32(k), **jax_data)
            for var_name, v in ll_batch.items():
                arr = np.asarray(v)
                if var_name not in tree_var_lls:
                    tree_var_lls[var_name] = [arr]
                else:
                    tree_var_lls[var_name].append(arr)

            del batch_k, ll_batch

        ll_per_tree.append({
            var: np.concatenate(tree_var_lls[var], axis=0)
            for var in tree_var_lls.keys()
        })
        del tree_var_lls
        gc.collect()

    if not ll_per_tree or not ll_per_tree[0]:
        return None

    log_lik_dict = {
        var_name: np.concatenate([ll_per_tree[k][var_name] for k in range(len(tree_samples_list))], axis=0)
        for var_name in ll_per_tree[0].keys()
    }
    res = accumulate_waic_from_log_lik(log_lik_dict)
    del log_lik_dict, ll_per_tree
    gc.collect()
    return res


def calculate_waic_internal(model_func, mcmc, jax_data):
    """
    Computes WAIC for standard NumPyro models.
    """
    from numpyro.infer import log_likelihood
    prim = get_jax_primitives()
    jnp = prim["jnp"]

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

    return accumulate_waic_from_log_lik(log_lik_dict)


def calculate_waic(model_func, mcmc, jax_data, has_multiPhylo_ensemble=False, ens_model_func=None, tree_samples_list=None):
    """
    High-level WAIC entrypoint dispatching between ensemble and standard runs.
    """
    if has_multiPhylo_ensemble and ens_model_func is not None and tree_samples_list is not None:
        return calculate_waic_ensemble(ens_model_func, tree_samples_list, jax_data)
    return calculate_waic_internal(model_func, mcmc, jax_data)
