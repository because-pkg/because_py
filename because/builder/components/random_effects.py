import re
import numpy as np

def can_spectral_marginalize(
    has_multiPhylo,
    family,
    is_latent,
    obs_data,
    has_nans,
    eq,
    target_size,
    custom_transforms,
):
    """
    Checks if a Gaussian response with multiPhylo random effect qualifies for
    analytical spectral Gaussian marginalization.
    """
    if not (has_multiPhylo and family == "gaussian" and not is_latent and obs_data is not None and not has_nans and len(eq.get("random", [])) == 1):
        return False, None

    m_rand = re.search(r"\(1\s*\|\s*([^)]+)\)", eq["random"][0])
    if m_rand:
        cand_grp = m_rand.group(1).strip()
        if (
            cand_grp in custom_transforms
            and custom_transforms[cand_grp].get("type") == "multiPhylo"
            and isinstance(custom_transforms[cand_grp].get("matrix"), dict)
            and "eigvals" in custom_transforms[cand_grp]["matrix"]
            and "eigvecs" in custom_transforms[cand_grp]["matrix"]
        ):
            eigvals_mat = custom_transforms[cand_grp]["matrix"]["eigvals"]
            if eigvals_mat.shape[0] == target_size:
                return True, cand_grp

    return False, None


def sample_spectral_marginalization(
    var,
    spectral_group_name,
    obs_data,
    mu,
    data,
    custom_transforms,
    var_variances,
    induced_cor_set,
    induced_err_storage,
    shared_state,
    force_plate_obs,
    target_size,
    numpyro,
    dist,
    jnp,
):
    """
    Integrates out N-dimensional phylogenetic random effects analytically in the tree eigenbasis:
    Q_K^T (y - mu) ~ Normal(0, sqrt(sigma_phylo^2 * d_K + sigma_res^2))
    """
    group_name = spectral_group_name
    if group_name not in data:
        raise ValueError(f"Grouping variable '{group_name}' not found in data.")
    group_idx = data[group_name]

    if var in var_variances and group_name in var_variances[var]:
        sigma_group = var_variances[var][group_name]
    else:
        sigma_group = numpyro.sample(f"sigma_{var}_{group_name}", dist.HalfNormal(2.0))

    if var in var_variances and "indep" in var_variances[var]:
        sigma = numpyro.deterministic(f"sigma_{var}", var_variances[var]["indep"])
    else:
        sigma = numpyro.sample(f"sigma_{var}", dist.HalfNormal(2.0))

    if var not in induced_cor_set:
        numpyro.deterministic(
            f"lambda_{var}_{group_name}",
            (sigma_group ** 2) / (sigma_group ** 2 + sigma ** 2)
        )

    if var in induced_err_storage:
        for err in induced_err_storage[var]:
            mu = mu + err

    matrix_dict = custom_transforms[group_name]["matrix"]
    eigvals_all = jnp.array(matrix_dict["eigvals"])
    eigvecs_all = jnp.array(matrix_dict["eigvecs"])
    n_trees = eigvals_all.shape[-1]

    k_name = f"K_tree_{group_name}"
    if "_active_tree_idx" in matrix_dict and matrix_dict["_active_tree_idx"] is not None:
        K = matrix_dict["_active_tree_idx"]
        if k_name not in shared_state:
            shared_state[k_name] = numpyro.deterministic(k_name, K + 1)
    else:
        if k_name not in shared_state:
            shared_state[k_name] = numpyro.sample(
                k_name, dist.Categorical(probs=jnp.ones(n_trees) / n_trees)
            )
        K = shared_state[k_name]

    eigvals = jnp.take(eigvals_all, K, axis=-1)
    if eigvals.ndim > 1:
        eigvals = jnp.moveaxis(eigvals, 0, -1)

    eigvecs = jnp.take(eigvecs_all, K, axis=-1)
    if eigvecs.ndim > 2:
        eigvecs = jnp.moveaxis(eigvecs, [0, 1], [-2, -1])

    eigvals_safe = jnp.maximum(eigvals, 1e-8)
    eigvecs_aligned = eigvecs[..., group_idx, :]

    res = obs_data - mu
    tilde_r = jnp.einsum("...ij,...i->...j", eigvecs_aligned, res)
    tot_eig_var = (sigma_group ** 2) * eigvals_safe + (sigma ** 2)
    tot_eig_sd = jnp.sqrt(tot_eig_var)

    tilde_u_mean = ((sigma_group ** 2) * eigvals_safe / tot_eig_var) * tilde_r
    u_group = jnp.einsum("...ij,...j->...i", eigvecs, tilde_u_mean)
    u_group = u_group - jnp.mean(u_group, axis=-1, keepdims=True)
    u_group = numpyro.deterministic(f"u_{var}_{group_name}", u_group)

    if not force_plate_obs:
        numpyro.sample(
            var,
            dist.Normal(tilde_r, tot_eig_sd).to_event(1),
            obs=jnp.zeros_like(obs_data)
        )
    else:
        with numpyro.plate(f"{var}_plate", target_size):
            numpyro.sample(
                var,
                dist.Normal(mu + u_group[..., group_idx], sigma),
                obs=obs_data
            )

    return obs_data


def sample_structural_random_effects(
    var,
    eq,
    family,
    mu,
    target_size,
    data,
    var_to_level,
    graph,
    var_variances,
    custom_transforms,
    L_matrices,
    shared_state,
    numpyro,
    dist,
    jnp,
    jax,
):
    """
    Samples hierarchical random effects / intercept plates and applies structural covariance
    transforms (spatial GP Cholesky, phylogenetic VCV).
    """
    sigma_struct_dict = {}

    for rand_term in eq.get("random", []):
        match = re.search(r"\(1\s*\|\s*([^)]+)\)", rand_term)
        if not match:
            continue
        group_name = match.group(1).strip()
        if group_name not in data:
            raise ValueError(f"Grouping variable '{group_name}' not found in data.")

        group_idx = data[group_name]

        # Broadcasting for random effects (COARSE group -> FINE response)
        if hasattr(group_idx, "shape") and len(group_idx.shape) > 0 and target_size is not None and group_idx.shape[0] < target_size:
            possible_idx = [
                k for k in data.keys()
                if "idx" in k and hasattr(data[k], "shape") and len(data[k].shape) > 0 and data[k].shape[0] == target_size
            ]

            level_name = None
            if group_name in var_to_level:
                level_name = var_to_level[group_name]
            elif hasattr(graph, 'levels') and graph.levels:
                for lvl, vars_in_lvl in graph.levels.items():
                    if group_name in vars_in_lvl:
                        level_name = lvl
                        break

            expected_size = None
            if f"N_{group_name}" in data:
                expected_size = int(data[f"N_{group_name}"])
            elif group_name in custom_transforms:
                expected_size = len(custom_transforms[group_name]["matrix"])
            elif group_name in L_matrices:
                expected_size = L_matrices[group_name].shape[0]

            if level_name is None and expected_size is not None:
                for k, v in data.items():
                    if k.startswith("N_") and not k.endswith("ID") and not k.endswith("_id") and isinstance(v, (int, float, np.integer, np.floating)):
                        if int(v) == expected_size:
                            potential_lvl = k[2:]
                            if any(idx.startswith(f"{potential_lvl}_idx_") for idx in possible_idx):
                                level_name = potential_lvl
                                break
            if level_name is None:
                level_name = group_name

            best_idx = None
            if var in var_to_level and f"{level_name}_idx_{var_to_level[var]}" in data:
                best_idx = f"{level_name}_idx_{var_to_level[var]}"
            else:
                for k in possible_idx:
                    if k.startswith(f"{level_name}_idx_"):
                        best_idx = k
                        break

            if best_idx:
                group_idx = data[best_idx]
            elif possible_idx:
                group_idx = data[possible_idx[0]]
            else:
                raise ValueError(
                    f"Shape mismatch for random effect: {var} ({target_size}) vs {group_name} ({group_idx.shape[0]}). "
                    f"No bridging index found."
                )

            n_var = f"N_{group_name}"
            if n_var in data:
                num_groups = int(data[n_var])
            elif expected_size is not None:
                num_groups = expected_size
            else:
                try:
                    num_groups = int(np.max(np.asarray(group_idx))) + 1
                except Exception:
                    raise ValueError(
                        f"Cannot dynamically determine number of groups for '{group_name}' "
                        f"during JAX compilation. Please provide '{n_var}' as an integer in your data dictionary."
                    )
        else:
            n_var = f"N_{group_name}"
            if n_var in data:
                num_groups = int(data[n_var])
            else:
                try:
                    num_groups = int(np.max(np.asarray(group_idx))) + 1
                except Exception:
                    raise ValueError(
                        f"Cannot dynamically determine number of groups for '{group_name}' "
                        f"during JAX compilation. Please provide '{n_var}' as an integer in your data dictionary."
                    )

        if var in var_variances and group_name in var_variances[var]:
            sigma_group = var_variances[var][group_name]
        else:
            if family in ["poisson", "binomial", "negbinomial"]:
                sigma_group = numpyro.sample(f"sigma_{var}_{group_name}", dist.HalfNormal(1.0))
            else:
                sigma_group = numpyro.sample(f"sigma_{var}_{group_name}", dist.HalfNormal(2.0))

        with numpyro.plate(f"{var}_{group_name}_plate", num_groups):
            z_group_raw = numpyro.sample(f"z_{var}_{group_name}_raw", dist.Normal(0, 1))

        if group_name in custom_transforms:
            z_group, sigma_group = custom_transforms[group_name]["transform_func"](
                numpyro, jnp, jax, dist, var, group_name, num_groups,
                custom_transforms[group_name]["matrix"], z_group_raw, sigma_group, shared_state
            )
        elif group_name in L_matrices:
            L = L_matrices[group_name]
            if L.shape[0] != num_groups:
                raise ValueError(
                    f"Correlation matrix for '{group_name}' has shape {L.shape} but num_groups is {num_groups}"
                )
            z_group = jnp.dot(L, z_group_raw)
            z_group = z_group - jnp.mean(z_group)
            z_group = numpyro.deterministic(f"z_{var}_{group_name}", z_group)
        else:
            z_group = z_group_raw

        u_group = numpyro.deterministic(f"u_{var}_{group_name}", z_group * sigma_group)
        mu = mu + u_group[group_idx]

        if group_name in custom_transforms or group_name in L_matrices:
            sigma_struct_dict[group_name] = sigma_group

    return mu, sigma_struct_dict
