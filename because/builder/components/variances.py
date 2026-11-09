import re

def setup_variance_partitioning(topo_order, families, equations_dict, induced_cor_set, numpyro, dist, jnp):
    """
    Partitions total variance using Beta (K=2) or Dirichlet (K>2) distribution
    to prevent non-identifiability when variables have multiple competing variance components.
    """
    var_variances = {}
    for var in topo_order:
        family = families.get(var, "gaussian").lower()
        if family != "gaussian":
            continue

        eq = equations_dict.get(var)
        if not eq and var not in induced_cor_set:
            continue

        if not eq:
            eq = {"random": []}

        components = ["indep"]
        if var in induced_cor_set:
            components.append("shared")
        for rand_term in eq.get("random", []):
            match = re.search(r"\(1\s*\|\s*([^)]+)\)", rand_term)
            if match:
                components.append(match.group(1).strip())

        K = len(components)
        if K > 1:
            tot_name = f"_sigma_tot_{var}" if var in induced_cor_set else f"sigma_tot_{var}"
            sigma_tot = numpyro.sample(tot_name, dist.HalfNormal(2.0))
            if K == 2:
                prop_name = f"_prop_{var}" if var in induced_cor_set else f"prop_{var}"
                prop = numpyro.sample(prop_name, dist.Beta(2.0, 2.0))
                props = jnp.array([prop, 1.0 - prop])
            else:
                props_name = f"_props_{var}" if var in induced_cor_set else f"props_{var}"
                props = numpyro.sample(props_name, dist.Dirichlet(jnp.ones(K) * 2.0))

            var_variances[var] = {}
            for i, comp in enumerate(components):
                if comp == "indep":
                    var_variances[var][comp] = sigma_tot * jnp.sqrt(props[i]) + 1e-3
                elif comp == "shared":
                    var_variances[var][comp] = numpyro.deterministic(
                        f"sigma_shared_{var}", sigma_tot * jnp.sqrt(props[i]) + 1e-3
                    )
                else:
                    if var not in induced_cor_set:
                        var_variances[var][comp] = numpyro.deterministic(
                            f"sigma_{var}_{comp}", sigma_tot * jnp.sqrt(props[i]) + 1e-3
                        )
                    else:
                        var_variances[var][comp] = sigma_tot * jnp.sqrt(props[i]) + 1e-3

    return var_variances


def sample_induced_correlations(
    ind_cor_pairs,
    var_variances,
    N,
    data,
    has_multiPhylo,
    force_plate_obs,
    numpyro,
    dist,
    jnp,
):
    """
    Samples MAG induced correlations between variable pairs using LKJ Cholesky
    and Non-Centered Parameterization (NCP) for latent MVN errors.
    """
    induced_err_storage = {}
    for pair in ind_cor_pairs:
        var1, var2 = pair[0], pair[1]
        pair_tag = f"{var1}_{var2}"

        sigma_res1 = var_variances[var1]["shared"]
        sigma_res2 = var_variances[var2]["shared"]

        L_corr = numpyro.sample(
            f"L_corr_{pair_tag}",
            dist.LKJCholesky(2, concentration=2.0)
        )
        numpyro.deterministic(f"rho_{pair_tag}", L_corr[1, 0])

        scale_diag = jnp.array([sigma_res1, sigma_res2])
        L_cov = scale_diag[:, None] * L_corr

        n_pair = N
        for _v in [var1, var2]:
            if _v in data and hasattr(data[_v], "shape") and len(data[_v].shape) > 0:
                n_pair = int(data[_v].shape[0])
                break
            n_key = f"N_{_v}"
            if n_key in data:
                n_pair = int(data[n_key])
                break

        if has_multiPhylo and not force_plate_obs:
            z_err = numpyro.sample(
                f"z_err_{pair_tag}",
                dist.Normal(0, 1).expand([n_pair, 2]).to_event(2)
            )
        else:
            with numpyro.plate(f"err_res_{pair_tag}_plate", n_pair, dim=-1):
                z_err = numpyro.sample(f"z_err_{pair_tag}", dist.Normal(0, 1).expand([2]).to_event(1))

        err = numpyro.deterministic(f"err_res_{pair_tag}", jnp.dot(z_err, L_cov.T))

        if var1 not in induced_err_storage:
            induced_err_storage[var1] = []
        induced_err_storage[var1].append(err[..., 0])

        if var2 not in induced_err_storage:
            induced_err_storage[var2] = []
        induced_err_storage[var2].append(err[..., 1])

    return induced_err_storage
