def create_family_distribution(
    var,
    family,
    mu,
    data,
    var_variances,
    induced_cor_set,
    sigma_struct_dict,
    numpyro,
    dist,
    jnp,
):
    """
    Constructs the distribution and prior parameters for a given response family.
    """
    extra_params = {}

    if family == "gaussian":
        if var in var_variances and "indep" in var_variances[var]:
            sigma = var_variances[var]["indep"]
            sigma = numpyro.deterministic(f"sigma_{var}", sigma)
        else:
            sigma = numpyro.sample(f"sigma_{var}", dist.HalfNormal(2.0))
        distribution = dist.Normal(mu, sigma)
        extra_params["sigma"] = sigma

        if var not in induced_cor_set:
            for g_name, sig_g in sigma_struct_dict.items():
                tot_res_var = sigma ** 2
                numpyro.deterministic(
                    f"lambda_{var}_{g_name}",
                    (sig_g ** 2) / (sig_g ** 2 + tot_res_var)
                )

    elif family == "poisson":
        distribution = dist.Poisson(rate=jnp.exp(mu))

    elif family == "binomial":
        distribution = dist.Bernoulli(logits=mu)

    elif family == "negbinomial":
        r = numpyro.sample(f"r_{var}", dist.Gamma(2.0, 0.1))
        distribution = dist.NegativeBinomial2(mean=jnp.exp(mu), concentration=r)
        extra_params["r"] = r

    elif family == "zip":
        psi_logit = numpyro.sample(f"psi_{var}", dist.Normal(0, 2))
        gate = jnp.exp(psi_logit) / (1 + jnp.exp(psi_logit))
        distribution = dist.ZeroInflatedPoisson(gate=gate, rate=jnp.exp(mu))
        extra_params["gate"] = gate

    elif family == "zinb":
        psi_logit = numpyro.sample(f"psi_{var}", dist.Normal(0, 2))
        gate = jnp.exp(psi_logit) / (1 + jnp.exp(psi_logit))
        r = numpyro.sample(f"r_{var}", dist.Gamma(2.0, 0.1))
        distribution = dist.ZeroInflatedNegativeBinomial2(gate=gate, mean=jnp.exp(mu), concentration=r)
        extra_params["gate"] = gate
        extra_params["r"] = r

    elif family == "multinomial":
        distribution = dist.Categorical(logits=mu)

    elif family == "ordinal":
        num_categories = int(data[f"K_{var}"]) if f"K_{var}" in data else 3
        c1 = numpyro.sample(f"c1_{var}", dist.Normal(0, 5))
        increments = numpyro.sample(f"c_inc_{var}", dist.Exponential(1).expand([num_categories - 2]))
        cutpoints = jnp.concatenate([jnp.array([c1]), c1 + jnp.cumsum(increments)])
        distribution = dist.OrderedLogistic(predictor=mu, cutpoints=cutpoints)
        extra_params["cutpoints"] = cutpoints

    else:
        raise ValueError(f"Family '{family}' is not supported.")

    return distribution, extra_params


def sample_response_likelihood(
    var,
    family,
    distribution,
    extra_params,
    mu,
    obs_data,
    target_size,
    has_nans,
    missing_idx,
    obs_idx,
    has_multiPhylo,
    force_plate_obs,
    numpyro,
    dist,
    jnp,
):
    """
    Scores observations or marginalizes/imputes missing values (NaNs) under the response distribution.
    """
    if has_nans:
        if family == "gaussian":
            sigma = extra_params["sigma"]
            dist_imputed = dist.Normal(mu[missing_idx], sigma)
            dist_obs = dist.Normal(mu[obs_idx], sigma)
        elif family == "poisson":
            dist_imputed = dist.Poisson(rate=jnp.exp(mu[missing_idx]))
            dist_obs = dist.Poisson(rate=jnp.exp(mu[obs_idx]))
        elif family == "binomial":
            dist_imputed = dist.Bernoulli(logits=mu[missing_idx])
            dist_obs = dist.Bernoulli(logits=mu[obs_idx])
        elif family == "negbinomial":
            r = extra_params["r"]
            dist_imputed = dist.NegativeBinomial2(mean=jnp.exp(mu[missing_idx]), concentration=r)
            dist_obs = dist.NegativeBinomial2(mean=jnp.exp(mu[obs_idx]), concentration=r)
        elif family == "zip":
            gate = extra_params["gate"]
            dist_imputed = dist.ZeroInflatedPoisson(gate=gate, rate=jnp.exp(mu[missing_idx]))
            dist_obs = dist.ZeroInflatedPoisson(gate=gate, rate=jnp.exp(mu[obs_idx]))
        elif family == "zinb":
            gate = extra_params["gate"]
            r = extra_params["r"]
            dist_imputed = dist.ZeroInflatedNegativeBinomial2(gate=gate, mean=jnp.exp(mu[missing_idx]), concentration=r)
            dist_obs = dist.ZeroInflatedNegativeBinomial2(gate=gate, mean=jnp.exp(mu[obs_idx]), concentration=r)
        elif family == "multinomial":
            dist_imputed = dist.Categorical(logits=mu[missing_idx])
            dist_obs = dist.Categorical(logits=mu[obs_idx])
        elif family == "ordinal":
            cutpoints = extra_params["cutpoints"]
            dist_imputed = dist.OrderedLogistic(predictor=mu[missing_idx], cutpoints=cutpoints)
            dist_obs = dist.OrderedLogistic(predictor=mu[obs_idx], cutpoints=cutpoints)
        else:
            raise ValueError(f"Family '{family}' is not supported.")

        if has_multiPhylo and not force_plate_obs:
            unobs = numpyro.sample(f"{var}_imputed", dist_imputed.to_event(1))
            numpyro.sample(f"{var}_observed", dist_obs.to_event(1), obs=obs_data[obs_idx])
        else:
            with numpyro.plate(f"{var}_imputed_plate", missing_idx.shape[0]):
                unobs = numpyro.sample(f"{var}_imputed", dist_imputed)
            with numpyro.plate(f"{var}_observed_plate", obs_idx.shape[0]):
                numpyro.sample(f"{var}_observed", dist_obs, obs=obs_data[obs_idx])

        sampled_var = jnp.where(jnp.isnan(obs_data), 0.0, obs_data)
        sampled_var = sampled_var.at[missing_idx].set(unobs)
    else:
        if target_size is not None:
            if has_multiPhylo and not force_plate_obs:
                sampled_var = numpyro.sample(
                    var,
                    distribution.expand([target_size]).to_event(1),
                    obs=obs_data
                )
            else:
                with numpyro.plate(f"{var}_plate", target_size):
                    sampled_var = numpyro.sample(var, distribution, obs=obs_data)
        else:
            sampled_var = numpyro.sample(var, distribution, obs=obs_data)

    return sampled_var
