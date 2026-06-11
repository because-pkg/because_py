import re

class NumPyroBuilder:
    """
    Takes a parsed causal graph and generates the dynamic NumPyro model via a closure.
    """
    def __init__(self, causal_graph, family_dict=None, deterministic_terms=None, cor_matrices=None, fix_latent="loading", induced_correlations=None):
        """
        :param causal_graph: The compiled CausalGraph object.
        :param family_dict: Dictionary specifying families (e.g. {"y": "poisson"}). Default is "gaussian".
        :param deterministic_terms: Dictionary of deterministic terms.
        :param cor_matrices: Dictionary mapping grouping variable names to their covariance/correlation matrices.
        :param fix_latent: Method for anchoring latents: 'loading' or 'sign'.
        :param induced_correlations: List of variable-pair lists/tuples for MAG induced correlations.
            e.g. [["Brain", "Mass"]] means Brain and Mass share a correlated residual.
        """
        self.graph = causal_graph
        self.parsed_equations = {eq["response"]: eq for eq in causal_graph.parsed_equations.values() if eq["response"]}
        self.family_dict = family_dict or {}
        self.deterministic_terms = deterministic_terms or {}
        self.cor_matrices = cor_matrices or {}
        self.fix_latent = fix_latent
        self.pinned_latents = set()
        # Normalise to list-of-lists (R may pass as list of character vectors)
        self.induced_correlations = [list(pair) for pair in induced_correlations] if induced_correlations else []

    def generate_model_function(self, data_for_compilation=None, force_plate_obs=False):
        """
        Returns a callable python function representing the NumPyro model.
        :param data_for_compilation: Optional dict of the data to statically optimize the JAX trace (e.g. for NaN masking).
        :param force_plate_obs: If True, forces the use of numpyro.plate for observations even if multiPhylo is detected. Used for WAIC calculation.
        """
        topo_order = self.graph.get_topological_order()
        # self.parsed_equations is already a dict mapping response -> parsed dict
        equations_dict = self.parsed_equations
        families = self.family_dict
        det_terms = self.deterministic_terms
        cor_mats = self.cor_matrices
        
        # Precompute cholesky factors for efficiency
        import jax.numpy as jnp
        import jax
        L_matrices = {}
        custom_transforms = {}
        for k, mat in cor_mats.items():
            import numpy as np
            if isinstance(mat, dict) and "matrix" in mat and "transform_func" in mat:
                custom_transforms[k] = mat
                continue
            mat = np.array(mat, dtype=float)
            L_matrices[k] = jax.scipy.linalg.cholesky(mat, lower=True)
        
        # Check if any custom transform is multiPhylo - affects sampling strategy
        has_multiPhylo = any(
            isinstance(v, dict) and v.get("type") == "multiPhylo"
            for v in cor_mats.values()
        )
        
        # Capture induced correlations for closure
        induced_correlations_closure = self.induced_correlations
        
        def numpyro_model(**kwargs):
            data = kwargs.copy()
            import numpyro
            import numpyro.distributions as dist
            import jax.numpy as jnp
            import numpy as np
            import jax
            
            computed_vars = dict(data)
            
            # Infer dataset size N
            N = len(next(iter(data.values()))) if data else 1
            
            shared_state = {}
            local_pinned_latents = set()
            
            # --- MAG Induced Correlations (latent_method = "correlation") ---
            # JAGS-correct formulation: variables in induced correlations have NO
            # separate sigma.  Their variance is entirely captured by the MVN.
            # Phase 1 (here): sample LKJ correlation + scale parameters and store
            #   the resulting covariance matrices.
            # Phase 2 (after variable loop): score (obs - structural_mu) residuals
            #   jointly against each pair's bivariate normal.
            #
            # This mirrors JAGS exactly:
            #   err_res[i,1:2] ~ dmnorm(0, TAU_res)  <- the only likelihood for var1/var2
            #   var1[i] ~ dnorm(mu_var1[i], <skipped>) <- standard sigma block is skipped
            induced_mvn_storage = {}   # pair_tag -> {cov_mat, var1, var2}
            induced_cor_set    = set() # all variables involved in any induced correlation
            induced_mus        = {}    # var -> mu array (filled during variable loop)
            ind_cor_pairs = induced_correlations_closure
            for pair in ind_cor_pairs:
                var1, var2 = pair[0], pair[1]
                pair_tag = f"{var1}_{var2}"
                induced_cor_set.add(var1)
                induced_cor_set.add(var2)
                
                # --- LKJ prior on 2x2 correlation matrix (equivalent to JAGS dwish) ---
                L_corr = numpyro.sample(
                    f"L_corr_{pair_tag}",
                    dist.LKJCholesky(2, concentration=1.0)
                )
                corr_mat = L_corr @ L_corr.T
                
                # --- Scale parameters (marginal residual SDs for each variable) ---
                sigma_res1 = numpyro.sample(f"sigma_res_{var1}_{var2}", dist.HalfNormal(1.0))
                sigma_res2 = numpyro.sample(f"sigma_res_{var2}_{var1}", dist.HalfNormal(1.0))
                scale_diag = jnp.array([sigma_res1, sigma_res2])
                cov_mat    = jnp.outer(scale_diag, scale_diag) * corr_mat
                
                # --- Deterministic rho (same meaning as JAGS rho_var1_var2) ---
                numpyro.deterministic(f"rho_{pair_tag}", corr_mat[1, 0])
                
                induced_mvn_storage[pair_tag] = {"cov_mat": cov_mat, "var1": var1, "var2": var2}
            
            for var in topo_order:
                # 1. Deterministic Node Evaluation
                if var in det_terms:
                    expr = det_terms[var]["expression"]
                    target_size = None
                    if f"N_{var}" in data:
                        target_size = int(data[f"N_{var}"])
                    else:
                        # Find the parent node that uses this deterministic term
                        for p_var, eq in equations_dict.items():
                            if var in eq.get("fixed", []) or var in eq.get("random", []):
                                if p_var in data:
                                    target_size = data[p_var].shape[0]
                                elif f"N_{p_var}" in data:
                                    target_size = int(data[f"N_{p_var}"])
                                break
                    
                    if target_size is not None:
                        import ast
                        used_vars = [node.id for node in ast.walk(ast.parse(expr)) if isinstance(node, ast.Name)]
                        used_vars = [v for v in used_vars if v in computed_vars]
                        
                        local_vars = dict(computed_vars)
                        for v in used_vars:
                            v_data = computed_vars[v]
                            if hasattr(v_data, "shape") and len(v_data.shape) > 0 and v_data.shape[0] < target_size:
                                v_size = v_data.shape[0]
                                
                                # Robust index finding for criss-crossing hierarchies
                                # We need an index array of length target_size with max value < v_size.
                                # To disambiguate, we find the level names.
                                def get_level_name(size):
                                    candidates = []
                                    for k, val in data.items():
                                        if k.startswith("N_") and int(val) == size:
                                            lname = k[2:]
                                            if not lname.isupper() and "ID" not in lname and "_id" not in lname:
                                                candidates.append(lname)
                                    return candidates
                                
                                source_levels = get_level_name(v_size)
                                target_levels = get_level_name(target_size)
                                
                                idx_array = None
                                for s in source_levels:
                                    for t in target_levels:
                                        idx_name = f"{s}_idx_{t}"
                                        if idx_name in data:
                                            idx_array = data[idx_name]
                                            break
                                    if idx_array is not None: break
                                
                                if idx_array is None:
                                    # Fallback: any valid index array
                                    for k, val in data.items():
                                        if "idx" in k and hasattr(val, "shape") and len(val.shape) > 0 and val.shape[0] == target_size:
                                            if jnp.max(val) < v_size:
                                                idx_array = val
                                                break
                                                
                                if idx_array is not None:
                                    local_vars[v] = v_data[idx_array]
                                else:
                                    raise ValueError(f"Shape mismatch in deterministic node {var}: {v} ({v_size}) vs target ({target_size}). No bridging index found.")
                        
                        computed_vars[var] = eval(expr, {"jnp": jnp, "np": np}, local_vars)
                    else:
                        computed_vars[var] = eval(expr, {"jnp": jnp, "np": np}, computed_vars)
                    continue
                
                eq = equations_dict.get(var)
                family = families.get(var, "gaussian").lower()
                
                is_latent = var not in data
                
                # Check if it's an exogenous latent variable (no equation, not in data)
                if is_latent and not eq:
                    # It's an exogenous latent variable, sample from standard normal
                    computed_vars[var] = numpyro.sample(var, dist.Normal(0, 1).expand([N]))
                    continue
                    
                target_size = None
                obs_data = data.get(var, None)
                if obs_data is not None:
                    target_size = obs_data.shape[0]
                elif eq:
                    # Try to infer target size from predictors
                    for pred in eq["fixed"]:
                        if pred in computed_vars:
                            target_size = computed_vars[pred].shape[0]
                            break
                            
                if target_size is None:
                    target_size = N # fallback
                    
                if not eq:
                    # Exogenous observed variable, just store it
                    computed_vars[var] = obs_data
                    continue
                    
                # --- Endogenous Variable ---
                
                # --- Fixed Effects ---
                mu = jnp.zeros(target_size)
                if eq["intercept"]:
                    alpha = numpyro.sample(f"alpha_{var}", dist.Normal(0, 10))
                    mu = mu + alpha
                    
                for pred in eq["fixed"]:
                    # Latent variable identification logic
                    is_pred_latent = pred not in data and pred not in det_terms
                    if is_pred_latent and pred not in local_pinned_latents:
                        if self.fix_latent == "loading":
                            beta = 1.0
                        elif self.fix_latent == "sign":
                            beta = numpyro.sample(f"beta_{var}_{pred}", dist.TruncatedNormal(loc=0.0, scale=10.0, low=0.0))
                        else:
                            beta = numpyro.sample(f"beta_{var}_{pred}", dist.Normal(0, 10))
                        local_pinned_latents.add(pred)
                    else:
                        beta = numpyro.sample(f"beta_{var}_{pred}", dist.Normal(0, 10))
                    
                    pred_data = computed_vars[pred]
                    
                    if pred_data.shape[0] != target_size:
                        max_len = max(pred_data.shape[0], target_size)
                        min_len = min(pred_data.shape[0], target_size)
                        
                        def get_level_name(size):
                            candidates = []
                            for k, val in data.items():
                                if k.startswith("N_") and int(val) == size:
                                    lname = k[2:]
                                    if not lname.isupper() and "ID" not in lname and "_id" not in lname:
                                        candidates.append(lname)
                            return candidates
                            
                        source_levels = get_level_name(min_len)
                        target_levels = get_level_name(max_len)
                        
                        idx_array = None
                        for s in source_levels:
                            for t in target_levels:
                                idx_name = f"{s}_idx_{t}"
                                if idx_name in data:
                                    idx_array = data[idx_name]
                                    idx_name_for_print = idx_name
                                    break
                            if idx_array is not None: break
                            
                        if idx_array is None:
                            idx_name_1 = f"idx_{pred}"
                            idx_array = data.get(idx_name_1)
                            idx_name_for_print = idx_name_1
                            if idx_array is None or idx_array.shape[0] != max_len:
                                possible_idx = [k for k, val in data.items() if k.startswith("idx_") and hasattr(val, "shape") and len(val.shape) > 0 and val.shape[0] == max_len and jnp.max(val) < min_len]
                                if not possible_idx:
                                    possible_idx = [k for k, val in data.items() if "idx" in k and hasattr(val, "shape") and len(val.shape) > 0 and val.shape[0] == max_len and jnp.max(val) < min_len]
                                    if not possible_idx:
                                        raise ValueError(f"Shape mismatch: {var} ({target_size}) vs {pred} ({pred_data.shape[0]}). No bridging index found of length {max_len}.")
                                idx_array = data[possible_idx[0]]
                                idx_name_for_print = possible_idx[0]
                        
                        if pred_data.shape[0] < target_size:
                            # COARSE predictor -> FINE response (Broadcasting)
                            mu = mu + beta * pred_data[idx_array]
                        else:
                            # FINE predictor -> COARSE response (Resolution Locking)
                            import jax
                            sum_pred = jax.ops.segment_sum(pred_data, idx_array, num_segments=target_size)
                            count = jax.ops.segment_sum(jnp.ones_like(pred_data), idx_array, num_segments=target_size)
                            mean_pred = sum_pred / jnp.where(count > 0, count, 1.0)
                            mu = mu + beta * mean_pred
                    else:
                        mu = mu + beta * pred_data
                    
                # Dictionary to store structural standard deviations for lambda calculation
                sigma_struct_dict = {}
                
                for rand_term in eq["random"]:
                    import re
                    match = re.search(r"\(1\s*\|\s*([^)]+)\)", rand_term)
                    if not match:
                        continue
                    group_name = match.group(1).strip()
                    if group_name not in data:
                        raise ValueError(f"Grouping variable '{group_name}' not found in data.")
                        
                    group_idx = data[group_name]
                    
                    # Broadcasting for random effects (COARSE group -> FINE response)
                    if hasattr(group_idx, "shape") and len(group_idx.shape) > 0 and target_size is not None and group_idx.shape[0] < target_size:
                        possible_idx = [k for k in data.keys() if "idx" in k and hasattr(data[k], "shape") and len(data[k].shape) > 0 and data[k].shape[0] == target_size]
                        
                        # Try to find the exact match using the base level name if possible
                        level_name = group_name
                        if hasattr(self.graph, 'levels') and self.graph.levels:
                            for lvl, vars_in_lvl in self.graph.levels.items():
                                if group_name in vars_in_lvl:
                                    level_name = lvl
                                    break
                                    
                        # If it's a custom structure (e.g., phylo), it might not be in levels.
                        # But we can figure out its level by matching its required size.
                        expected_size = None
                        if f"N_{group_name}" in data:
                            expected_size = int(data[f"N_{group_name}"])
                        elif group_name in custom_transforms:
                            expected_size = len(custom_transforms[group_name]["matrix"])
                            
                        if expected_size is not None:
                            # Find which level has this size
                            for k, v in data.items():
                                if k.startswith("N_") and not k.endswith("ID") and not k.endswith("_id") and isinstance(v, (int, float, np.integer, np.floating)):
                                    if int(v) == expected_size:
                                        potential_lvl = k[2:]
                                        # Check if this potential_lvl has an index in possible_idx
                                        if any(idx.startswith(f"{potential_lvl}_idx_") for idx in possible_idx):
                                            level_name = potential_lvl
                                            break

                        best_idx = None
                        for k in possible_idx:
                            if k.startswith(f"{level_name}_idx_"):
                                best_idx = k
                                break
                        
                        if best_idx:
                            group_idx = data[best_idx]
                        elif possible_idx:
                            group_idx = data[possible_idx[0]]
                        else:
                            raise ValueError(f"Shape mismatch for random effect: {var} ({target_size}) vs {group_name} ({group_idx.shape[0]}). No bridging index found.")
                            
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
                    
                    sigma_group = numpyro.sample(f"sigma_{var}_{group_name}", dist.HalfNormal(5))
                    with numpyro.plate(f"{var}_{group_name}_plate", num_groups):
                        z_group_raw = numpyro.sample(f"z_{var}_{group_name}_raw", dist.Normal(0, 1))
                    
                    if group_name in custom_transforms:
                        z_group, sigma_group = custom_transforms[group_name]["transform_func"](numpyro, jnp, jax, dist, var, group_name, num_groups, custom_transforms[group_name]["matrix"], z_group_raw, sigma_group, shared_state)
                    elif group_name in L_matrices:
                        # Correlated errors: z_group = L @ z_group_raw
                        L = L_matrices[group_name]
                        if L.shape[0] != num_groups:
                            raise ValueError(f"Correlation matrix for '{group_name}' has shape {L.shape} but num_groups is {num_groups}")
                        z_group = jnp.dot(L, z_group_raw)
                        # Register the correlated z_group for visibility
                        z_group = numpyro.deterministic(f"z_{var}_{group_name}", z_group)
                    else:
                        # Independent random effects
                        z_group = z_group_raw
                        # Rename deterministic variable for trace parity
                        z_group = numpyro.deterministic(f"z_{var}_{group_name}", z_group)
                        
                    u_group = numpyro.deterministic(f"u_{var}_{group_name}", z_group * sigma_group)
                    mu = mu + u_group[group_idx]
                    
                    # Store for lambda calculation
                    if group_name in custom_transforms or group_name in L_matrices:
                        sigma_struct_dict[group_name] = sigma_group
                
                # --- MAG: if var is in an induced correlation pair, skip sigma +
                # likelihood entirely.  Store mu so the post-loop block can compute
                # the residual (obs - mu) and score it against the pair's MVN.
                if var in induced_cor_set and obs_data is not None and not is_latent:
                    induced_mus[var] = mu
                    # Downstream nodes see the actual observations, not a sampled value
                    computed_vars[var] = obs_data
                    continue  # skip sigma sampling and Normal likelihood
                
                # --- Distribution Dispatcher ---
                if family == "gaussian":
                    sigma = numpyro.sample(f"sigma_{var}", dist.HalfNormal(5))
                    distribution = dist.Normal(mu, sigma)
                    
                    # Post-hoc calculate Pagel's lambda for any structural/phylogenetic random effects
                    for g_name, sig_g in sigma_struct_dict.items():
                        numpyro.deterministic(f"lambda_{var}_{g_name}", (sig_g**2) / (sig_g**2 + sigma**2))
                elif family == "poisson":
                    distribution = dist.Poisson(rate=jnp.exp(mu))
                elif family == "binomial":
                    distribution = dist.Bernoulli(logits=mu)
                elif family == "negbinomial":
                    r = numpyro.sample(f"r_{var}", dist.Gamma(2.0, 0.1))
                    distribution = dist.NegativeBinomial2(mean=jnp.exp(mu), concentration=r)
                elif family == "zip":
                    psi_logit = numpyro.sample(f"psi_{var}", dist.Normal(0, 2))
                    gate = jnp.exp(psi_logit) / (1 + jnp.exp(psi_logit))
                    distribution = dist.ZeroInflatedPoisson(gate=gate, rate=jnp.exp(mu))
                elif family == "zinb":
                    psi_logit = numpyro.sample(f"psi_{var}", dist.Normal(0, 2))
                    gate = jnp.exp(psi_logit) / (1 + jnp.exp(psi_logit))
                    r = numpyro.sample(f"r_{var}", dist.Gamma(2.0, 0.1))
                    distribution = dist.ZeroInflatedNegativeBinomial2(gate=gate, mean=jnp.exp(mu), concentration=r)
                elif family == "multinomial":
                    distribution = dist.Categorical(logits=mu)
                elif family == "ordinal":
                    num_categories = len(jnp.unique(obs_data)) if obs_data is not None else 3
                    c1 = numpyro.sample(f"c1_{var}", dist.Normal(0, 5))
                    increments = numpyro.sample(f"c_inc_{var}", dist.Exponential(1).expand([num_categories - 2]))
                    cutpoints = jnp.concatenate([jnp.array([c1]), c1 + jnp.cumsum(increments)])
                    distribution = dist.OrderedLogistic(predictor=mu, cutpoints=cutpoints)
                else:
                    raise ValueError(f"Family '{family}' is not supported.")
                
                if is_latent:
                    # Endogenous latent variable: sample it from the model, no observations
                    computed_vars[var] = numpyro.sample(var, distribution)
                    continue
                
                # ----------------------------------------------------
                # Missing Data Imputation & Likelihood Scoring
                # ----------------------------------------------------
                has_nans = False
                if data_for_compilation is not None and var in data_for_compilation:
                    import numpy as np
                    np_data = np.asarray(data_for_compilation[var])
                    has_nans = bool(np.isnan(np_data).any())
                    if has_nans:
                        # Pre-compute static indices of missing and observed data
                        missing_idx = jnp.array(np.where(np.isnan(np_data))[0])
                        obs_idx = jnp.array(np.where(~np.isnan(np_data))[0])
                    
                if obs_data is not None and has_nans:
                    # We sample ONLY the missing values using the subsetted parameters
                    if family == "gaussian":
                        dist_imputed = dist.Normal(mu[missing_idx], sigma)
                        dist_obs = dist.Normal(mu[obs_idx], sigma)
                    elif family == "poisson":
                        dist_imputed = dist.Poisson(rate=jnp.exp(mu[missing_idx]))
                        dist_obs = dist.Poisson(rate=jnp.exp(mu[obs_idx]))
                    elif family == "binomial":
                        dist_imputed = dist.Bernoulli(logits=mu[missing_idx])
                        dist_obs = dist.Bernoulli(logits=mu[obs_idx])
                    elif family == "negbinomial":
                        dist_imputed = dist.NegativeBinomial2(mean=jnp.exp(mu[missing_idx]), concentration=r)
                        dist_obs = dist.NegativeBinomial2(mean=jnp.exp(mu[obs_idx]), concentration=r)
                    elif family == "zip":
                        dist_imputed = dist.ZeroInflatedPoisson(gate=gate, rate=jnp.exp(mu[missing_idx]))
                        dist_obs = dist.ZeroInflatedPoisson(gate=gate, rate=jnp.exp(mu[obs_idx]))
                    elif family == "zinb":
                        dist_imputed = dist.ZeroInflatedNegativeBinomial2(gate=gate, mean=jnp.exp(mu[missing_idx]), concentration=r)
                        dist_obs = dist.ZeroInflatedNegativeBinomial2(gate=gate, mean=jnp.exp(mu[obs_idx]), concentration=r)
                    elif family == "multinomial":
                        dist_imputed = dist.Categorical(logits=mu[missing_idx])
                        dist_obs = dist.Categorical(logits=mu[obs_idx])
                    elif family == "ordinal":
                        dist_imputed = dist.OrderedLogistic(predictor=mu[missing_idx], cutpoints=cutpoints)
                        dist_obs = dist.OrderedLogistic(predictor=mu[obs_idx], cutpoints=cutpoints)
                    else:
                        raise ValueError(f"Family '{family}' is not supported.")
                    
                    if has_multiPhylo and not force_plate_obs:
                        # multiPhylo + NaN path:
                        # DiscreteHMCGibbs requires a scalar log_prob (to_event(1)).
                        # We sample imputed values as free latent variables (no obs=),
                        # and score only the observed rows with to_event(1).
                        # This correctly marginalizes out the missing values while
                        # keeping the discrete tree index integrable by Gibbs.
                        unobs = numpyro.sample(
                            f"{var}_imputed",
                            dist_imputed.to_event(1)   # latent, no obs constraint
                        )
                        numpyro.sample(
                            f"{var}_observed",
                            dist_obs.to_event(1),      # scalar log_prob for Gibbs
                            obs=obs_data[obs_idx]
                        )
                    else:
                        # Standard NaN path: two independent plates
                        with numpyro.plate(f"{var}_imputed_plate", missing_idx.shape[0]):
                            unobs = numpyro.sample(f"{var}_imputed", dist_imputed)
                        with numpyro.plate(f"{var}_observed_plate", obs_idx.shape[0]):
                            numpyro.sample(f"{var}_observed", dist_obs, obs=obs_data[obs_idx])
                    
                    # Merge observed + imputed into a complete vector for downstream nodes
                    sampled_var = jnp.where(jnp.isnan(obs_data), 0.0, obs_data)
                    sampled_var = sampled_var.at[missing_idx].set(unobs)
                else:
                    target_size = obs_data.shape[0] if obs_data is not None else None
                    if target_size is not None:
                        if has_multiPhylo and not force_plate_obs:
                            # Use to_event(1) instead of plate: log_prob reduces to a scalar,
                            # which is required for DiscreteHMCGibbs exact Gibbs (random_walk=False)
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
                    
                computed_vars[var] = sampled_var
            
            # ----------------------------------------------------------------
            # MAG Induced Correlations - Phase 2: score residuals against MVN
            # ----------------------------------------------------------------
            # For each pair, the JAGS-equivalent likelihood is:
            #   (obs_var1[i] - mu_var1[i], obs_var2[i] - mu_var2[i]) ~ MVN(0, Sigma)
            # This is the ONLY variance source for induced-correlation variables.
            for pair_tag, pair_info in induced_mvn_storage.items():
                var1     = pair_info["var1"]
                var2     = pair_info["var2"]
                cov_mat  = pair_info["cov_mat"]
                
                mu1  = induced_mus.get(var1, jnp.zeros(N))
                mu2  = induced_mus.get(var2, jnp.zeros(N))
                obs1 = data.get(var1)
                obs2 = data.get(var2)
                
                if obs1 is not None and obs2 is not None:
                    resid1      = obs1 - mu1                            # shape (N,)
                    resid2      = obs2 - mu2                            # shape (N,)
                    joint_resid = jnp.stack([resid1, resid2], axis=-1) # shape (N, 2)
                    
                    mvn = dist.MultivariateNormal(
                        loc=jnp.zeros(2),
                        covariance_matrix=cov_mat
                    )
                    if has_multiPhylo and not force_plate_obs:
                        # DiscreteHMCGibbs needs a scalar log_prob: use to_event
                        numpyro.sample(
                            f"resid_{pair_tag}",
                            mvn.expand([N]).to_event(1),
                            obs=joint_resid
                        )
                    else:
                        with numpyro.plate(f"resid_{pair_tag}_plate", N):
                            numpyro.sample(f"resid_{pair_tag}", mvn, obs=joint_resid)
        
        return numpyro_model

    # ------------------------------------------------------------------
    # Code-string generator
    # ------------------------------------------------------------------
    def generate_model_code_string(self, data_shapes=None):
        """
        Returns a self-contained, executable Python source string that
        represents the NumPyro model built by generate_model_function().

        The output is intended for inspection, auditing, or copy-paste
        modification in a standalone Python script.  Covariance / Cholesky
        matrices are referenced as arguments so the function signature
        matches what because-py passes internally.

        :param data_shapes: Optional dict mapping variable names to their
            shapes (e.g. {\"Brain\": (678,), \"phy\": (678, 678)}).  Used to
            annotate the function signature with dimension comments.
        :return: str — a Python source string.
        """
        topo_order    = self.graph.get_topological_order()
        equations_dict = self.parsed_equations
        families      = self.family_dict
        det_terms     = self.deterministic_terms
        cor_mats      = self.cor_matrices

        lines = []
        lines.append("# Auto-generated NumPyro model from because-py")
        lines.append("# -----------------------------------------------")
        lines.append("# This script is provided for inspection / customisation.")
        lines.append("# Covariance matrices and data arrays must be supplied")
        lines.append("# as keyword arguments when calling numpyro_model().")
        lines.append("")
        lines.append("import jax")
        lines.append("import jax.numpy as jnp")
        lines.append("import numpy as np")
        lines.append("import numpyro")
        lines.append("import numpyro.distributions as dist")
        lines.append("")

        # Build function signature
        all_vars = list(topo_order)
        struct_args = [f"L_{k}=None" for k in cor_mats.keys()]
        sig_parts = all_vars + struct_args + ["**kwargs"]
        lines.append(f"def numpyro_model({', '.join(sig_parts)}):")
        lines.append("    \"\"\"NumPyro causal model — generated by because.\"\"\"")
        lines.append("    computed_vars = {k: v for k, v in locals().items() if v is not None}")
        lines.append("    N = len(next(v for v in [" +
                     ", ".join(all_vars) + "] if v is not None))")
        lines.append("")

        def ind(n=1):
            return "    " * n

        pinned = set()

        for var in topo_order:
            # Deterministic node
            if var in det_terms:
                expr = det_terms[var]["expression"]
                lines.append(f"{ind()}# --- {var} [deterministic] ---")
                lines.append(f"{ind()}computed_vars['{var}'] = {expr}")
                lines.append("")
                continue

            eq     = equations_dict.get(var)
            family = families.get(var, "gaussian").lower()
            is_latent = True  # assume; observed status embedded in obs= arg

            lines.append(f"{ind()}# --- {var}  (family: {family}) ---")

            if not eq:
                lines.append(f"{ind()}computed_vars['{var}'] = {var}  # exogenous observed")
                lines.append("")
                continue

            lines.append(f"{ind()}mu_{var} = jnp.zeros(N)")

            # Intercept
            if eq.get("intercept", True):
                lines.append(f"{ind()}alpha_{var} = numpyro.sample('alpha_{var}', dist.Normal(0, 10))")
                lines.append(f"{ind()}mu_{var} = mu_{var} + alpha_{var}")

            # Fixed effects
            for pred in eq.get("fixed", []):
                is_pred_latent = pred not in [v for v in topo_order if equations_dict.get(v) is None]
                if is_pred_latent and pred not in pinned and self.fix_latent == "loading":
                    lines.append(f"{ind()}# beta_{var}_{pred} fixed to 1.0 (latent loading anchor)")
                    lines.append(f"{ind()}mu_{var} = mu_{var} + 1.0 * computed_vars.get('{pred}', {pred})")
                    pinned.add(pred)
                else:
                    lines.append(f"{ind()}beta_{var}_{pred} = numpyro.sample('beta_{var}_{pred}', dist.Normal(0, 10))")
                    lines.append(f"{ind()}mu_{var} = mu_{var} + beta_{var}_{pred} * computed_vars.get('{pred}', {pred})")

            # Random effects
            sigma_struct_names = []
            for rand_term in eq.get("random", []):
                import re as _re
                match = _re.search(r"\(1\s*\|\s*([^)]+)\)", rand_term)
                if not match:
                    continue
                grp = match.group(1).strip()
                lines.append(f"{ind()}# Random effect: (1 | {grp})")
                lines.append(f"{ind()}sigma_{var}_{grp} = numpyro.sample('sigma_{var}_{grp}', dist.HalfNormal(5))")
                lines.append(f"{ind()}with numpyro.plate('{var}_{grp}_plate', N_{grp}):")
                lines.append(f"{ind(2)}z_{var}_{grp}_raw = numpyro.sample('z_{var}_{grp}_raw', dist.Normal(0, 1))")

                if grp in cor_mats:
                    mat_info = cor_mats[grp]
                    mat_type = mat_info.get("type", "corr") if isinstance(mat_info, dict) else "corr"
                    lines.append(f"{ind()}# {mat_type} transform for '{grp}'")
                    lines.append(f"{ind()}z_{var}_{grp} = numpyro.deterministic('z_{var}_{grp}', L_{grp} @ z_{var}_{grp}_raw)")
                else:
                    lines.append(f"{ind()}z_{var}_{grp} = numpyro.deterministic('z_{var}_{grp}', z_{var}_{grp}_raw)")

                lines.append(f"{ind()}u_{var}_{grp} = numpyro.deterministic('u_{var}_{grp}', z_{var}_{grp} * sigma_{var}_{grp})")
                lines.append(f"{ind()}mu_{var} = mu_{var} + u_{var}_{grp}[{grp}]")
                sigma_struct_names.append(grp)

            # Distribution
            lines.append(f"{ind()}# Likelihood")
            if family == "gaussian":
                lines.append(f"{ind()}sigma_{var} = numpyro.sample('sigma_{var}', dist.HalfNormal(5))")
                lines.append(f"{ind()}with numpyro.plate('{var}_plate', N):")
                lines.append(f"{ind(2)}numpyro.sample('{var}', dist.Normal(mu_{var}, sigma_{var}), obs={var})")
                for grp in sigma_struct_names:
                    lines.append(f"{ind()}numpyro.deterministic('lambda_{var}_{grp}',")
                    lines.append(f"{ind()}    sigma_{var}_{grp}**2 / (sigma_{var}_{grp}**2 + sigma_{var}**2))")
            elif family == "poisson":
                lines.append(f"{ind()}with numpyro.plate('{var}_plate', N):")
                lines.append(f"{ind(2)}numpyro.sample('{var}', dist.Poisson(rate=jnp.exp(mu_{var})), obs={var})")
            elif family == "binomial":
                lines.append(f"{ind()}with numpyro.plate('{var}_plate', N):")
                lines.append(f"{ind(2)}numpyro.sample('{var}', dist.Bernoulli(logits=mu_{var}), obs={var})")
            elif family == "negbinomial":
                lines.append(f"{ind()}r_{var} = numpyro.sample('r_{var}', dist.Gamma(2.0, 0.1))")
                lines.append(f"{ind()}with numpyro.plate('{var}_plate', N):")
                lines.append(f"{ind(2)}numpyro.sample('{var}', dist.NegativeBinomial2(")
                lines.append(f"{ind(3)}mean=jnp.exp(mu_{var}), concentration=r_{var}), obs={var})")
            else:
                lines.append(f"{ind()}# Family '{family}' — see compiler.py for full implementation")
                lines.append(f"{ind()}with numpyro.plate('{var}_plate', N):")
                lines.append(f"{ind(2)}numpyro.sample('{var}', dist.Normal(mu_{var}, 1.0), obs={var})  # PLACEHOLDER")

            lines.append(f"{ind()}computed_vars['{var}'] = mu_{var}")
            lines.append("")

        return "\n".join(lines)

