"""
Compiler module: translates parsed causal graph equations into executable
NumPyro probabilistic models.
"""

import numpy as np
from ..engine.env import get_jax_primitives

from .codegen import generate_model_code_string
from .components.variances import (
    setup_variance_partitioning,
    sample_induced_correlations,
)
from .components.multiscale import (
    evaluate_deterministic_node,
    bridge_predictor,
)
from .components.random_effects import (
    can_spectral_marginalize,
    sample_spectral_marginalization,
    sample_structural_random_effects,
)
from .components.likelihoods import (
    create_family_distribution,
    sample_response_likelihood,
)


class NumPyroBuilder:
    """
    Takes a parsed causal graph and generates the dynamic NumPyro model via a closure.
    """
    def __init__(
        self,
        causal_graph,
        family_dict=None,
        deterministic_terms=None,
        cor_matrices=None,
        fix_latent="loading",
        induced_correlations=None,
        prior_scale_fixed=None,
        levels=None,
    ):
        """
        :param causal_graph: The compiled CausalGraph object.
        :param family_dict: Dictionary specifying families (e.g. {"y": "poisson"}). Default is "gaussian".
        :param deterministic_terms: Dictionary of deterministic terms.
        :param cor_matrices: Dictionary mapping grouping variable names to their covariance/correlation matrices.
        :param fix_latent: Method for anchoring latents: 'loading' or 'sign'.
        :param induced_correlations: List of variable-pair lists/tuples for MAG induced correlations.
        :param prior_scale_fixed: Optional dictionary for fixing prior scales.
        :param levels: Optional dictionary mapping level names to variable/structure names.
        """
        self.graph = causal_graph
        self.parsed_equations = {
            eq["response"]: eq
            for eq in causal_graph.parsed_equations.values()
            if eq["response"]
        }
        self.family_dict = family_dict or {}
        self.deterministic_terms = deterministic_terms or {}
        self.cor_matrices = cor_matrices or {}
        self.fix_latent = fix_latent
        self.pinned_latents = set()
        self.prior_scale_fixed = prior_scale_fixed
        self.levels = levels or {}
        self.var_to_level = {}
        if isinstance(self.levels, dict):
            for lvl, vars_in_lvl in self.levels.items():
                if isinstance(vars_in_lvl, (list, tuple, set)):
                    for v in vars_in_lvl:
                        self.var_to_level[str(v)] = str(lvl)
                elif isinstance(vars_in_lvl, str):
                    self.var_to_level[vars_in_lvl] = str(lvl)

        self.induced_correlations = (
            [list(pair) for pair in induced_correlations]
            if induced_correlations else []
        )
        self.latent_method = "correlation" if self.induced_correlations else None

    def generate_model_function(self, data_for_compilation=None, force_plate_obs=False):
        """
        Returns a callable python function representing the NumPyro model.
        :param data_for_compilation: Optional dict of the data to statically optimize the JAX trace.
        :param force_plate_obs: If True, forces the use of numpyro.plate for observations even if multiPhylo is detected.
        """
        prim = get_jax_primitives()
        jax = prim["jax"]
        jnp = prim["jnp"]
        numpyro = prim["numpyro"]
        dist = prim["dist"]

        topo_order = self.graph.get_topological_order()
        equations_dict = self.parsed_equations
        families = self.family_dict
        det_terms = self.deterministic_terms
        cor_mats = self.cor_matrices

        # Precompute cholesky factors for efficiency
        L_matrices = {}
        custom_transforms = {}
        for k, mat in cor_mats.items():
            if isinstance(mat, dict) and "matrix" in mat:
                if isinstance(mat["matrix"], dict):
                    if "eigvals" in mat["matrix"] and not isinstance(mat["matrix"]["eigvals"], np.ndarray):
                        mat["matrix"]["eigvals"] = np.asarray(mat["matrix"]["eigvals"])
                    if "eigvecs" in mat["matrix"] and not isinstance(mat["matrix"]["eigvecs"], np.ndarray):
                        mat["matrix"]["eigvecs"] = np.asarray(mat["matrix"]["eigvecs"])
                if "transform_func" in mat:
                    custom_transforms[k] = mat
                    continue
            mat = np.array(mat, dtype=float)
            L_matrices[k] = jax.scipy.linalg.cholesky(mat, lower=True)

        has_multiPhylo = any(
            isinstance(v, dict) and v.get("type") == "multiPhylo"
            for v in cor_mats.values()
        )

        induced_correlations_closure = self.induced_correlations
        var_to_level = self.var_to_level
        fix_latent_closure = self.fix_latent
        prior_scale_fixed_closure = self.prior_scale_fixed
        graph_ref = self.graph

        def numpyro_model(**kwargs):
            data = kwargs.copy()
            computed_vars = dict(data)

            # Infer dataset size N from the first 1D+ array in data
            N = next((v.shape[0] for v in data.values() if hasattr(v, "shape") and len(v.shape) > 0), 1)

            shared_state = {}
            local_pinned_latents = set()

            # --- Induced Correlation Setup ---
            induced_cor_set = set()
            ind_cor_pairs = induced_correlations_closure
            for pair in ind_cor_pairs:
                induced_cor_set.add(pair[0])
                induced_cor_set.add(pair[1])

            # --- Universal Variance Component Partitioning ---
            var_variances = setup_variance_partitioning(
                topo_order=topo_order,
                families=families,
                equations_dict=equations_dict,
                induced_cor_set=induced_cor_set,
                numpyro=numpyro,
                dist=dist,
                jnp=jnp,
            )

            # --- Sample MAG Induced Correlations ---
            induced_err_storage = sample_induced_correlations(
                ind_cor_pairs=ind_cor_pairs,
                var_variances=var_variances,
                N=N,
                data=data,
                has_multiPhylo=has_multiPhylo,
                force_plate_obs=force_plate_obs,
                numpyro=numpyro,
                dist=dist,
                jnp=jnp,
            )

            # --- Traverse Causal Graph in Topological Order ---
            for var in topo_order:
                # 1. Deterministic Node Evaluation
                if var in det_terms:
                    res_det = evaluate_deterministic_node(
                        var=var,
                        det_terms=det_terms,
                        equations_dict=equations_dict,
                        data=data,
                        computed_vars=computed_vars,
                        var_to_level=var_to_level,
                        jnp=jnp,
                    )
                    if res_det is not None:
                        computed_vars[var] = res_det
                    continue

                eq = equations_dict.get(var)
                family = families.get(var, "gaussian").lower()
                is_latent = var not in data

                # Exogenous latent variable
                if is_latent and not eq:
                    computed_vars[var] = numpyro.sample(var, dist.Normal(0, 1).expand([N]))
                    continue

                obs_data = data.get(var, None)
                target_size = obs_data.shape[0] if obs_data is not None else None
                if target_size is None and eq:
                    for pred in eq["fixed"]:
                        if pred in computed_vars:
                            target_size = computed_vars[pred].shape[0]
                            break
                if target_size is None:
                    target_size = N

                if not eq:
                    if var not in induced_cor_set:
                        computed_vars[var] = obs_data
                        continue
                    else:
                        eq = {"intercept": True, "fixed": [], "random": []}

                # --- Fixed Effects ---
                mu = jnp.zeros(target_size)
                if prior_scale_fixed_closure is not None:
                    prior_scale = prior_scale_fixed_closure
                else:
                    prior_scale = 1.0 if family in [
                        "poisson", "binomial", "bernoulli", "occupancy",
                        "zip", "zinb", "multinomial", "ordinal"
                    ] else 2.5

                if eq["intercept"]:
                    alpha = numpyro.sample(f"alpha_{var}", dist.Normal(0, prior_scale))
                    mu = mu + alpha

                for pred in eq["fixed"]:
                    is_pred_latent = pred not in data and pred not in det_terms
                    if is_pred_latent and pred not in local_pinned_latents:
                        if fix_latent_closure == "loading":
                            beta = 1.0
                        elif fix_latent_closure == "sign":
                            beta = numpyro.sample(
                                f"beta_{var}_{pred}",
                                dist.TruncatedNormal(loc=0.0, scale=prior_scale, low=0.0)
                            )
                        else:
                            beta = numpyro.sample(f"beta_{var}_{pred}", dist.Normal(0, prior_scale))
                        local_pinned_latents.add(pred)
                    else:
                        beta = numpyro.sample(f"beta_{var}_{pred}", dist.Normal(0, prior_scale))

                    pred_data = computed_vars[pred]
                    aligned_pred = bridge_predictor(
                        pred=pred,
                        var=var,
                        pred_data=pred_data,
                        target_size=target_size,
                        data=data,
                        var_to_level=var_to_level,
                        jnp=jnp,
                        jax=jax,
                    )
                    mu = mu + beta * aligned_pred

                # Check for missing data (NaNs)
                has_nans = False
                missing_idx = None
                obs_idx = None
                if data_for_compilation is not None and var in data_for_compilation:
                    np_data = np.asarray(data_for_compilation[var])
                    has_nans = bool(np.isnan(np_data).any())
                    if has_nans:
                        missing_idx = jnp.array(np.where(np.isnan(np_data))[0])
                        obs_idx = jnp.array(np.where(~np.isnan(np_data))[0])

                # Spectral Gaussian Marginalization (multiPhylo dcat)
                eligible_spectral, spectral_grp = can_spectral_marginalize(
                    has_multiPhylo=has_multiPhylo,
                    family=family,
                    is_latent=is_latent,
                    obs_data=obs_data,
                    has_nans=has_nans,
                    eq=eq,
                    target_size=target_size,
                    custom_transforms=custom_transforms,
                )
                if eligible_spectral:
                    computed_vars[var] = sample_spectral_marginalization(
                        var=var,
                        spectral_group_name=spectral_grp,
                        obs_data=obs_data,
                        mu=mu,
                        data=data,
                        custom_transforms=custom_transforms,
                        var_variances=var_variances,
                        induced_cor_set=induced_cor_set,
                        induced_err_storage=induced_err_storage,
                        shared_state=shared_state,
                        force_plate_obs=force_plate_obs,
                        target_size=target_size,
                        numpyro=numpyro,
                        dist=dist,
                        jnp=jnp,
                    )
                    continue

                # Structural & Random Effects
                mu, sigma_struct_dict = sample_structural_random_effects(
                    var=var,
                    eq=eq,
                    family=family,
                    mu=mu,
                    target_size=target_size,
                    data=data,
                    var_to_level=var_to_level,
                    graph=graph_ref,
                    var_variances=var_variances,
                    custom_transforms=custom_transforms,
                    L_matrices=L_matrices,
                    shared_state=shared_state,
                    numpyro=numpyro,
                    dist=dist,
                    jnp=jnp,
                    jax=jax,
                )

                # Add latent induced correlation errors to structural mean
                if var in induced_err_storage:
                    for err in induced_err_storage[var]:
                        mu = mu + err

                # Distribution dispatcher
                distribution, extra_params = create_family_distribution(
                    var=var,
                    family=family,
                    mu=mu,
                    data=data,
                    var_variances=var_variances,
                    induced_cor_set=induced_cor_set,
                    sigma_struct_dict=sigma_struct_dict,
                    numpyro=numpyro,
                    dist=dist,
                    jnp=jnp,
                )

                if is_latent:
                    computed_vars[var] = numpyro.sample(var, distribution)
                    continue

                # Missing Data & Observation Scoring
                sampled_var = sample_response_likelihood(
                    var=var,
                    family=family,
                    distribution=distribution,
                    extra_params=extra_params,
                    mu=mu,
                    obs_data=obs_data,
                    target_size=target_size,
                    has_nans=has_nans,
                    missing_idx=missing_idx,
                    obs_idx=obs_idx,
                    has_multiPhylo=has_multiPhylo,
                    force_plate_obs=force_plate_obs,
                    numpyro=numpyro,
                    dist=dist,
                    jnp=jnp,
                )
                computed_vars[var] = sampled_var

        return numpyro_model

    def generate_model_code_string(self, data_shapes=None):
        """
        Returns a self-contained, executable Python source string that
        represents the NumPyro model built by generate_model_function().
        """
        return generate_model_code_string(self, data_shapes=data_shapes)
