import warnings
import numpy as np
from .env import get_jax_primitives

def run_mcmc(
    model_func,
    jax_data,
    rng_key,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    thinning=1,
    n_cores=1,
    adapt_delta=0.95,
    max_treedepth=10,
    cor_matrices=None,
    quiet=False,
):
    """
    Sets up and executes standard NumPyro MCMC (NUTS / DiscreteHMCGibbs).
    Dispatches to parallel or sequential chains based on local JAX devices.
    """
    prim = get_jax_primitives()
    jax = prim["jax"]
    jnp = prim["jnp"]
    numpyro = prim["numpyro"]
    MCMC = prim["MCMC"]
    NUTS = prim["NUTS"]
    DiscreteHMCGibbs = prim["DiscreteHMCGibbs"]

    rng_key, subkey = jax.random.split(rng_key)
    kernel_base = NUTS(
        model_func,
        target_accept_prob=adapt_delta,
        max_tree_depth=max_treedepth,
        init_strategy=numpyro.infer.init_to_median(),
    )

    if cor_matrices and any(isinstance(v, dict) and v.get("type") == "multiPhylo" for v in cor_matrices.values()):
        n_trees = int(jax_data.get("Ntree", 10))
        def rw_fn(rng_key_rw, discrete_sites):
            new_sites = {}
            for k, v in discrete_sites.items():
                if k.startswith("K_tree"):
                    new_sites[k] = jax.random.randint(rng_key_rw, shape=jnp.shape(v), minval=0, maxval=n_trees)
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
                f"  [NumPyro MCMC] WARNING: Only {avail_devices} XLA CPU device(s) active in this session "
                f"(requested n.cores={n_cores} for {num_chains} chains). "
                f"Falling back to sequential chains. Please restart R / Python once "
                f"so JAX can allocate multiple CPU cores in parallel."
            )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*There are not enough devices.*")
        mcmc = MCMC(
            kernel_base,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            thinning=thinning,
            chain_method=c_method,
            progress_bar=not quiet,
        )

    mcmc.run(subkey, **jax_data)
    samples = mcmc.get_samples(group_by_chain=True)
    numpy_samples = {k: np.asarray(v) for k, v in samples.items()}

    return rng_key, mcmc, numpy_samples
