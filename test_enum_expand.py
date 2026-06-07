import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.funsor import config_enumerate

def model():
    n_trees = 2
    N = 3
    eigvals_all = jnp.array([[1., 2., 3.], [4., 5., 6.]]).T  
    eigvecs_all = jnp.zeros((3, 3, 2))
    for i in range(3):
        eigvecs_all = eigvecs_all.at[i, i, :].set(1.0)
    
    K = numpyro.sample("K", dist.Categorical(probs=jnp.ones(n_trees)/n_trees), infer={'enumerate': 'parallel'})
    
    eigvals_t = jnp.moveaxis(eigvals_all, -1, 0)
    eigvecs_t = jnp.moveaxis(eigvecs_all, -1, 0)
    
    eigvals = eigvals_t[K] 
    eigvecs = eigvecs_t[K]
    
    z_raw = numpyro.sample("z_raw", dist.Normal(0, 1).expand([N]))
    z_scaled = z_raw * jnp.sqrt(eigvals)
    z_group = jnp.einsum('...ij,...j->...i', eigvecs, z_scaled)
    
    numpyro.sample("y", dist.Normal(z_group, 1), obs=jnp.array([1., 2., 3.]))

model_enum = config_enumerate(model)
mcmc = MCMC(NUTS(model_enum), num_warmup=10, num_samples=10)
try:
    mcmc.run(jax.random.PRNGKey(0))
    mcmc.print_summary()
except Exception as e:
    print("MCMC error:", e)

