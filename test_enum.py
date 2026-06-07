import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model():
    n_trees = 2
    N = 3
    eigvals_all = jnp.array([[1., 2., 3.], [4., 5., 6.]]).T  
    
    K = numpyro.sample("K", dist.Categorical(probs=jnp.ones(n_trees)/n_trees), infer={'enumerate': 'parallel'})
    
    eigvals_t = eigvals_all.T  
    eigvals = eigvals_t[K]     
    
    with numpyro.plate("N", N):
        z_raw = numpyro.sample("z_raw", dist.Normal(0, 1))
        numpyro.sample("y", dist.Normal(z_raw * eigvals, 1), obs=jnp.array([1., 2., 3.]))

mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
mcmc.run(jax.random.PRNGKey(0))
mcmc.print_summary()
