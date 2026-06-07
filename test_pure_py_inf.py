import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist

def simple_negbinom_model(N, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 10))
    r = numpyro.sample("r", dist.Gamma(2.0, 0.1))
    
    # Intentionally overflow
    mu_huge = mu + 1000.0
    
    numpyro.sample("obs", dist.NegativeBinomial2(mean=jnp.exp(mu_huge)*jnp.ones(N), concentration=r), obs=y)

print("Running MCMC...")
mcmc = MCMC(NUTS(simple_negbinom_model), num_warmup=10, num_samples=10, num_chains=1)
mcmc.run(jax.random.PRNGKey(0), N=1000, y=jnp.zeros(1000))
print("Done!")
