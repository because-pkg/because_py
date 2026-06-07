import json
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from because.builder.compiler import NumPyroBuilder
from because.builder.graph import CausalGraph
from because.builder.parser import FormulaParser

# Just a simple negative binomial model to test if it's the distribution
def simple_negbinom_model(N, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 10))
    r = numpyro.sample("r", dist.HalfNormal(10))
    numpyro.sample("obs", dist.NegativeBinomial2(mean=jnp.exp(mu)*jnp.ones(N), concentration=r), obs=y)

print("Running MCMC...")
mcmc = MCMC(NUTS(simple_negbinom_model), num_warmup=10, num_samples=10, num_chains=1)
mcmc.run(jax.random.PRNGKey(0), N=1000, y=jnp.zeros(1000))
print("Done!")
