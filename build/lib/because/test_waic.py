import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood

def model_to_event(y):
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    numpyro.sample("y", dist.Normal(mu, sigma).expand([y.shape[0]]).to_event(1), obs=y)

def model_plate(y):
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    with numpyro.plate("y_plate", y.shape[0]):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

y = jnp.array([1.0, 2.0, 3.0])

mcmc = MCMC(NUTS(model_to_event), num_warmup=10, num_samples=10)
mcmc.run(jax.random.PRNGKey(0), y)
samples = mcmc.get_samples()

# Calculate log likelihood using plate model
ll = log_likelihood(model_plate, samples, y=y)
print("LL shape with plate:", ll["y"].shape)

# Calculate log likelihood using to_event model
ll2 = log_likelihood(model_to_event, samples, y=y)
print("LL shape with to_event:", ll2["y"].shape)
