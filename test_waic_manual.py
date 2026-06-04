import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood
import numpy as np

def model(x, y):
    alpha = numpyro.sample("alpha", dist.Normal(0, 1))
    beta = numpyro.sample("beta", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    mu = alpha + beta * x
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = 2.0 + 3.0 * x + np.random.normal(0, 0.5, 100)

mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=200)
mcmc.run(jax.random.PRNGKey(0), x=x, y=y)

log_lik = log_likelihood(model, mcmc.get_samples(), x=x, y=y)
ll_y = log_lik["y"] # shape: (num_samples, num_obs)

# WAIC computation
lpd = np.sum(np.log(np.mean(np.exp(ll_y), axis=0)))
p_waic = np.sum(np.var(ll_y, axis=0))
waic = -2 * (lpd - p_waic)

print("Manual WAIC:", waic)
print("p_waic:", p_waic)

# LOO computation using arviz_stats
import arviz as az
import arviz_stats as azs
idata = az.from_numpyro(mcmc, log_likelihood=log_lik)
loo_res = azs.loo(idata, var_name="y")
print(loo_res)

