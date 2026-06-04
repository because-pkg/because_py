import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood
import arviz as az
import numpy as np

def model(x, y, w):
    alpha = numpyro.sample("alpha", dist.Normal(0, 1))
    beta = numpyro.sample("beta", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    
    mu_y = alpha + beta * x
    numpyro.sample("y", dist.Normal(mu_y, sigma), obs=y)
    
    mu_w = beta + alpha * x
    numpyro.sample("w", dist.Normal(mu_w, sigma), obs=w)

np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = 2.0 + 3.0 * x + np.random.normal(0, 0.5, 100)
w = 3.0 + 2.0 * x + np.random.normal(0, 0.5, 100)

mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=200)
mcmc.run(jax.random.PRNGKey(0), x=x, y=y, w=w)

log_lik = log_likelihood(model, mcmc.get_samples(), x=x, y=y, w=w)
print("Log_lik dict keys:", log_lik.keys())
idata = az.from_numpyro(mcmc, log_likelihood=log_lik)

try:
    waic_y = az.waic(idata, var_name="y")
    print("WAIC (y):", waic_y.waic)
except Exception as e:
    print(e)
    
try:
    # See if we can sum the log-likelihoods for a joint WAIC
    joint_ll = log_lik["y"] + log_lik["w"]
    idata_joint = az.from_numpyro(mcmc, log_likelihood={"joint": joint_ll})
    waic_joint = az.waic(idata_joint, var_name="joint")
    print("WAIC (joint):", waic_joint.waic)
except Exception as e:
    print(e)
