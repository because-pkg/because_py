import because
import numpy as np
rng = np.random.default_rng(42)
n   = 2000
x   = rng.normal(size=n)
y   = 0.7 * x + rng.normal(size=n)
result = because.fit(
    equations   = ["y ~ x"],
    data        = {"x": x, "y": y},
    num_samples = 2000,
    num_warmup  = 1000,
    num_chains  = 3       
)
import arviz as az
idata = az.from_numpyro(result["mcmc"])
print("Successfully extracted MCMC object for ArviZ!")
