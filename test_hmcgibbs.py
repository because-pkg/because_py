import sys
sys.path.insert(0, ".")
import jax.numpy as jnp
import jax
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from numpyro.distributions import Categorical, Normal

def model(data=None):
    k = numpyro.sample("K_tree_multiPhylo", Categorical(probs=jnp.array([0.5, 0.5])))
    numpyro.sample("obs", Normal(k, 1), obs=jnp.array([1., 2.]))

nuts = NUTS(model)
kernel = DiscreteHMCGibbs(nuts)
mcmc = MCMC(kernel, num_warmup=1, num_samples=1)
mcmc.run(jax.random.PRNGKey(0), _gibbs_sites={"K_tree_multiPhylo": jnp.array(0)})
print("SUCCESS!")
