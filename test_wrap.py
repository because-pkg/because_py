import sys
sys.path.insert(0, ".")
import numpyro
from numpyro.handlers import condition, substitute, trace, seed
import jax.numpy as jnp
import jax

def model(data=None):
    k = numpyro.sample("K_tree_multiPhylo", numpyro.distributions.Categorical(probs=jnp.array([0.5, 0.5])))
    return k

def _wrap_model(model, *args, **kwargs):
    gibbs_values = kwargs.pop("_gibbs_sites", {})
    print("GIBBS VALUES:", gibbs_values)
    with condition(data=gibbs_values), substitute(data=gibbs_values):
        return model(*args, **kwargs)

tr = trace(seed(lambda **kwargs: _wrap_model(model, **kwargs), jax.random.PRNGKey(0))).get_trace(_gibbs_sites={"K_tree_multiPhylo": jnp.array(0)})
print(tr["K_tree_multiPhylo"]["is_observed"])
