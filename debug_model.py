import sys
sys.path.insert(0, ".")
from because.builder.parser import FormulaParser
from because.builder.compiler import NumPyroBuilder
from because.builder.graph import CausalGraph
import numpy as np
import jax.numpy as jnp
import jax
import numpyro

N = 20
data = {
    "Lifespan": np.random.randn(N),
    "Brain": np.random.randn(N),
    "N": N,
    "Ntree": 3,
}

cor_matrices = {
    "multiPhylo": {
        "type": "multiPhylo",
        "matrix": {
            "eigvals": np.random.rand(N, 3),
            "eigvecs": np.random.rand(N, N, 3)
        },
        "transform_func": "MOCKED"
    }
}

parser = FormulaParser(["Lifespan ~ Brain + (1|multiPhylo)"])
graph = CausalGraph(parser)

builder = NumPyroBuilder(graph)

def mock_transform(numpyro, jnp, jax, dist, var, group_name, num_groups, matrix_dict, z_raw, sigma, shared_state):
    return z_raw, sigma

cor_matrices["multiPhylo"]["transform_func"] = mock_transform
builder.cor_matrices = cor_matrices

model = builder.generate_model_function()

from numpyro.infer.util import log_density
rng_key = jax.random.PRNGKey(0)

try:
    log_p, tr2 = log_density(model, (), data, {})
    print(f"\nLog density SUCCESS: {log_p}")
except Exception as e:
    import traceback
    print("\nLog density ERROR:")
    traceback.print_exc()

