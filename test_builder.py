import sys
from pprint import pprint
import numpy as np

# Ensure the local because package is discoverable
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from because.builder import FormulaParser, CausalGraph, NumPyroBuilder

def test_builder():
    # We will test the MISSING DATA IMPUTATION logic!
    # y ~ x (both will have 20% missing data)
    equations = [
        "y ~ x"
    ]
    
    print("=== 1. Testing FormulaParser ===")
    parser = FormulaParser(equations)
    parsed = parser.parse()
    pprint(parsed)
    
    print("\n=== 2. Testing CausalGraph ===")
    graph = CausalGraph(parsed)
    dag = graph.build()
    
    print("Nodes:", list(dag.nodes(data=True)))
    print("Topological order:", graph.get_topological_order())
    
    print("\n=== 3. Simulating Dataset with Missing Values ===")
    np.random.seed(42)
    
    N = 1000
    
    true_alpha_x = 0.0
    true_sigma_x = 1.0
    x = np.random.normal(true_alpha_x, true_sigma_x, N)
    
    true_alpha_y = 2.0
    true_beta_y_x = 1.5
    true_sigma_y = 1.0
    
    mu_y = true_alpha_y + true_beta_y_x * x
    y = mu_y + np.random.normal(0, true_sigma_y, N)
    
    # Inject 20% Missing Data randomly into both X and Y
    missing_x_idx = np.random.choice(N, size=int(0.2 * N), replace=False)
    missing_y_idx = np.random.choice(N, size=int(0.2 * N), replace=False)
    
    x[missing_x_idx] = np.nan
    y[missing_y_idx] = np.nan
    
    data = {
        "x": x,
        "y": y
    }
    
    print(f"Simulated {N} rows.")
    print(f"Missing in x: {np.isnan(x).sum()}")
    print(f"Missing in y: {np.isnan(y).sum()}")
    
    print("\n=== 4. Testing NumPyroBuilder Execution (Imputation) ===")
    compiler = NumPyroBuilder(graph)
    model_func = compiler.generate_model_function(data_for_compilation=data)
    
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    
    # Convert data to jax arrays
    jax_data = {k: jnp.array(v) for k, v in data.items()}
    
    print("Starting MCMC Sampling on model with Missing Data...")
    rng_key = jax.random.PRNGKey(0)
    kernel = NUTS(model_func)
    # We use fewer samples here just to quickly verify it doesn't crash
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
    
    mcmc.run(rng_key, **jax_data)
    
    print("\n=== MCMC Posterior Summary (Core Parameters Only) ===")
    mcmc.print_summary(exclude_deterministic=False)
    
    print("\nExpected parameter recovery:")
    print("alpha_y ~ 2.0")
    print("beta_y_x ~ 1.5")
    
    # Let's peek at one of the imputed values if possible (NumPyro suppresses large arrays in print_summary by default)
    # but we will see x_imputed and y_imputed in the keys.
    samples = mcmc.get_samples()
    print("\nKeys in posterior samples:")
    print(list(samples.keys()))

if __name__ == "__main__":
    test_builder()
