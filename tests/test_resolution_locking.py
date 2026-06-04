import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from because.api import fit

np.random.seed(42)
N_year = 5
N_ind_per_year = 20
N_total = N_year * N_ind_per_year

idx_year = np.repeat(np.arange(N_year), N_ind_per_year)

# 1. Fine variable drives Coarse variable!
# (E.g., Mean body mass of population drives Annual Population Growth)
Mass = np.random.normal(0, 1, N_total)

# Aggregate manually to create the ground truth
import jax.numpy as jnp
import jax
sum_mass = jax.ops.segment_sum(jnp.array(Mass), jnp.array(idx_year), num_segments=N_year)
mean_mass = sum_mass / N_ind_per_year

# Growth is coarse (5 years)
Growth = 2.0 * mean_mass + np.random.normal(0, 0.5, N_year)

data = {
    "Mass": Mass,
    "Growth": np.array(Growth),
    "idx_Mass": idx_year
}

print("Fitting Resolution Locked DAG...")
# DAG: Growth ~ Mass
# Growth is length 5. Mass is length 100.
# The compiler should automatically segment_sum Mass to length 5!

res = fit(["Growth ~ Mass"], data=data, num_samples=500, quiet=True)
samples = res["mcmc"].get_samples()
beta = np.mean(samples["beta_Growth_Mass"])
print(f"Recovered Beta (Expected ~2.0): {beta:.3f}")

