import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from because.api import fit

np.random.seed(42)
N_year = 5
N_ind_per_year = 20
N_total = N_year * N_ind_per_year

# 1. Coarse variables (Year scale)
Temp = np.random.normal(0, 1, N_year)
Rain = 0.5 * Temp + np.random.normal(0, 0.5, N_year)

# 2. Link variable
idx_year = np.repeat(np.arange(N_year), N_ind_per_year)

# 3. Fine variables (Individual scale)
# Mass depends on coarse (Temp) and fine noise
Mass = 1.2 * Temp[idx_year] + np.random.normal(0, 0.5, N_total)

# Offspring depends on Mass
Offspring = 2.0 * Mass + np.random.normal(0, 1.0, N_total)

data = {
    "Temp": Temp,
    "Rain": Rain,
    "idx_Temp": idx_year,
    "idx_Rain": idx_year,
    "Mass": Mass,
    "Offspring": Offspring
}

print("Fitting Multiscale DAG...")
# DAG: Rain ~ Temp
#      Mass ~ Temp
#      Offspring ~ Mass
# Implied d-sep: Rain _|_ Mass | Temp (cross-scale)
#                Rain _|_ Offspring | Temp (cross-scale)
#                Temp _|_ Offspring | Mass (cross-scale)

res = fit(["Rain ~ Temp", "Mass ~ Temp", "Offspring ~ Mass"], data=data, dsep=True, num_samples=500, quiet=True)

for claim in res["dsep_results"]:
    print(f"Test: {claim['claim']}")
    print(f"  Coefficient: {claim['mean']:.3f} [{claim['ci_2.5']:.3f}, {claim['ci_97.5']:.3f}] - {claim['is_independent']}")

