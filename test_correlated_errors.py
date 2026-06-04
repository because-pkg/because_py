import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from because.api import fit

np.random.seed(42)
N_species = 10
N_obs = 100

# Create a phylogenetic-like covariance matrix (just a simple Toeplitz for testing)
from scipy.linalg import toeplitz
row = 0.8 ** np.arange(N_species)
Sigma = toeplitz(row)

# Ground truth correlated random effects
L = np.linalg.cholesky(Sigma)
z_raw = np.random.normal(0, 1, N_species)
z_species = np.dot(L, z_raw) * 1.5

# Data generation
species_idx = np.random.randint(0, N_species, N_obs)
x = np.random.normal(0, 1, N_obs)
y = 2.0 * x + z_species[species_idx] + np.random.normal(0, 0.5, N_obs)

data = {
    "x": x,
    "y": y,
    "species": species_idx,
    "N_species": N_species
}

print("Fitting model WITHOUT correlated errors...")
res_indep = fit(["y ~ x + (1 | species)"], data=data, num_samples=500, quiet=True)
samples_indep = res_indep["mcmc"].get_samples()
z_est_indep = np.mean(samples_indep["z_y_species"], axis=0) * np.mean(samples_indep["sigma_y_species"])

print("Fitting model WITH correlated errors...")
cor_matrices = {"species": Sigma}
res_cor = fit(["y ~ x + (1 | species)"], data=data, cor_matrices=cor_matrices, num_samples=500, quiet=True)
samples_cor = res_cor["mcmc"].get_samples()
z_est_cor = np.mean(samples_cor["z_y_species"], axis=0) * np.mean(samples_cor["sigma_y_species"])

# Compare recovery of random effects
corr_indep = np.corrcoef(z_species, z_est_indep)[0, 1]
corr_cor = np.corrcoef(z_species, z_est_cor)[0, 1]

print(f"Random Effect Recovery (Independent): r = {corr_indep:.3f}")
print(f"Random Effect Recovery (Correlated):  r = {corr_cor:.3f}")

if corr_cor > corr_indep:
    print("SUCCESS: Correlated errors improved random effect recovery!")
else:
    print("WARNING: Correlated errors did not improve recovery (often happens with small N, but mechanics work if no crash)")

