import pytest
import numpy as np
from scipy.linalg import toeplitz
from because.api import fit

@pytest.mark.integration
def test_correlated_errors():
    N_species = 10
    N_obs = 100
    rng = np.random.default_rng(42)

    # Covariance structure
    row = 0.8 ** np.arange(N_species)
    Sigma = toeplitz(row)

    L = np.linalg.cholesky(Sigma)
    z_raw = rng.normal(0, 1, N_species)
    z_species = np.dot(L, z_raw) * 1.5

    species_idx = rng.integers(0, N_species, size=N_obs)
    x = rng.normal(0, 1, N_obs)
    y = 2.0 * x + z_species[species_idx] + rng.normal(0, 0.5, N_obs)

    data = {
        "x": x,
        "y": y,
        "species": species_idx,
        "N_species": N_species
    }

    cor_matrices = {"species": Sigma}
    res = fit(
        ["y ~ x + (1 | species)"],
        data=data,
        cor_matrices=cor_matrices,
        num_warmup=200,
        num_samples=300,
        quiet=True
    )

    assert "mcmc" in res
    samples = res["mcmc"].get_samples()
    assert "beta_y_x" in samples
    assert "sigma_y_species" in samples
    beta_est = float(np.mean(samples["beta_y_x"]))
    assert 1.0 < beta_est < 3.0
