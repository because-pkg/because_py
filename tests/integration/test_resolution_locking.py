import pytest
import numpy as np
from because.api import fit

@pytest.mark.integration
def test_resolution_locking():
    rng = np.random.default_rng(42)
    N_year = 5
    N_ind_per_year = 20
    N_total = N_year * N_ind_per_year

    idx_year = np.repeat(np.arange(N_year), N_ind_per_year)

    # Fine variable drives Coarse variable
    Mass = rng.normal(0, 1, N_total)

    # Aggregate manually to create ground truth
    mean_mass = np.zeros(N_year)
    for y in range(N_year):
        mean_mass[y] = np.mean(Mass[idx_year == y])

    Growth = 2.0 * mean_mass + rng.normal(0, 0.3, N_year)

    data = {
        "Mass": Mass,
        "Growth": Growth,
        "idx_Mass": idx_year
    }

    # Growth is coarse (5 years), Mass is length 100.
    # The compiler automatically aggregates Mass to length 5.
    res = fit(["Growth ~ Mass"], data=data, num_warmup=150, num_samples=200, quiet=True)
    samples = res["mcmc"].get_samples()
    assert "beta_Growth_Mass" in samples
    beta = float(np.mean(samples["beta_Growth_Mass"]))
    assert 1.0 < beta < 3.0
