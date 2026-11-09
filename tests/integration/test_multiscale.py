import pytest
import numpy as np
from because.api import fit

@pytest.mark.integration
def test_multiscale_dag_dsep():
    rng = np.random.default_rng(42)
    N_year = 5
    N_ind_per_year = 20
    N_total = N_year * N_ind_per_year

    # 1. Coarse variables (Year scale)
    Temp = rng.normal(0, 1, N_year)
    Rain = 0.5 * Temp + rng.normal(0, 0.5, N_year)

    # 2. Link variable
    idx_year = np.repeat(np.arange(N_year), N_ind_per_year)

    # 3. Fine variables (Individual scale)
    Mass = 1.2 * Temp[idx_year] + rng.normal(0, 0.5, N_total)
    Offspring = 2.0 * Mass + rng.normal(0, 1.0, N_total)

    data = {
        "Temp": Temp,
        "Rain": Rain,
        "idx_Temp": idx_year,
        "idx_Rain": idx_year,
        "Mass": Mass,
        "Offspring": Offspring
    }

    res = fit(
        ["Rain ~ Temp", "Mass ~ Temp", "Offspring ~ Mass"],
        data=data,
        dsep=True,
        num_warmup=150,
        num_samples=200,
        quiet=True
    )

    assert "mcmc" in res
    assert "dsep_results" in res
    assert len(res["dsep_results"]) > 0
    for claim in res["dsep_results"]:
        assert "mean" in claim
        assert "ci_2.5" in claim
        assert "ci_97.5" in claim
