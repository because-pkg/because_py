import pytest
import numpy as np
from because.api import fit

@pytest.mark.integration
def test_cascaded_dag_dsep():
    # X -> Y -> Z => Z _|_ X | Y
    N = 300
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, N)
    y = 2.0 + 1.5 * x + rng.normal(0, 0.5, N)
    z = -1.0 + 0.8 * y + rng.normal(0, 0.5, N)

    data = {"x": x, "y": y, "z": z}
    equations = ["y ~ x", "z ~ y"]

    results = fit(
        equations=equations,
        data=data,
        dsep=True,
        num_warmup=200,
        num_samples=300,
        quiet=True
    )

    assert "mcmc" in results
    assert "dsep_results" in results
    assert len(results["dsep_results"]) == 1

    claim = results["dsep_results"][0]
    # In implied independence Z _|_ X | Y, beta_z_x should be close to 0
    # and 95% CI should span 0
    assert claim["ci_2.5"] < 0.2
    assert claim["ci_97.5"] > -0.2
