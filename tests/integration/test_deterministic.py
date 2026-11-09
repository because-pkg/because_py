import pytest
import numpy as np
from because.api import fit

@pytest.mark.integration
def test_deterministic_terms():
    N = 300
    rng = np.random.default_rng(42)

    x = rng.normal(0, 1, N)
    z = rng.normal(0, 1, N)
    y = 2.0 * x + 1.5 * z - 3.0 * (x * z) + rng.normal(0, 0.5, N)
    w = 0.5 * (x**2) + rng.normal(0, 0.5, N)

    data = {"x": x, "z": z, "y": y, "w": w}
    equations = [
        "y ~ x + z + x:z",
        "w ~ I(x^2)"
    ]

    results = fit(
        equations=equations,
        data=data,
        dsep=True,
        num_warmup=200,
        num_samples=300,
        quiet=True
    )

    assert "mcmc" in results
    samples = results["mcmc"].get_samples()
    # Check interaction and polynomial coefficients
    assert "beta_y_x_x_z" in samples or any("x_x_z" in k for k in samples)
    assert "beta_w_x_pow_2" in samples or any("x_pow_2" in k for k in samples)
    assert len(results["dsep_results"]) > 0
