import pytest
import numpy as np
from because.api import fit

@pytest.mark.integration
def test_latent_mag():
    N = 300
    rng = np.random.default_rng(42)

    # Unmeasured common cause L
    L = rng.normal(0, 1, N)
    x = 1.0 * L + rng.normal(0, 0.5, N)
    y = -0.8 * L + rng.normal(0, 0.5, N)
    z = 1.5 * y + rng.normal(0, 0.5, N)

    data = {"x": x, "y": y, "z": z}
    equations = ["x ~ L", "y ~ L", "z ~ y"]

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
    assert len(results["dsep_results"]) > 0

    # Ensure induced correlation or d-sep test was evaluated
    claims = [res["claim"] for res in results["dsep_results"]]
    assert any("z" in c and "x" in c for c in claims)
