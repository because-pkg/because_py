import pytest
import numpy as np
from because.api import fit

@pytest.mark.integration
def test_calculate_waic():
    N = 150
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, N)
    y = 1.5 * x + rng.normal(0, 0.5, N)

    data = {"x": x, "y": y}
    res = fit(["y ~ x"], data=data, calculate_waic=True, num_warmup=150, num_samples=200, quiet=True)

    assert "waic" in res
    waic_res = res["waic"]
    assert "waic" in waic_res
    assert "p_waic" in waic_res
    assert isinstance(waic_res["waic"]["Estimate"], (float, int, np.floating))
    assert isinstance(waic_res["p_waic"]["Estimate"], (float, int, np.floating))
    assert waic_res["waic"]["Estimate"] > 0
    assert waic_res["p_waic"]["Estimate"] > 0
