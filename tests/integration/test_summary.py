import pytest
import numpy as np
from because.api import fit
from because.summary import summary, dsep_summary

@pytest.mark.integration
def test_summary_requires_pandas():
    N = 50
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, N)
    y = 2.0 * x + rng.normal(0, 0.5, N)

    data = {"x": x, "y": y}
    res = fit(["y ~ x"], data=data, num_warmup=50, num_samples=50, quiet=True)

    try:
        import pandas as pd
        df_sum = summary(res)
        assert "mean" in df_sum.columns
    except ImportError:
        with pytest.raises(ImportError, match="pandas is required"):
            summary(res)
        with pytest.raises(ImportError, match="pandas is required"):
            dsep_summary(res)
