import os
import sys

# Ensure JAX thread environment variables are set before any JAX import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest
import numpy as np

@pytest.fixture
def rng():
    return np.random.default_rng(42)
