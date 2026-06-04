import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from because.api import fit
import numpy as np

N = 200
np.random.seed(123)
x = np.random.normal(0, 1, N)
y = 1.5 * x + np.random.normal(0, 0.5, N)

data = {"x": x, "y": y}

print("Fitting model...")
res = fit(["y ~ x"], data=data, calculate_waic=True, num_samples=500, quiet=True)

waic_res = res["waic"]
print(f"WAIC: {waic_res['waic']['Estimate']:.2f} (SE: {waic_res['waic']['SE']:.2f})")
print(f"p_waic: {waic_res['p_waic']['Estimate']:.2f} (SE: {waic_res['p_waic']['SE']:.2f})")

