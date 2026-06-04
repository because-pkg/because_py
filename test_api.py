import sys
import os
import numpy as np
import pprint

# Ensure the local because package is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from because.api import fit

def test_dsep():
    print("=== Simulating Cascaded DAG: X -> Y -> Z ===")
    
    # We want Z completely independent of X given Y
    N = 1000
    np.random.seed(123)
    
    x = np.random.normal(0, 1, N)
    
    # Y depends on X
    y = 2.0 + 1.5 * x + np.random.normal(0, 1, N)
    
    # Z depends ONLY on Y
    z = -1.0 + 0.8 * y + np.random.normal(0, 1, N)
    
    data = {
        "x": x,
        "y": y,
        "z": z
    }
    
    equations = [
        "y ~ x",
        "z ~ y"
    ]
    
    print("True equations:")
    print("y ~ x (beta_y_x = 1.5)")
    print("z ~ y (beta_z_y = 0.8)")
    print("\nImplied Independence: Z _|_ X | Y")
    print("D-Separation should test: z ~ x + y")
    print("And beta_z_x should be close to 0 (Confidence interval includes 0).")
    
    print("\n=== Running because_py.fit(dsep=True) ===")
    
    results = fit(
        equations=equations,
        data=data,
        dsep=True,
        num_warmup=500,
        num_samples=1000,
        dsep_max_obs=1000  # don't subsample for this fast test
    )
    
    print("\n=== Main MCMC Posterior Summary ===")
    results["mcmc"].print_summary(exclude_deterministic=True)
    
    print("\n=== D-Separation Results ===")
    if results["dsep_results"]:
        for res in results["dsep_results"]:
            pprint.pprint(res)
    else:
        print("No implied conditional independencies found!")

if __name__ == "__main__":
    test_dsep()
