import sys
import os
import numpy as np
import pprint

# Ensure the local because package is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from because.api import fit

def test_mag():
    print("=== Simulating DAG with Latent Variable ===")
    print("L -> X")
    print("L -> Y")
    print("Y -> Z")
    print("\nL is UNMEASURED (Latent).")
    
    N = 1000
    np.random.seed(42)
    
    # Latent common cause
    L = np.random.normal(0, 1, N)
    
    # X and Y are confounded by L
    x = 1.0 * L + np.random.normal(0, 0.5, N)
    y = -0.8 * L + np.random.normal(0, 0.5, N)
    
    # Z only depends on Y
    z = 1.5 * y + np.random.normal(0, 0.5, N)
    
    # Data only contains X, Y, Z
    data = {
        "x": x,
        "y": y,
        "z": z
    }
    
    equations = [
        "x ~ L",
        "y ~ L",
        "z ~ y"
    ]
    
    print("\nExpected MAG Tests:")
    print("1. Z _|_ X | Y (Conditional Independence, beta_z_x should be 0)")
    print("2. X <-> Y (Induced Correlation due to L, beta_x_y should NOT be 0)")
    
    print("\n=== Running because_py.fit() with Latent MAG Support ===")
    
    results = fit(
        equations=equations,
        data=data,
        dsep=True,
        num_warmup=500,
        num_samples=1000
    )
    
    print("\n=== Main MCMC Posterior Summary ===")
    results["mcmc"].print_summary(exclude_deterministic=True)
    
    print("\n=== M-Separation / MAG Results ===")
    if results["dsep_results"]:
        for res in results["dsep_results"]:
            print(f"\nClaim: {res['claim']}")
            print(f"Equation: {res['equation']}")
            print(f"Coefficient: {res['coefficient']} = {res['mean']:.3f} CI: [{res['ci_2.5']:.3f}, {res['ci_97.5']:.3f}]")
            print(f"Is Independent? {res['is_independent']}")
    else:
        print("No implied claims found!")

if __name__ == "__main__":
    test_mag()
