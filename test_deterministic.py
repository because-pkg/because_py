import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from because.api import fit

def test_deterministic():
    print("=== Simulating Interaction and Polynomial DAG ===")
    print("X -> Y")
    print("Z -> Y")
    print("X*Z -> Y (Interaction)")
    print("I(X^2) -> W (Polynomial)")
    
    N = 1000
    np.random.seed(42)
    
    # Exogenous
    x = np.random.normal(0, 1, N)
    z = np.random.normal(0, 1, N)
    
    # Interaction
    # y = 2x + 1.5z - 3(x*z) + e
    y = 2.0 * x + 1.5 * z - 3.0 * (x * z) + np.random.normal(0, 0.5, N)
    
    # Polynomial
    # w = 0.5 * x^2 + e
    w = 0.5 * (x**2) + np.random.normal(0, 0.5, N)
    
    data = {
        "x": x,
        "z": z,
        "y": y,
        "w": w
    }
    
    equations = [
        "y ~ x + z + x:z",  # or x*z, but we are directly supplying the interaction term for testing
        "w ~ I(x^2)"
    ]
    
    print("\nExpected D-Sep Tests:")
    print("1. Z _|_ X")
    print("2. W _|_ Z | X")
    print("3. W _|_ Y | X, Z (and det_x_x_z)")
    
    results = fit(
        equations=equations,
        data=data,
        dsep=True,
        num_warmup=500,
        num_samples=1000,
        quiet=False
    )
    
    print("\n=== Main MCMC Posterior Summary ===")
    results["mcmc"].print_summary(exclude_deterministic=True)
    
    print("\n=== M-Separation / D-Sep Results ===")
    if results["dsep_results"]:
        for res in results["dsep_results"]:
            print(f"\nClaim: {res['claim']}")
            print(f"Equation: {res['equation']}")
            print(f"Coefficient: {res['coefficient']} = {res['mean']:.3f} CI: [{res['ci_2.5']:.3f}, {res['ci_97.5']:.3f}]")
            print(f"Is Independent? {res['is_independent']}")
    else:
        print("No implied claims found!")

if __name__ == "__main__":
    test_deterministic()
