import numpy as np
from .engine.env import get_jax_primitives

def summary(result, digits=3):
    """
    Print a posterior summary table for a because.fit() result.

    Computes mean, standard deviation, 2.5% and 97.5% quantiles, Rhat
    (Gelman-Rubin convergence diagnostic), and effective sample size (n_eff)
    for every sampled parameter. Mirrors the output of summary(fit) in
    the R because package.

    :param result: Dictionary returned by fit().
    :param digits: Number of decimal places to display (default 3).
    :return: A pandas.DataFrame with one row per parameter.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for because.summary(). "
            "Install it with: pip install pandas"
        )

    prim = get_jax_primitives()
    diag = prim["diag"]

    samples = result.get("samples")
    if samples is None:
        raise ValueError(
            "result['samples'] not found. Make sure fit() was called without "
            "dsep_only=True."
        )

    rows = []
    for param, vals in samples.items():
        vals = np.asarray(vals)
        flat = vals.flatten()

        mean  = float(np.mean(flat))
        sd    = float(np.std(flat, ddof=1))
        q2_5  = float(np.percentile(flat, 2.5))
        q97_5 = float(np.percentile(flat, 97.5))

        if vals.ndim == 2 and vals.shape[0] > 1:
            rhat  = float(diag.gelman_rubin(vals))
            n_eff = float(diag.effective_sample_size(vals))
        else:
            rhat  = float("nan")
            n_eff = float(flat.size)

        rows.append({
            "parameter": param,
            "mean":    round(mean,  digits),
            "sd":      round(sd,    digits),
            "2.5%":    round(q2_5,  digits),
            "97.5%":   round(q97_5, digits),
            "Rhat":    round(rhat,  digits),
            "n_eff":   round(n_eff, 1),
        })

    df = pd.DataFrame(rows).set_index("parameter")
    return df


def dsep_summary(result, digits=3, verbose=False):
    """
    Print a summary table of d-separation test results from a because.fit() call.

    :param result: Dictionary returned by fit() with dsep=True.
    :param digits: Number of decimal places to display (default 3).
    :param verbose: If True, prints a one-line pass/fail verdict above the table.
    :return: A pandas.DataFrame with one row per d-sep claim.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for because.dsep_summary(). "
            "Install it with: pip install pandas"
        )

    tests = result.get("dsep_results")
    if tests is None:
        raise ValueError(
            "result['dsep_results'] not found. "
            "Make sure fit() was called with dsep=True."
        )

    rows = []
    for t in tests:
        row = {
            "claim":       t["claim"],
            "coefficient": t["coefficient"],
            "mean":        round(t["mean"],    digits),
            "2.5%":        round(t["ci_2.5"],  digits),
            "97.5%":       round(t["ci_97.5"], digits),
            "result":      "PASS" if t["is_independent"] else "FAIL",
            "Rhat":        round(t["rhat"],    digits),
            "n_eff":       round(t["n_eff"],   1),
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("claim")

    if verbose:
        n_pass  = sum(1 for t in tests if t["is_independent"])
        n_total = len(tests)
        verdict = "Model is consistent with the data." if n_pass == n_total \
                  else "One or more independence claims FAILED — consider revising the DAG."
        print(f"D-separation tests: {n_pass}/{n_total} passed. {verdict}\n")

    return df
