import warnings
import os
import sys

# ---------------------------------------------------------------------------
# Lazy JAX / NumPyro initialisation
# ---------------------------------------------------------------------------
# JAX creates its C++ threadpools (tf_XLAEigen, tf_foreach, llvm-worker) the
# moment "import jax" executes.  The pool sizes are read from
# std::thread::hardware_concurrency(), which returns the *physical* CPU count
# of the host node — NOT the cgroup/container quota.  We therefore must set
# all thread-limiting env vars *before* any jax import statement runs.
#
# The sentinel _jax_initialized ensures we only call _init_jax() once per
# Python process even if fit() is called many times.
# ---------------------------------------------------------------------------
_jax_initialized: bool = False

# Module-level placeholders filled by _init_jax()
jax               = None
jnp               = None
numpyro           = None
dist              = None
MCMC              = None
NUTS              = None
Predictive        = None
DiscreteHMCGibbs  = None
diag              = None


def _detect_available_cores() -> int:
    """Detect allocated CPU cores across SLURM/PBS/SGE/LSF, cgroups, or local OS."""
    for env_var in (
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "PBS_NP",
        "NSLOTS",
        "LSB_DJOB_NUMPROC",
        "MC_CORES",
    ):
        val = os.environ.get(env_var, "").strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
    try:
        if os.path.exists("/sys/fs/cgroup/cpu.max"):
            with open("/sys/fs/cgroup/cpu.max", "r") as f:
                parts = f.read().strip().split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    quota, period = int(parts[0]), int(parts[1])
                    if quota > 0 and period > 0:
                        return max(1, quota // period)
        elif os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") and os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as fq, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as fp:
                quota, period = int(fq.read().strip()), int(fp.read().strip())
                if quota > 0 and period > 0:
                    return max(1, quota // period)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) - 1)


def _init_jax(n_cores: int = 1) -> None:
    """
    Lazily initialise JAX and NumPyro with thread limits already set.

    Must be called at the top of every public function that uses JAX *before*
    any JAX symbol is referenced.  Subsequent calls after the first are
    no-ops (Python's import system caches modules, so the threadpools are
    created only once).

    Parameters
    ----------
    n_cores : int
        Number of parallel chains / devices requested by the caller.
        Used to size the XLA device-count flag and numpyro host devices.
    """
    global _jax_initialized
    global jax, jnp, numpyro, dist, MCMC, NUTS, Predictive, DiscreteHMCGibbs, diag

    if _jax_initialized:
        return

    avail_cores = _detect_available_cores()
    thread_limit = max(1, int(n_cores), avail_cores)
    # Allocate sufficient logical XLA CPU devices so subsequent fit() calls in
    # the same session can vary n_cores freely without restarting Python/R.
    host_devices = max(int(n_cores), avail_cores, 32)

    # ------------------------------------------------------------------ #
    # Detect whether JAX was already imported by external code (e.g. the  #
    # user ran `import jax` before calling because.fit()).  In that case  #
    # the threadpools are already sized from hardware_concurrency() and   #
    # our env-var limits are too late for the C++ layer.  Warn clearly.   #
    # ------------------------------------------------------------------ #
    jax_already_loaded = "jax" in sys.modules

    if jax_already_loaded:
        warnings.warn(
            "JAX was imported before because.fit() was called.  "
            "Thread-pool limits (OMP_NUM_THREADS, XLA thread counts, etc.) "
            "cannot be applied retroactively to the JAX C++ backend.  "
            "To control thread usage on a cluster, set thread-limit env vars "
            "and ensure JAX is first imported inside because.fit() — i.e. do "
            "NOT call `import jax` at the top of your script before using because.",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        # -------------------------------------------------------------- #
        # Set ALL thread-limiting env vars before any jax import.        #
        # os.environ.setdefault() respects values already set by the     #
        # user (e.g. via Sys.setenv() in R or the Docker ENV block).     #
        # -------------------------------------------------------------- #
        _thread_env = {
            "OMP_NUM_THREADS":               "1",
            "OPENBLAS_NUM_THREADS":          "1",
            "GOTO_NUM_THREADS":              "1",
            "MKL_NUM_THREADS":               "1",
            "MKL_DOMAIN_NUM_THREADS":        "1",
            "NUMEXPR_NUM_THREADS":           "1",
            "LLVM_NUM_THREADS":              "1",   # llvm-worker-N threads
            "TF_NUM_INTEROP_THREADS":        str(thread_limit),
            "TF_NUM_INTRAOP_THREADS":        str(thread_limit),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        for k, v in _thread_env.items():
            os.environ.setdefault(k, v)

        # XLA_FLAGS: merge with any flags already present
        existing_xla = os.environ.get("XLA_FLAGS", "")
        new_flags = []
        if "--xla_force_host_platform_device_count" not in existing_xla:
            new_flags.append(f"--xla_force_host_platform_device_count={host_devices}")
        if "--xla_cpu_multi_thread_eigen" not in existing_xla:
            new_flags.append("--xla_cpu_multi_thread_eigen=false")
        if "intra_op_parallelism_threads" not in existing_xla:
            new_flags.append(f"intra_op_parallelism_threads={thread_limit}")
        if "inter_op_parallelism_threads" not in existing_xla:
            new_flags.append(f"inter_op_parallelism_threads={thread_limit}")
        if new_flags:
            os.environ["XLA_FLAGS"] = (existing_xla + " " + " ".join(new_flags)).strip()

    # ------------------------------------------------------------------ #
    # Now it is safe to import JAX and NumPyro.                           #
    # If jax_already_loaded is True the imports below are instant cache   #
    # hits — no new threadpools are created.                              #
    # ------------------------------------------------------------------ #
    import jax            as _jax
    import jax.numpy      as _jnp
    import numpyro        as _numpyro
    import numpyro.distributions as _dist
    import numpyro.diagnostics   as _diag
    from numpyro.infer import (
        MCMC              as _MCMC,
        NUTS              as _NUTS,
        Predictive        as _Predictive,
        DiscreteHMCGibbs  as _DiscreteHMCGibbs,
    )

    # Expose as module-level globals so the rest of the package works
    jax              = _jax
    jnp              = _jnp
    numpyro          = _numpyro
    dist             = _dist
    diag             = _diag
    MCMC             = _MCMC
    NUTS             = _NUTS
    Predictive       = _Predictive
    DiscreteHMCGibbs = _DiscreteHMCGibbs

    # numpyro.set_host_device_count MUST be called before any JAX computation
    if host_devices > 1 and not jax_already_loaded:
        try:
            _numpyro.set_host_device_count(host_devices)
        except Exception as exc:
            warnings.warn(
                f"because: numpyro.set_host_device_count({host_devices}) failed: {exc}",
                RuntimeWarning,
            )

    _jax_initialized = True


def get_jax_primitives():
    """Returns a dict of initialised JAX and NumPyro modules/classes."""
    if not _jax_initialized:
        _init_jax()
    return {
        "jax": jax,
        "jnp": jnp,
        "numpyro": numpyro,
        "dist": dist,
        "diag": diag,
        "MCMC": MCMC,
        "NUTS": NUTS,
        "Predictive": Predictive,
        "DiscreteHMCGibbs": DiscreteHMCGibbs,
    }
