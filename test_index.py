import jax.numpy as jnp

n_trees = 2
N = 3
eigvals_all = jnp.array([[1., 2., 3.], [4., 5., 6.]]).T  # (3, 2)
K = jnp.arange(n_trees).reshape(-1, 1)  # (2, 1)

try:
    res = eigvals_all[..., K]
    print("Normal indexing worked, shape:", res.shape)
except Exception as e:
    print("Error:", e)

# What if we transpose?
eigvals_t = eigvals_all.T # (2, 3)
res2 = eigvals_t[K] # indexing the first dim
print("Transpose indexing shape:", res2.shape)

