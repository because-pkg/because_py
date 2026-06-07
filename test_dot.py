import jax.numpy as jnp

N = 3
n_trees = 2

# Suppose K is enumerated (2, 1)
eigvecs = jnp.ones((2, 1, N, N))
z_scaled = jnp.ones((2, 1, N))

try:
    res = jnp.dot(eigvecs, z_scaled)
    print("dot shape:", res.shape)
except Exception as e:
    print("dot error:", e)

# What about jnp.einsum?
try:
    res2 = jnp.einsum('...ij,...j->...i', eigvecs, z_scaled)
    print("einsum shape:", res2.shape)
except Exception as e:
    print("einsum error:", e)

