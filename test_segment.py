import jax.numpy as jnp
import jax

Mass = jnp.array([10.0, 20.0, 30.0, 40.0])
idx_year = jnp.array([0, 0, 1, 1])

# Aggregate Mass to Year (length 2)
# using jax.ops.segment_sum
sum_mass = jax.ops.segment_sum(Mass, idx_year, num_segments=2)
count = jax.ops.segment_sum(jnp.ones_like(Mass), idx_year, num_segments=2)
mean_mass = sum_mass / count

print("Mean Mass per Year:", mean_mass)
