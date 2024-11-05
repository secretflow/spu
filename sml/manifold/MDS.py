import jax
import jax.numpy as jnp

from sml.manifold.jacobi import Jacobi, normalization


def mds(D, num_samples, n_components):
    D_2 = jnp.square(D)
    B = jnp.zeros((num_samples, num_samples))
    B = -0.5 * D_2
    # 按行求和
    dist_2_i = jnp.sum(B, axis=1)
    dist_2_i = dist_2_i / num_samples
    # 按列求和
    dist_2_j = dist_2_i.T
    # 全部求和
    dist_2 = jnp.sum(dist_2_i)
    dist_2 = dist_2 / (num_samples)
    for i in range(num_samples):
        for j in range(num_samples):
            B = B.at[i, j].set(B[i][j] - dist_2_i[i] - dist_2_j[j] + dist_2)

    values, vectors = Jacobi(B, num_samples)

    values = jnp.diag(values)
    values = jnp.array(values)
    perm = jnp.argsort(values)

    vectors = jnp.take(vectors, perm[num_samples - n_components : num_samples], axis=0)
    vectors = vectors.T
    values = jnp.take(values, perm[num_samples - n_components : num_samples], axis=0)
    values = jnp.sqrt(jnp.diag(values))
    ans = jnp.dot(vectors, values)

    return B, ans, values, vectors
