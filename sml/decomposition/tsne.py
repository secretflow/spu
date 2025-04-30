# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import jax
import jax.numpy as jnp
from jax import grad, jit, lax, vmap
from jax.numpy.linalg import svd

from .pca import PCA

MACHINE_EPSILON = jnp.finfo(jnp.float32).eps


def pairwise_squared_distances(X):
    """Computes pairwise squared Euclidean distances."""
    sum_X = jnp.sum(X**2, axis=1)
    D2 = -2 * jnp.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]

    D2 = D2.at[jnp.diag_indices_from(D2)].set(0.0)
    return jnp.maximum(D2, 0.0)


def Hbeta(D, beta=1.0):
    """Compute the perplexity and P-row for a specific precision value."""
    P = jnp.exp(-D * beta)
    sumP = jnp.maximum(jnp.sum(P), MACHINE_EPSILON)
    H = jnp.log(sumP) + beta * jnp.sum(D * P) / sumP
    P = P / sumP
    return H, P


@jit
def binary_search_perplexity(distances_row, i, perplexity, tol=1e-5, max_tries=50):
    """Compute P-row for a point using a fixed number of iterations for binary search."""
    Di = distances_row
    log_perplexity = jnp.log(perplexity)

    def compute_H_P(beta):

        P_row = jnp.exp(-Di * beta)

        P_row = P_row.at[i].set(0.0)
        sumP = jnp.maximum(jnp.sum(P_row), MACHINE_EPSILON)
        H = jnp.log(sumP) + beta * jnp.sum(Di * P_row) / sumP
        P_row = P_row / sumP
        return H, P_row

    def body_fun(tries, state):
        beta, betamin, betamax, prev_H, prev_P = state
        H, P = compute_H_P(beta)
        Hdiff = H - log_perplexity

        betamin_new = jnp.where(Hdiff > 0, beta, betamin)
        betamax_new = jnp.where(Hdiff < 0, beta, betamax)

        beta_new = jnp.where(
            jnp.isinf(betamax_new), beta * 2.0, (betamin_new + betamax_new) / 2.0
        )
        beta_new = jnp.where(jnp.isinf(betamin_new), beta / 2.0, beta_new)

        return beta_new, betamin_new, betamax_new, H, P

    beta = 1.0
    betamin = -jnp.inf
    betamax = jnp.inf

    prev_H, prev_P = compute_H_P(beta)

    beta, betamin, betamax, H, P = lax.fori_loop(
        0, max_tries, body_fun, (beta, betamin, betamax, prev_H, prev_P)
    )

    P = P.at[i].set(0.0)
    return P


def joint_probabilities_jax(distances, perplexity, verbose=0):
    """Compute symmetric joint probabilities P_ij from distances using vmap."""
    n_samples = distances.shape[0]

    if verbose > 0:
        print("Computing probabilities for all points...")

    vectorized_binary_search = vmap(
        lambda dist_row, idx: binary_search_perplexity(
            dist_row, idx, perplexity, tol=1e-5, max_tries=50
        ),
        in_axes=(0, 0),
        in_axes=(0, 0),
    )

    indices = jnp.arange(n_samples)
    P = vectorized_binary_search(distances, indices)

    if verbose > 0:
        print("Symmetrizing probabilities...")
    P = (P + P.T) / (2 * jnp.sum(P))
    P = jnp.maximum(P, MACHINE_EPSILON)
    return P


def kl_divergence_jax(Y_flat, P, degrees_of_freedom, n_samples, n_components):
    """Compute KL divergence between P and Q (computed from Y)."""
    Y = Y_flat.reshape(n_samples, n_components)
    dist_sq_low = pairwise_squared_distances(Y)
    inv_dist = 1.0 / (1.0 + dist_sq_low / degrees_of_freedom)
    inv_dist = inv_dist.at[jnp.diag_indices_from(inv_dist)].set(0.0)
    sum_inv_dist = jnp.maximum(jnp.sum(inv_dist), MACHINE_EPSILON)
    Q = inv_dist / sum_inv_dist
    Q = jnp.maximum(Q, MACHINE_EPSILON)

    P_safe = jnp.maximum(P, MACHINE_EPSILON)
    kl_div = jnp.sum(P * (jnp.log(P_safe) - jnp.log(Q)))
    return kl_div


def Tsne(
    X,
    Y_init=None,
    n_components=2,
    perplexity=30.0,
    learning_rate="auto",
    max_iter=1000,
    early_exaggeration=12.0,
    early_exaggeration_iter=250,
    momentum=0.8,
    verbose=10,
    init='pca',
):
    """
    Main function of t-SNE with gradient descent optimization.

    Returns
    -------
    Y : ndarray of shape (n_samples, n_components)
        Low-dimensional embedding.
    final_kl_divergence : float
        The KL divergence of the final embedding.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    n_samples, n_features = X.shape

    if perplexity >= n_samples:
        raise ValueError("Perplexity must be less than n_samples")

    if learning_rate == "auto":
        learning_rate = max(n_samples / early_exaggeration / 4, 50)
        if verbose > 0:
            print(f"Setting auto learning rate: {learning_rate}")
    elif isinstance(learning_rate, (int, float)):
        learning_rate = learning_rate
        if verbose > 0:
            print(f"Setting learning rate: {learning_rate}")
    else:
        raise ValueError("learning_rate must be float or 'auto'")

    if init == "pca":
        pca = PCA(
            method='power_iteration', n_components=n_components, max_power_iter=200
        )
        pca.fit(X)
        Y = pca.transform(X)
        Y = Y / jnp.std(Y[:, 0]) * 1e-4
    elif init == "random":
        if Y_init is None:
            raise ValueError("For init='random', Y_init must be provided.")
        if Y_init.shape != (n_samples, n_components):
            raise ValueError(f"Y_init must have shape ({n_samples}, {n_components}).")
        Y = Y_init
    else:
        raise ValueError("init must be 'pca' or 'random'")

    Y_flat = Y.ravel()

    distances_sq = pairwise_squared_distances(X)
    P = joint_probabilities_jax(distances_sq, perplexity, verbose=verbose)

    degrees_of_freedom = max(n_components - 1.0, 1.0)
    update = jnp.zeros_like(Y_flat)
    initial_momentum = 0.5

    kl_divergence_with_grad = jit(
        jax.value_and_grad(kl_divergence_jax), static_argnums=(2, 3, 4)
    )

    if verbose > 0:
        print("Starting optimization...")

    for it in range(max_iter):
        is_early_exaggeration = it < early_exaggeration_iter
        current_momentum = momentum if not is_early_exaggeration else initial_momentum
        current_P = P * early_exaggeration if is_early_exaggeration else P

        kl_div_iter, grad = kl_divergence_with_grad(
            Y_flat, current_P, degrees_of_freedom, n_samples, n_components
        )

        update = current_momentum * update - learning_rate * grad
        Y_flat = Y_flat + update

        Y = Y_flat.reshape(n_samples, n_components)
        Y = Y - jnp.mean(Y, axis=0)
        Y_flat = Y.ravel()

        if is_early_exaggeration and it == early_exaggeration_iter - 1:
            print(f"Finished early exaggeration phase at iter {it + 1}")

    final_embedding = Y
    final_kl_divergence = kl_divergence_jax(
        final_embedding.ravel(), P, degrees_of_freedom, n_samples, n_components
    )

    return final_embedding, final_kl_divergence
