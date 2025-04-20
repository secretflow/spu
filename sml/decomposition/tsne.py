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

MACHINE_EPSILON = jnp.finfo(jnp.float32).eps


def pairwise_squared_distances(X):
    """Computes pairwise squared Euclidean distances using JAX."""
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
def binary_search_perplexity(
    distances_row, i, perplexity, tol=1e-5, max_tries=50, verbose=10
):
    """Compute P-row for a point using a fixed number of iterations for binary search."""
    Di = distances_row
    log_perplexity = jnp.log(perplexity)

    def compute_H_P(beta):
        P = jnp.exp(-Di * beta)
        P = P.at[i].set(0.0)
        sumP = jnp.maximum(jnp.sum(P), MACHINE_EPSILON)
        H = jnp.log(sumP) + beta * jnp.sum(Di * P) / sumP
        P = P / sumP
        return H, P

    def body_fun(tries, state):
        beta, betamin, betamax, prev_H, prev_P = state
        H, P = compute_H_P(beta)
        Hdiff = H - log_perplexity

        beta_new = (betamin + betamax) / 2.0
        beta_new = jnp.where(jnp.isinf(betamax), beta * 2.0, beta_new)
        beta_new = jnp.where(jnp.isinf(betamin), beta / 2.0, beta_new)

        betamin_new = jnp.where(Hdiff > 0, beta, betamin)
        betamax_new = jnp.where(Hdiff < 0, beta, betamax)

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
    P = jnp.zeros((n_samples, n_samples))

    if verbose > 0:
        print("Computing probabilities for all points...")

    vectorized_binary_search = vmap(
        lambda dist_row, idx: binary_search_perplexity(
            dist_row, idx, perplexity, tol=1e-5, max_tries=50, verbose=verbose
        ),
        in_axes=(0, 0),
    )

    indices = jnp.arange(n_samples)
    P = vectorized_binary_search(distances, indices)

    if verbose > 0:
        print("Symmetrizing probabilities...")
    P = (P + P.T) / (2 * n_samples)
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
    kl_div = jnp.sum(P * (jnp.log(P) - jnp.log(Q)))
    return kl_div


def basic_tsne(
    X,
    n_components=2,
    perplexity=30.0,
    learning_rate=200.0,
    max_iter=1000,
    early_exaggeration=12.0,
    early_exaggeration_iter=250,
    momentum=0.8,
    min_grad_norm=1e-7,
    random_state=None,
    verbose=10,
):
    """
    JAX implementation of t-SNE with gradient descent optimization.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        Target perplexity for joint probabilities.
    learning_rate : float, default=200.0
        Learning rate for gradient descent.
    max_iter : int, default=1000
        Maximum number of iterations.
    early_exaggeration : float, default=12.0
        Early exaggeration factor for P.
    early_exaggeration_iter : int, default=250
        Number of iterations with early exaggeration.
    momentum : float, default=0.8
        Momentum factor after early exaggeration.
    min_grad_norm : float, default=1e-7
        Convergence threshold for gradient norm (not used in this version).
    random_state : int or None, default=None
        Seed for random initialization.
    verbose : int, default=10
        Verbosity level (print every verbose iterations).

    Returns
    -------
    Y : ndarray of shape (n_samples, n_components)
        Low-dimensional embedding.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    n_samples, n_features = X.shape

    if perplexity >= n_samples:
        raise ValueError("Perplexity must be less than n_samples")

    # Set random seed
    key = jax.random.PRNGKey(random_state if random_state is not None else 0)

    # Step 1: Compute pairwise distances
    if verbose > 0:
        print("Computing pairwise distances...")
    t0 = time.time()
    distances_sq = pairwise_squared_distances(X)
    t1 = time.time()
    if verbose > 0:
        print(f"Done in {t1 - t0:.2f}s")

    # Step 2: Compute joint probabilities P
    t0 = time.time()
    P = joint_probabilities_jax(distances_sq, perplexity, verbose=verbose)
    t1 = time.time()
    if verbose > 0:
        print(f"Computed P matrix in {t1 - t0:.2f}s")

    # Step 3: Initialize embedding Y
    Y = 1e-4 * jax.random.normal(key, (n_samples, n_components))
    Y_flat = Y.ravel()

    # Optimization parameters
    degrees_of_freedom = max(n_components - 1, 1)
    update = jnp.zeros_like(Y_flat)
    initial_momentum = 0.5 if early_exaggeration_iter > 0 else momentum

    # JIT-compiled KL divergence and gradient
    kl_divergence_with_grad = jit(
        jax.value_and_grad(kl_divergence_jax), static_argnums=(2, 3, 4)
    )

    if verbose > 0:
        print("Starting optimization...")
    t_start_opt = time.time()

    # Early exaggeration
    P_exaggerated = P * early_exaggeration

    for it in range(max_iter):
        current_momentum = (
            momentum if it >= early_exaggeration_iter else initial_momentum
        )
        current_P = P if it >= early_exaggeration_iter else P_exaggerated

        # Compute KL divergence and gradient
        kl_div, grad = kl_divergence_with_grad(
            Y_flat, current_P, degrees_of_freedom, n_samples, n_components
        )

        # Update with momentum
        update = current_momentum * update - learning_rate * grad
        Y_flat = Y_flat + update

        # Center the embedding
        Y = Y_flat.reshape(n_samples, n_components)
        Y = Y - jnp.mean(Y, axis=0)
        Y_flat = Y.ravel()

        # Logging (no early stopping)
        if (it + 1) % verbose == 0:
            t_now = time.time()
            grad_norm = jnp.linalg.norm(grad)
            # print(f"Iter {it + 1}/{max_iter}: KL divergence={kl_div:.4f}, Grad norm={grad_norm:.4e}, Time={(t_now - t_start_opt):.2f}s")
            if it == early_exaggeration_iter - 1 and early_exaggeration != 1:
                print(f"Finished early exaggeration phase at iter {it + 1}")

    t_end_opt = time.time()
    if verbose > 0:
        print(f"Optimization finished in {t_end_opt - t_start_opt:.2f}s")

    final_embedding = Y_flat.reshape(n_samples, n_components)
    return final_embedding
