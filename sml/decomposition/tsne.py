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


import jax.numpy as jnp
from jax import lax

from .pca import PCA


class TSNE:
    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        learning_rate="auto",
        max_iter=300,
        early_exaggeration=12.0,
        early_exaggeration_iter=200,
        momentum=0.8,
        init='pca',
        pca_method='power_iteration',
        pca_max_power_iter=150,
        pca_max_jacobi_iter=5,
        pca_projection_iter=4,
        pca_random_matrix=None,
        pca_scale=None,
        pca_n_oversamples=10,
        max_attempts=50,
        sigma_maxs=1e12,
        sigma_mins=1e-12,
    ):
        """
        t-distributed Stochastic Neighbor Embedding.

        Parameters
        ----------
        n_components : int, default=2
            Dimension of the embedded space.
        perplexity : float, default=30.0
            The perplexity is related to the number of nearest neighbors that
            are considered in the high-dimensional space. Larger datasets
            usually require a larger perplexity. Consider selecting a value
            between 5 and 50. Different values can result in significantly
            different results.
        learning_rate : float or "auto", default="auto"
            The learning rate for t-SNE. If "auto", the learning rate is set to
            `max(n_samples / early_exaggeration / 4, 50)`.
        max_iter : int, default=300
            Maximum number of iterations for the optimization. Should be at
            least 250.
        early_exaggeration : float, default=12.0
            Controls how tight natural clusters in the original space are in
            the embedded space and how much space will be between them. For
            larger values, the space between natural clusters will be larger
            in the embedded space.
        early_exaggeration_iter : int, default=200
            Number of iterations during which the early exaggeration factor
            is applied.
        momentum : float, default=0.8
            The momentum for the update.
        init : {'pca', 'random'}, default='pca'
            Initialization of embedding.
            - 'pca': PCA initialization is used.
            - 'random': Initialize with random numbers. If this option is
              chosen, `Y_init` must be provided in the `fit` method.
        pca_method : str, default='power_iteration'
            Method for PCA if `init='pca'`. See `sml.decomposition.PCA` for details.
        pca_max_power_iter : int, default=150
            Maximum number of power iterations for PCA if `init='pca'` and `pca_method='power_iteration'`.
        pca_max_jacobi_iter : int, default=5
            Maximum number of Jacobi iterations for PCA if `init='pca'` and `pca_method='jacobi_iteration'`.
        pca_projection_iter : int, default=4
            Number of projection iterations for PCA if `init='pca'`.
        pca_random_matrix : jax.numpy.ndarray, optional
            User-provided random matrix for PCA if `init='pca'`.
        pca_scale : float, optional
            Scaling factor for PCA if `init='pca'`.
        pca_n_oversamples : int, default=10
            Number of oversamples for PCA if `init='pca'`.
        max_attempts : int, default=50
            Maximum number of attempts to find the correct sigma for each point
            in the binary search for perplexity.
        sigma_maxs : float, default=1e12
            Upper bound for sigma in the binary search for perplexity.
        sigma_mins : float, default=1e-12
            Lower bound for sigma in the binary search for perplexity.
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.momentum = momentum
        self.init = init
        self.pca_method = pca_method
        self.pca_max_power_iter = pca_max_power_iter
        self.pca_max_jacobi_iter = pca_max_jacobi_iter
        self.pca_projection_iter = pca_projection_iter
        self.pca_random_matrix = pca_random_matrix
        self.pca_scale = pca_scale
        self.pca_n_oversamples = pca_n_oversamples
        self.max_attempts = max_attempts
        self.sigma_maxs = sigma_maxs
        self.sigma_mins = sigma_mins

        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_samples_in_ = None
        self.n_features_in_ = None

    def _squared_dist_mat(self, data):
        """Compute squared Euclidean distance matrix.

        Parameters
        ----------
        data : jax.numpy.ndarray
            Input data, shape (n_samples, n_features).

        Returns
        -------
        dist_mat : jax.numpy.ndarray
            Squared Euclidean distance matrix, shape (n_samples, n_samples).
        """
        return jnp.sum((data[:, None, :] - data[None, :, :]) ** 2, axis=-1)

    def _pairwise_affinities(self, data, sigmas, dist_mat):
        """Compute pairwise affinities based on Gaussian kernel.

        Parameters
        ----------
        data : jax.numpy.ndarray
            Input data, shape (n_samples, n_features).
        sigmas : jax.numpy.ndarray
            Standard deviations of Gaussian kernels for each sample, shape (n_samples, 1).
        dist_mat : jax.numpy.ndarray
            Squared Euclidean distance matrix, shape (n_samples, n_samples).

        Returns
        -------
        P_normalized : jax.numpy.ndarray
            Pairwise affinities, normalized per row, shape (n_samples, n_samples).
        """
        P_unnormalized = jnp.exp(-dist_mat / (2 * (sigmas**2) + 1e-12))
        P_unnormalized = P_unnormalized.at[jnp.diag_indices_from(P_unnormalized)].set(
            0.0
        )
        P_normalized = P_unnormalized / (
            jnp.sum(P_unnormalized, axis=1, keepdims=True) + 1e-12
        )
        return P_normalized

    def _get_perplexities(self, P):
        """Compute perplexity for each row of the affinity matrix.

        Parameters
        ----------
        P : jax.numpy.ndarray
            Pairwise affinity matrix, shape (n_samples, n_samples).

        Returns
        -------
        perplexities : jax.numpy.ndarray
            Perplexity for each row, shape (n_samples,).
        """
        entropy = -jnp.sum(P * jnp.log2(P + 1e-12), axis=1)
        return 2**entropy

    def _all_sym_affinities(self, data, perp):
        """
        Compute symmetric affinity matrix P using a binary search for sigmas
        that yield the target perplexity.

        Parameters
        ----------
        data : jax.numpy.ndarray
            Input data, shape (n_samples, n_features).
        perp : float
            Target perplexity.

        Returns
        -------
        P : jax.numpy.ndarray
            Symmetric affinity matrix, shape (n_samples, n_samples).
        """
        dist_mat = self._squared_dist_mat(data)
        n_samples = data.shape[0]

        sigma_maxs = jnp.full(n_samples, self.sigma_maxs)
        sigma_mins = jnp.full(n_samples, self.sigma_mins)
        sigmas = (sigma_mins + sigma_maxs) / 2

        def body_fun(carry, _):
            current_sigmas, current_sigma_mins, current_sigma_maxs = carry
            P_body = self._pairwise_affinities(data, current_sigmas[:, None], dist_mat)
            current_perps = self._get_perplexities(P_body)

            is_too_high = (current_perps > perp).astype(jnp.float32)
            is_too_low = (current_perps < perp).astype(jnp.float32)

            new_sigma_maxs = (
                is_too_high * current_sigmas + (1.0 - is_too_high) * current_sigma_maxs
            )
            new_sigma_mins = (
                is_too_low * current_sigmas + (1.0 - is_too_low) * current_sigma_mins
            )
            updated_sigmas = (new_sigma_mins + new_sigma_maxs) / 2.0
            return (updated_sigmas, new_sigma_mins, new_sigma_maxs), current_perps

        (final_sigmas, _, _), _ = lax.scan(
            body_fun, (sigmas, sigma_mins, sigma_maxs), None, length=self.max_attempts
        )

        P = self._pairwise_affinities(data, final_sigmas[:, None], dist_mat)
        P = (P + P.T) / (2 * n_samples)
        return P

    def _low_dim_affinities(self, Y_embedding):
        """Compute the low-dimensional affinity matrix Q.

        Uses a Student's t-distribution with one degree of freedom (which
        is the Cauchy distribution).

        Parameters
        ----------
        Y_embedding : jax.numpy.ndarray
            Low-dimensional embedding, shape (n_samples, n_components).

        Returns
        -------
        Q : jax.numpy.ndarray
            Low-dimensional affinity matrix, shape (n_samples, n_samples).
        Y_dist_mat : jax.numpy.ndarray
            Squared Euclidean distance matrix of the low-dimensional embeddings,
            shape (n_samples, n_samples).
        """
        Y_dist_mat = self._squared_dist_mat(Y_embedding)
        numers = (1 + Y_dist_mat) ** (-1)
        denom = jnp.sum(numers) - jnp.sum(jnp.diag(numers)) + 1e-12
        Q = numers / denom
        Q = Q.at[jnp.diag_indices_from(Q)].set(0.0)
        return Q, Y_dist_mat

    def _compute_grad(self, P, Q, Y, Y_dist_mat):
        """Compute the gradient of the KL divergence.

        The KL divergence is between the high-dimensional affinities P and
        the low-dimensional affinities Q.

        Parameters
        ----------
        P : jax.numpy.ndarray
            High-dimensional symmetric affinity matrix, shape (n_samples, n_samples).
        Q : jax.numpy.ndarray
            Low-dimensional affinity matrix, shape (n_samples, n_samples).
        Y : jax.numpy.ndarray
            Current low-dimensional embedding, shape (n_samples, n_components).
        Y_dist_mat : jax.numpy.ndarray
            Squared Euclidean distance matrix of the low-dimensional embeddings,
            shape (n_samples, n_samples).

        Returns
        -------
        grad : jax.numpy.ndarray
            Gradient of the KL divergence with respect to Y,
            shape (n_samples, n_components).
        """
        Ydiff = Y[:, None, :] - Y[None, :, :]
        pq_factor = (P - Q)[:, :, None]
        dist_factor = ((1 + Y_dist_mat) ** (-1))[:, :, None]
        return jnp.sum(4 * pq_factor * Ydiff * dist_factor, axis=1)

    def fit(self, X, Y_init=None):
        """
        Fit X into an embedded space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            High-dimensional data.
        Y_init : array, shape (n_samples, n_components), optional
            Initial low-dimensional embedding. Used if init='random'.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = jnp.asarray(X, dtype=jnp.float32)
        self.n_samples_in_, self.n_features_in_ = X.shape

        if self.perplexity >= self.n_samples_in_:
            raise ValueError("Perplexity must be less than n_samples")

        learning_rate = self.learning_rate
        if learning_rate == "auto":
            learning_rate = max(self.n_samples_in_ / self.early_exaggeration / 4, 50)
        elif not isinstance(learning_rate, (int, float)):
            raise ValueError("learning_rate must be float or 'auto'")

        current_momentum = self.momentum

        if self.init == "pca":
            pca = PCA(
                method=self.pca_method,
                n_components=self.n_components,
                max_power_iter=self.pca_max_power_iter,
                max_jacobi_iter=self.pca_max_jacobi_iter,
                projection_iter=self.pca_projection_iter,
                random_matrix=self.pca_random_matrix,
                scale=self.pca_scale,
                n_oversamples=self.pca_n_oversamples,
            )
            pca.fit(X)
            Y = pca.transform(X)
            Y = Y / jnp.std(Y[:, 0]) * 1e-4
        elif self.init == "random":
            if Y_init is None:
                raise ValueError("For init='random', Y_init must be provided.")
            if Y_init.shape != (self.n_samples_in_, self.n_components):
                raise ValueError(
                    f"Y_init must have shape ({self.n_samples_in_}, {self.n_components})."
                )
            Y = Y_init
        else:
            raise ValueError("init must be 'pca' or 'random'")

        Y_old = Y.copy()
        P = self._all_sym_affinities(X, self.perplexity) * self.early_exaggeration
        P = jnp.clip(P, 1e-12, None)

        for t in range(self.max_iter):
            Q, Y_dist_mat = self._low_dim_affinities(Y)
            Q = jnp.clip(Q, 1e-12, None)
            grad = self._compute_grad(P, Q, Y, Y_dist_mat)

            current_momentum = (
                0.5 if t < self.early_exaggeration_iter else self.momentum
            )

            Y_update = learning_rate * grad
            Y_residuals = current_momentum * (Y - Y_old)

            Y = Y - Y_update + Y_residuals
            Y_old = Y.copy()

            if t == self.early_exaggeration_iter - 1:
                P = P / self.early_exaggeration

            if t == self.max_iter - 1:
                Q_final, _ = self._low_dim_affinities(Y)
                Q_final = jnp.clip(Q_final, 1e-12, None)

                P_for_kl = P.at[jnp.diag_indices_from(P)].set(0.0)
                P_for_kl = jnp.clip(P_for_kl, 0, None)

                log_P_part = P_for_kl * jnp.log(P_for_kl + 1e-12)
                log_Q_part = P_for_kl * jnp.log(Q_final + 1e-12)
                self.kl_divergence_ = jnp.sum(log_P_part - log_Q_part)

        self.embedding_ = Y
        return self

    def transform(self):
        """
        Return the embedding of the data.
        Note: t-SNE does not support transforming new unseen data.
        This method returns the embedding of the data used in .fit().

        Returns
        -------
        embedding_ : array, shape (n_samples, n_components)
            Low-dimensional embedding.
        """
        if self.embedding_ is None:
            raise RuntimeError(
                "This TSNE instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        return self.embedding_

    def fit_transform(self, X, Y_init=None):
        """
        Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            High-dimensional data.
        Y_init : array, shape (n_samples, n_components), optional
            Initial low-dimensional embedding. Used if init='random'.

        Returns
        -------
        embedding_ : array, shape (n_samples, n_components)
            Low-dimensional embedding.
        """
        self.fit(X, Y_init=Y_init)
        return self.embedding_
