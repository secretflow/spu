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

import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap


class QuantileTransformer:
    def __init__(
        self,
        n_quantiles=1000,
        output_distribution='uniform',
        subsample=100000,
        random_state=None,
    ):
        """Initialize the transformer for uniform output distribution. NaN values are not handled.

        Args:
            n_quantiles (int): The number of quantiles to compute. Defaults to 1000.
            output_distribution (str): Type of output distribution, currently only 'uniform' is supported.
            subsample (int): Maximum number of samples used for computation.
                             (Note: currently not used in the implementation). Defaults to 100,000.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.

        Raises:
            ValueError: If parameters are invalid.

        Note:
            This transformer assumes the input data contains no NaN values.
            The user **must** handle any NaNs (e.g., through imputation or removal)
            *before* passing the data to this transformer's methods (`fit`, `transform`).
            If the input data contains NaNs, the transformer may produce incorrect
            results or errors without warning.
            The output distribution is always 'uniform'.
        """
        if not isinstance(n_quantiles, int) or n_quantiles <= 0:
            raise ValueError("n_quantiles must be a positive integer.")
        if not isinstance(subsample, int) or subsample <= 0:
            raise ValueError("subsample must be a positive integer.")

        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample
        self.random_state = random_state
        self.quantiles_ = None
        self.references_ = None
        self.n_quantiles_ = None
        self._n_features = None
        self._input_shape = None
        self._key = random.PRNGKey(random_state if random_state is not None else 0)

    def fit(self, X):
        """Fit the transformer, assuming the input data X has no NaNs.

        Args:
            X (array-like): Input data, shape (n_samples, n_features).
                           Must not contain NaN values.

        Returns:
            self: The fitted transformer.

        Raises:
            ValueError: If X is not a 2D array or contains NaN values.
        """
        # Convert to a jax array
        X = jnp.asarray(X)

        if X.ndim != 2:
            raise ValueError("Input array X must be 2D.")

        n_samples, n_features = X.shape
        self._n_features = n_features
        self._input_shape = X.shape

        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        quantiles_ = jnp.zeros((self.n_quantiles_, n_features), dtype=jnp.float32)
        references_ = jnp.zeros((self.n_quantiles_, n_features), dtype=jnp.float32)
        target_quantiles = jnp.linspace(0, 1, self.n_quantiles_, dtype=jnp.float32)

        def compute_quantile(p, sorted_col):
            """Computes quantile using linear interpolation."""
            index = p * (n_samples - 1)
            floor_idx = jnp.clip(jnp.floor(index).astype(int), 0, n_samples - 1)
            ceil_idx = jnp.clip(jnp.ceil(index).astype(int), 0, n_samples - 1)
            floor_val = sorted_col[floor_idx]
            ceil_val = sorted_col[ceil_idx]
            weight = index - floor_idx
            return floor_val + weight * (ceil_val - floor_val)

        for j in range(n_features):
            column_vec = X[:, j]
            sorted_column = jnp.sort(column_vec)
            tol = 1e-4
            first_value = sorted_column[0]
            last_value = sorted_column[-1]
            is_constant = jnp.abs(first_value - last_value) < tol

            refs_j = jnp.where(
                is_constant,
                jnp.full((self.n_quantiles_,), first_value, dtype=jnp.float32),
                vmap(lambda p: compute_quantile(p, sorted_column))(target_quantiles),
            )
            references_ = references_.at[:, j].set(refs_j)

            quantiles_ = quantiles_.at[:, j].set(target_quantiles)

        self.references_ = references_
        self.quantiles_ = quantiles_
        return self

    def transform(self, X):
        """Transform the data to a uniform distribution. Input X must not contain NaNs.

        Args:
            X (array-like): Input data, shape (n_samples, n_features).
                           Must not contain NaN values.

        Returns:
            Transformed data (uniform distribution), shape (n_samples, n_features).
        """
        self._check_is_fitted()
        X = jnp.asarray(X)
        if X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}.")

        return _vmap_transform_features(X, self.quantiles_, self.references_)

    def inverse_transform(self, X_transformed):
        """Inverse transform the data from the uniform scale.

        Args:
            X_transformed (array-like): Transformed data (uniform scale),
                                         shape (n_samples, n_features).

        Returns:
            Data in the original scale, shape (n_samples, n_features).
        """
        self._check_is_fitted()
        X_transformed = jnp.asarray(X_transformed)
        if X_transformed.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {X_transformed.shape[1]}."
            )

        return _vmap_inverse_transform_features(
            X_transformed, self.quantiles_, self.references_
        )

    def fit_transform(self, X):
        """Fit the transformer and then transform the data to a uniform distribution.
        Input X must not contain NaNs."""
        return self.fit(X).transform(X)

    def _check_is_fitted(self):
        """Check if the transformer has been fitted."""
        if self.quantiles_ is None or self.references_ is None:
            raise RuntimeError(
                "Transformer is not fitted. Call the 'fit' method first."
            )


def _transform_single_feature(x_col, q_col, r_col):
    """Transform a single feature (column) to a uniform distribution."""

    is_constant_col = jnp.abs(r_col[0] - r_col[-1]) < 1e-4

    transformed_col = jnp.interp(x_col, r_col, q_col, left=0.0, right=1.0)

    constant_value = 0.0  # Hardcoded for uniform
    return jnp.where(is_constant_col, constant_value, transformed_col)


def _inverse_transform_single_feature(xt_col, q_col, r_col):
    """Inverse transform a single feature (column) from uniform scale."""

    is_constant_col = jnp.abs(r_col[0] - r_col[-1]) < 1e-4

    input_quantiles = xt_col  # Input is assumed to be on quantile scale [0, 1]

    input_quantiles = jnp.clip(input_quantiles, 0.0, 1.0)

    inversed_col = jnp.interp(input_quantiles, q_col, r_col)

    return jnp.where(is_constant_col, r_col[0], inversed_col)


_vmap_transform_features = vmap(
    _transform_single_feature, in_axes=(1, 1, 1), out_axes=1
)
_vmap_inverse_transform_features = vmap(
    _inverse_transform_single_feature, in_axes=(1, 1, 1), out_axes=1
)
