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

import warnings

import jax
import jax.numpy as jnp
from jax import vmap


class QuantileTransformer:
    def __init__(
        self,
        n_quantiles=1000,
        output_distribution='uniform',
        subsample=100_000,
    ):
        """Initialize the transformer.

        Transforms features using quantiles information. This implementation focuses
        on mapping to a uniform distribution.

        Args:
            n_quantiles (int): The number of quantiles to be computed. It corresponds
                               to the number of landmarks used to discretize the
                               cumulative distribution function. Defaults to 1000.
            output_distribution (str): Marginal distribution for the transformed data.
                                       Currently only 'uniform' is implemented and enforced.
                                       Defaults to 'uniform'.
            subsample (int): Maximum number of samples used to estimate the quantiles
                             for computational efficiency. Note: subsampling may lead to
                             a less precise transformation. If an int is provided, the first
                             `subsample` samples will be used. Defaults to 100,000.

        Raises:
            ValueError: If parameters are invalid.

        Note:
            - This transformer assumes the input data contains no NaN values.
              The user **must** handle any NaNs (e.g., through imputation or removal)
              *before* passing the data to this transformer's methods (`fit`, `transform`).
              If the input data contains NaNs, the behavior is undefined and may lead
              to errors or incorrect results (`jnp.percentile` might error or return NaN).
            - The `output_distribution` parameter is currently ignored, and the output
              is always uniform.
        """
        if not isinstance(n_quantiles, int) or n_quantiles <= 0:
            raise ValueError(
                f"n_quantiles must be a positive integer, got {n_quantiles}."
            )
        if output_distribution != 'uniform':
            warnings.warn(
                f"output_distribution='{output_distribution}' is not supported. "
                f"Using 'uniform' instead.",
                UserWarning,
            )
            self.output_distribution = 'uniform'
        else:
            self.output_distribution = output_distribution
        if not isinstance(subsample, int) or subsample <= 0:
            raise ValueError(f"subsample must be a positive integer, got {subsample}.")

        self.n_quantiles = n_quantiles
        self.subsample = subsample

        self.quantiles_ = None
        self.references_ = None
        self.n_quantiles_ = None
        self._n_features_in = None
        self._input_shape = None

    def fit(self, X):
        """Compute the quantiles used for transforming.

        Args:
            X (array-like): Input data, shape (n_samples, n_features).
                           Must not contain NaN values.

        Returns:
            self: The fitted transformer.

        Raises:
            ValueError: If X is not a 2D array.
        """
        X = jnp.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Input array X must be 2D, got {X.ndim} dimensions.")

        n_samples, n_features = X.shape
        self._n_features_in = n_features
        self._input_shape = X.shape

        actual_n_samples = n_samples
        if self.subsample < n_samples:
            actual_n_samples = self.subsample
            X_subset = X[: self.subsample, :]
        else:
            X_subset = X

        self.n_quantiles_ = max(1, min(self.n_quantiles, actual_n_samples))

        target_quantiles_prob = jnp.linspace(0, 1, self.n_quantiles_, dtype=jnp.float32)
        self.quantiles_ = target_quantiles_prob

        references_perc = target_quantiles_prob * 100

        empirical_quantiles = jnp.percentile(
            X_subset, q=references_perc, axis=0, method='linear'
        )

        self.references_ = empirical_quantiles

        return self

    def transform(self, X):
        """Transform the data to a uniform distribution using computed quantiles.

        Args:
            X (array-like): Input data, shape (n_samples, n_features).
                           Must not contain NaN values.

        Returns:
            Transformed data (uniform distribution), shape (n_samples, n_features).

        Raises:
            RuntimeError: If the transformer has not been fitted.
            ValueError: If the number of features in X does not match the fitted data.
        """
        self._check_is_fitted()
        X = jnp.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Input array X must be 2D, got {X.ndim} dimensions.")
        if X.shape[1] != self._n_features_in:
            raise ValueError(
                f"Input has {X.shape[1]} features, but QuantileTransformer "
                f"was fitted with {self._n_features_in} features."
            )

        return _vmap_transform_features(X, self.quantiles_, self.references_)

    def inverse_transform(self, X_transformed):
        """Inverse transform the data from the uniform scale back to the original scale.

        Args:
            X_transformed (array-like): Transformed data (uniform scale, values assumed ~[0, 1]),
                                         shape (n_samples, n_features).

        Returns:
            Data in the original scale, shape (n_samples, n_features).

        Raises:
            RuntimeError: If the transformer has not been fitted.
            ValueError: If the number of features in X_transformed does not match the fitted data.
        """
        self._check_is_fitted()
        X_transformed = jnp.asarray(X_transformed)

        if X_transformed.ndim != 2:
            raise ValueError(
                f"Input array X_transformed must be 2D, got {X_transformed.ndim} dimensions."
            )
        if X_transformed.shape[1] != self._n_features_in:
            raise ValueError(
                f"Input has {X_transformed.shape[1]} features, but QuantileTransformer "
                f"was fitted with {self._n_features_in} features."
            )

        return _vmap_inverse_transform_features(
            X_transformed, self.quantiles_, self.references_
        )

    def fit_transform(self, X):
        """Fit the transformer and then transform the data.

        Args:
            X (array-like): Input data, shape (n_samples, n_features).
                           Must not contain NaN values.

        Returns:
            Transformed data (uniform distribution), shape (n_samples, n_features).
        """
        return self.fit(X).transform(X)

    def _check_is_fitted(self):
        """Check if the transformer has been fitted."""
        if self.quantiles_ is None or self.references_ is None:
            raise RuntimeError(
                "This QuantileTransformer instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )


def _transform_single_feature(x_col, target_quantiles_prob, empirical_quantiles_vals):
    """Transform a single feature (column) to a uniform distribution [0, 1].

    Args:
        x_col: Input data column (n_samples,).
        target_quantiles_prob: Target quantiles (probabilities, linspace 0 to 1) (n_quantiles_,).
        empirical_quantiles_vals: Data values corresponding to target_quantiles_prob (n_quantiles_,).

    Returns:
        Transformed data column mapped to [0, 1].
    """

    transformed_col = jnp.interp(
        x_col,
        empirical_quantiles_vals,  # xp: Data values at known quantiles
        target_quantiles_prob,  # fp: The known quantiles (0 to 1)
        left=0.0,  # Map values below min reference to 0
        right=1.0,  # Map values above max reference to 1
    )

    return transformed_col


def _inverse_transform_single_feature(
    xt_col, target_quantiles_prob, empirical_quantiles_vals
):
    """Inverse transform a single feature (column) from uniform scale [0, 1] to original scale.

    Args:
        xt_col: Transformed data column (uniform scale, ~[0, 1]) (n_samples,).
        target_quantiles_prob: Target quantiles (probabilities, linspace 0 to 1) (n_quantiles_,).
        empirical_quantiles_vals: Data values corresponding to target_quantiles_prob (n_quantiles_,).

    Returns:
        Data column mapped back to the original scale.
    """

    input_quantiles = jnp.clip(xt_col, 0.0, 1.0)

    inversed_col = jnp.interp(
        input_quantiles,  # x: Values to interpolate (quantiles 0-1)
        target_quantiles_prob,  # xp: Reference points (quantiles 0-1)
        empirical_quantiles_vals,  # fp: Corresponding values (original data scale)
    )

    return inversed_col


_vmap_transform_features = vmap(
    _transform_single_feature, in_axes=(1, None, 1), out_axes=1
)

_vmap_inverse_transform_features = vmap(
    _inverse_transform_single_feature, in_axes=(1, None, 1), out_axes=1
)
