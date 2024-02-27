# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp


def label_binarize(y, *, classes, n_classes, neg_label=0, pos_label=1):
    """Binarize labels in a one-vs-all fashion.

    Parameters
    ----------
    y : {array-like}, shape (n_samples,)
        Input data.

    classes : {array-like}, shape (n_classes,)
        Uniquely holds the label for each class.

    n_classes : int
        Number of classes. SPU cannot support dynamic shape,
        so this parameter needs to be designated.

    neg_label : int, default=0
        Value with which negative labels must be encoded.

    pos_label : int, default=1
        Value with which positive labels must be encoded.

    Returns
    -------
    ndarray of shape (n_samples, n_classes)
        Shape will be (n_samples, 1) for binary problems.
    """
    eq_func = lambda x: jnp.where(classes == x, 1, 0)
    result = jax.vmap(eq_func)(y)

    if neg_label != 0 or pos_label != 1:
        result = jnp.where(result, pos_label, neg_label)

    if n_classes == 2:
        result = result[:, -1].reshape((-1, 1))
    return result


def _inverse_binarize_multiclass(y, classes):
    """Inverse label binarization transformation for multiclass.

    Multiclass uses the maximal score instead of a threshold.
    """
    return jnp.take(classes, y.argmax(axis=1), mode="clip")


def _inverse_binarize_thresholding(y, classes, threshold):
    """Inverse label binarization transformation using thresholding."""
    y = jnp.array(y > threshold, dtype=int)
    return classes[y[:, 1]]


class LabelBinarizer:
    """Binarize labels in a one-vs-all fashion.

    Firstly, use fit() to use an array to set the classes.
    The number of classes needs to be designated through parameter n_classes since SPU cannot support dynamic shape.
    The dynamic shape problem occurs when there are duplicated elements in input of fit function.
    The deduplication operation will cause complex computation, so it is not used by default.
    Noted that if unique==True, the order of the classes will be kept instead of sorted.

    Secondly, use transform() to convert the value to a one-hot label for classes.
    The input array needs to be 1d.  Users can directly use the transformation method like jax.ravel to transform
    the input array into 1d then use LabelBinarizer to do further transformation.

    Parameters
    ----------
    neg_label : int, default=0
        Value with which negative labels must be encoded.

    pos_label : int, default=1
        Value with which positive labels must be encoded.

    """

    def __init__(self, *, neg_label=0, pos_label=1):
        self.neg_label = neg_label
        self.pos_label = pos_label

    def fit(self, y, n_classes, unique=True):
        """Fit label binarizer.

        Parameters
        ----------
        y : {array-like}, shape (n_samples,)
            Input data.

        n_classes : int
            Number of classes. SPU cannot support dynamic shape,
            so this parameter needs to be designated.

        unique : bool
            Set to False to do deduplication on classes

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.neg_label >= self.pos_label:
            raise ValueError(
                f"neg_label={self.neg_label} must be strictly less than "
                f"pos_label={self.pos_label}."
            )
        if unique == True:
            self.classes_ = y
        else:
            # The output of jax needs to be tensor with known size.
            self.classes_ = jnp.unique(y, size=n_classes)
        self.n_classes_ = n_classes
        return self

    def fit_transform(self, y, n_classes, unique=True):
        """Fit label binarizer/transform multi-class labels to binary labels.

        Parameters
        ----------
        y : {array-like}, shape (n_samples,)
            Input data.

        n_classes : int
            Number of classes. SPU cannot support dynamic shape,
            so this parameter needs to be designated.

        unique : bool
            Set to False to do deduplication on classes

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems.
        """
        return self.fit(y, n_classes, unique=unique).transform(y)

    def transform(self, y):
        """Transform multi-class labels to binary labels.
        Parameters
        ----------
        y : {array-like}, shape (n_samples,)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems.
        """
        return label_binarize(
            y,
            classes=self.classes_,
            n_classes=self.n_classes_,
            neg_label=self.neg_label,
            pos_label=self.pos_label,
        )

    def inverse_transform(self, Y, threshold=None):
        """Transform binary labels back to multi-class labels.

        Parameters
        ----------
        Y : {array-like}, shape (n_samples, n_classes)
            Input data.

        threshold : float, default=None
            Threshold used in the binary cases.

        Returns
        -------
        ndarray of shape (n_samples,)

        """
        if threshold is None:
            threshold = (self.pos_label + self.neg_label) / 2.0
        if self.n_classes_ == 2:
            y_inv = _inverse_binarize_thresholding(Y, self.classes_, threshold)
        else:
            y_inv = _inverse_binarize_multiclass(Y, self.classes_)
        return y_inv


def binarize(X, *, threshold=0.0):
    """Binarize data (set feature values to 0 or 1) according to a threshold.

    Parameters
    ----------
    threshold : float, default=0.0
        Feature values below or equal to this are replaced by 0, above it by 1.

    """
    return jnp.where(X > threshold, 1, 0)


class Binarizer:
    """Binarize data (set feature values to 0 or 1) according to a threshold.

    Parameters
    ----------
    threshold : float, default=0.0
        Feature values below or equal to this are replaced by 0, above it by 1.

    """

    def __init__(self, *, threshold=0.0):
        self.threshold = threshold

    def transform(self, X):
        """Binarize each element of X.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data to binarize, element by element.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        return binarize(X, threshold=self.threshold)


def normalize(X, norm="l2"):
    """Scale input vectors individually to unit norm (vector length).

    Parameters
    ----------
    X : {array-like} of shape (n_samples, n_features)
        The data to normalize, element by element.

    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    Returns
    -------
    ndarray of shape (n_samples, n_features)
        Transformed array.
    """
    if norm == "l1":
        norms = jnp.abs(X).sum(axis=1)
        return X / norms[:, jnp.newaxis]
    elif norm == "l2":
        norms = jnp.einsum("ij,ij->i", X, X)
        norms = norms.astype(jnp.float32)
        # Use rsqrt instead of using combination of reciprocal and square for optimization
        return X * jax.lax.rsqrt(norms)[:, jnp.newaxis]
    elif norm == "max":
        norms = jnp.max(abs(X), axis=1)
        return X / norms[:, jnp.newaxis]


class Normalizer:
    """Normalize samples individually to unit norm.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample. If norm='max'
        is used, values will be rescaled by the maximum of the absolute
        values.
    """

    def __init__(self, norm="l2"):
        self.norm = norm

    def transform(self, X):
        """Scale each non zero row of X to unit norm.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data to normalize, row by row.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        return normalize(X, norm=self.norm)


class MinMaxScaler:
    """Transform features by scaling each feature to a given range.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.
    """

    def __init__(self, feature_range=(0, 1), *, clip=False):
        self.feature_range = feature_range
        self.clip = clip

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, *, zero_variance=False, contain_nan=False):
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        zero_variance : bool, default=False
            Set to True to handle the feature with zero variance, which will
            introduce additional computation.

        contain_nan : bool, default=False
            Set to True to handle the nan value.
            This option desiced whether to use nanmin and nanmax to compute the minimum
            and maximum.
        """
        self._reset()
        return self.partial_fit(X, zero_variance=zero_variance, contain_nan=contain_nan)

    def partial_fit(self, X, *, zero_variance=False, contain_nan=False):
        """
        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        zero_variance : bool, default=False
            Set to True to handle the feature with zero variance, which will
            introduce additional computation.

        contain_nan : bool, default=False
            Set to True to handle the nan value.
            This option desiced whether to use nanmin and nanmax to compute the minimum
            and maximum.
        """
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )
        first_pass = not hasattr(self, "n_samples_seen_")
        if contain_nan == False:
            data_min = jnp.min(X, axis=0)
            data_max = jnp.max(X, axis=0)
        else:
            data_min = jnp.nanmin(X, axis=0)
            data_max = jnp.nanmax(X, axis=0)
        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = jnp.minimum(self.data_min_, data_min)
            data_max = jnp.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]
        data_range = data_max - data_min
        if zero_variance == False:
            self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        else:
            self.scale_ = (feature_range[1] - feature_range[0]) / jnp.where(
                data_range == 0, 1, data_range
            )

        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        """Scale features of X according to feature_range.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Input data that will be transformed.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        X *= self.scale_
        X += self.min_
        if self.clip:
            X = jnp.clip(X, self.feature_range[0], self.feature_range[1])
        return X

    def fit_transform(self, X, *, zero_variance=False, contain_nan=False):
        """fit and transform with same input data.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        zero_variance : bool, default=False
            Set to True to handle the feature with zero variance, , which will
            introduce additional computation.

        contain_nan : bool, default=False
            Set to True to handle the nan value.
            This option desiced whether to use nanmin and nanmax to compute the minimum
            and maximum.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        return self.fit(
            X, zero_variance=zero_variance, contain_nan=contain_nan
        ).transform(X)

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        X -= self.min_
        X /= self.scale_
        return X


class MaxAbsScaler:
    """Scale each feature by its maximum absolute value."""

    def __init__(self):
        pass

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        if hasattr(self, "scale_"):
            del self.scale_
            del self.n_samples_seen_
            del self.max_abs_

    def fit(self, X, zero_maxabs=False, contain_nan=False):
        """Compute the maximum absolute value to be used for later scaling.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        zero_maxabs : bool, default=False
            Set to True to handle the feature with maximum absolute value of zero, which
            will introduce additional computation.

        contain_nan : bool, default=False
            Set to True to handle the nan value.
            This option desiced whether to use nanmin and nanmax to compute the minimum
            and maximum.
        """
        self._reset()
        return self.partial_fit(X, zero_maxabs=zero_maxabs, contain_nan=contain_nan)

    def partial_fit(self, X, *, zero_maxabs=False, contain_nan=False):
        """
        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        zero_maxabs : bool, default=False
            Set to True to handle the feature with maximum absolute value of zero, which
            will introduce additional computation.

        contain_nan : bool, default=False
            Set to True to handle the nan value.
            This option desiced whether to use nanmin and nanmax to compute the minimum
            and maximum.
        """
        first_pass = not hasattr(self, "n_samples_seen_")
        if contain_nan == False:
            max_abs = jnp.max(jnp.abs(X), axis=0)
        else:
            max_abs = jnp.nanmax(jnp.abs(X), axis=0)
        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            max_abs = jnp.maximum(self.max_abs_, max_abs)
            self.n_samples_seen_ += X.shape[0]
        self.max_abs_ = max_abs
        if zero_maxabs == False:
            self.scale_ = max_abs
        else:
            self.scale_ = jnp.where(max_abs == 0, 1, max_abs)
        return self

    def transform(self, X):
        """Scale the data.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data that should be scaled.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        return X / self.scale_

    def fit_transform(self, X, *, zero_maxabs=False, contain_nan=False):
        """fit and transform with same input data.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        zero_maxabs : bool, default=False
            Set to True to handle the feature with maximum absolute value of zero, which
            will introduce additional computation.

        contain_nan : bool, default=False
            Set to True to handle the nan value.
            This option desiced whether to use nanmin and nanmax to compute the minimum
            and maximum.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        return self.fit(X, zero_maxabs=zero_maxabs, contain_nan=contain_nan).transform(
            X
        )

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data that should be transformed back.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        return X * self.scale_


class KBinsDiscretizer:
    """Bin continuous data into intervals.

    Attribute encode is not implemented, since there is currently no onehotencoder
    in sml.
    Attribute subsample is not implemented, since random choise in SPU runtime does
    not work as expected.

    Parameters
    ----------
    n_bins : int, default=5
        The number of bins to produce. n_bins should be int >= 2.
        If diverse_n_bins is not None, n_bins should be int <= max(diverse_n_bins).

    diverse_n_bins : {array-like} of shape (n_features,) or None, default=None
        By default, all features are binned in the same number of bins.
        When diverse_n_bins is not None, it should be an array-like of shape
        (n_features,), which tells the number of bins for each feature.
        The elements in diverse_n_bins should be int >= 2 and <= n_bins.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used to define the widths of the bins.

        - 'uniform': All bins in each feature have identical widths.
        - 'quantile': All bins in each feature have the same number of points.
        - 'kmeans': Values in each bin have the same nearest center of a 1D
          k-means cluster.

        Note that the strategy 'kmeans' is not supported when diverse_n_bins is
        not None. Currently, there is no efficient solution to solve dynamic shape
        problem in this requirement.

    """

    def __init__(self, n_bins=5, diverse_n_bins=None, *, strategy="quantile"):
        self.n_bins = n_bins
        self.diverse_n_bins = diverse_n_bins
        self.strategy = strategy

    def fit(self, X, sample_weight=None, *, remove_bin=False, remove_ref=1e-3):
        """Fit the estimator.

        JAX does not support array with dynamic shape or type object, so the feature which
        has different number of n_bins will be presented by duplicated elements in bin_edges.

        Note that the remove_bin here will also influence transform function.
        Note that there is currently no support for handling constant values in a feature
        , since it introduces much redundant boolean computation because dynamic shape is
        not supported.
        In sklarn, feature with constant value will be replaced with 0 after transformation.
        (see https://github.com/scikit-learn/scikit-learn/blob/d139ff234b0f8ec30287e26e0bc801bdafdfbb1a/sklearn/preprocessing/tests/test_discretization.py#L192)

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Data to be discretized.

        sample_weight : {array-like} of shape (n_samples,)
            Contains weight values to be associated with each sample.
            Only possible when `strategy` is set to `"quantile"`.

        remove_bin : bool, default=False
            Set to True to remove bins with the small interval, which is less than
            remove_ref. This option is only available when `strategy` is set to
            `"quantile"` or `"kmeans"`.

        remove_ref : float, default=1e-3
            The reference value to remove the small interval. This option is only
            available when `strategy` is set to `"quantile"` or `"kmeans"` and
            remove_bin is set to True.
        """
        self.remove_bin = remove_bin
        if sample_weight is not None and self.strategy == "uniform":
            raise ValueError(
                "`sample_weight` was provided but it cannot be "
                "used with strategy='uniform'. Got strategy="
                f"{self.strategy!r} instead."
            )
        n_bins = self.n_bins

        if self.diverse_n_bins == None:
            if self.strategy == "uniform":
                bin_func = lambda x: jnp.linspace(jnp.min(x), jnp.max(x), n_bins + 1)
                bin_edges = jax.vmap(bin_func, in_axes=1, out_axes=1)(X)

            if self.strategy == "quantile":
                quantiles = jnp.linspace(0, 100, n_bins + 1)
                if sample_weight is None:
                    bin_func = lambda x: jnp.percentile(x, quantiles)
                    bin_edges = jax.vmap(bin_func, in_axes=1, out_axes=1)(X)

                else:

                    def bin_func(x):
                        def _weighted_percentile(x, q, w):
                            x = x.reshape((-1, 1))
                            if x.shape != w.shape and x.shape[0] == w.shape[0]:
                                w = jnp.tile(w, (x.shape[1], 1)).T
                            sorted_idx = jnp.argsort(x, axis=0)
                            sorted_weights = jnp.take_along_axis(w, sorted_idx, axis=0)
                            weight_cdf = jnp.cumsum(sorted_weights, axis=0)
                            adjusted_percentile = q / 100 * weight_cdf[-1]

                            def searchsorted_element(x_inner):
                                encoding = jnp.where(
                                    x_inner >= weight_cdf[0:-1, 0], 1, 0
                                )
                                return jnp.sum(encoding)

                            percentile_idx = jax.vmap(searchsorted_element)(
                                adjusted_percentile
                            )
                            col_index = jnp.arange(x.shape[1])
                            percentile_in_sorted = sorted_idx[percentile_idx, col_index]
                            percentile = x[percentile_in_sorted, col_index]
                            return percentile[0]

                        return jax.vmap(_weighted_percentile, (None, 0, None))(
                            x, quantiles, sample_weight
                        )

                    bin_edges = jax.vmap(bin_func, in_axes=1, out_axes=1)(X)

            if self.strategy == "kmeans":
                from ..cluster.kmeans import KMEANS

                def bin_func(x, KMeans):
                    x = x[:, None]
                    col_min = jnp.min(x)
                    col_max = jnp.max(x)
                    uniform_edges = jnp.linspace(col_min, col_max, n_bins + 1)
                    init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
                    km = KMeans(
                        n_clusters=n_bins, n_samples=x.shape[0], init=init, max_iter=10
                    )
                    km.fit(x)
                    centers = jnp.sort(km._centers[:, 0])
                    return jnp.r_[col_min, (centers[1:] + centers[:-1]) * 0.5, col_max]

                bin_edges = jax.vmap(bin_func, in_axes=(1, None), out_axes=1)(X, KMEANS)

            ### remove the small interval by iteratively comparing the adjacent bin edges
            ### and remove the small interval by setting the bin edge to the adjacent one.
            ### unqiue_count is used to record the number of unique bin edges for each feature
            ### which is used in transform function.
            if remove_bin == True and self.strategy in ("quantile", "kmeans"):

                def eliminate_func(x):
                    n = x.shape[0]

                    def loop_body(i, st):
                        count, x = st
                        ### Not sure whether to add abs here. Though the element in x bin_edges shuld be incremental, the precision problem of MPC may cuase additional unexpected behavior.
                        pred = (x[i] - x[i - 1]) >= remove_ref
                        x = jax.lax.cond(
                            pred, lambda _: x.at[count].set(x[i]), lambda _: x, count
                        )
                        count = jax.lax.cond(pred, lambda c: c + 1, lambda c: c, count)
                        return count, x

                    st = (1, x)
                    count, x = jax.lax.fori_loop(1, n, loop_body, st)
                    return x, count

                bin_edges, unqiue_count = jax.vmap(
                    eliminate_func, in_axes=1, out_axes=(1, 0)
                )(bin_edges)
                self.unqiue_count = unqiue_count

        else:
            diverse_n_bins = self.diverse_n_bins
            ### directly using jnp.linspace will cause dynamic shape problem,
            ### so we need to use jnp.arange and a branch function jnp.where
            if self.strategy == "uniform":
                arrange_array = jnp.arange(n_bins + 1)

                def bin_func(x, diverse_n_bin, arrange_array):
                    minval = jnp.min(x)
                    maxval = jnp.max(x)
                    delta = (maxval - minval) / diverse_n_bin

                    def bin_element_func(x_inner):
                        return jnp.where(
                            x_inner <= diverse_n_bin, minval + x_inner * delta, maxval
                        )

                    return jax.vmap(bin_element_func)(arrange_array)

                bin_edges = jax.vmap(bin_func, in_axes=(1, 0, None), out_axes=1)(
                    X, diverse_n_bins, arrange_array
                )

            ### Note that here we do not use length of qunaitle to control the number of bins,
            ### which will cause a dynamic shape problem.
            ### Here we use a duplicated number in qunaitle to control the number of bins
            ### Since there is precision problem in MPC, there may be undiscovered unexpected behavior.
            if self.strategy == "quantile":
                arrange_array = jnp.arange(n_bins + 1)
                if sample_weight is None:

                    def bin_func(x, diverse_n_bin):
                        delta = 100 / diverse_n_bin

                        def quantiles_func(x_inner):
                            return jnp.where(
                                x_inner <= diverse_n_bin, x_inner * delta, 100
                            )

                        quantiles = jax.vmap(quantiles_func)(arrange_array)
                        return jnp.percentile(x, quantiles)

                    bin_edges = jax.vmap(bin_func, in_axes=(1, 0), out_axes=1)(
                        X, diverse_n_bins
                    )

                else:

                    def bin_func(x, diverse_n_bin):
                        delta = 100 / diverse_n_bin

                        def quantiles_func(x_inner):
                            return jnp.where(
                                x_inner <= diverse_n_bin, x_inner * delta, 100
                            )

                        quantiles = jax.vmap(quantiles_func)(arrange_array)

                        def _weighted_percentile(x, q, w):
                            x = x.reshape((-1, 1))
                            if x.shape != w.shape and x.shape[0] == w.shape[0]:
                                w = jnp.tile(w, (x.shape[1], 1)).T
                            sorted_idx = jnp.argsort(x, axis=0)
                            sorted_weights = jnp.take_along_axis(w, sorted_idx, axis=0)
                            weight_cdf = jnp.cumsum(sorted_weights, axis=0)
                            adjusted_percentile = q / 100 * weight_cdf[-1]

                            def searchsorted_element(x_inner):
                                encoding = jnp.where(
                                    x_inner >= weight_cdf[0:-1, 0], 1, 0
                                )
                                return jnp.sum(encoding)

                            percentile_idx = jax.vmap(searchsorted_element)(
                                adjusted_percentile
                            )
                            col_index = jnp.arange(x.shape[1])
                            percentile_in_sorted = sorted_idx[percentile_idx, col_index]
                            percentile = x[percentile_in_sorted, col_index]
                            return percentile[0]

                        return jax.vmap(_weighted_percentile, (None, 0, None))(
                            x, quantiles, sample_weight
                        )

                    bin_edges = jax.vmap(bin_func, in_axes=(1, 0), out_axes=1)(
                        X, diverse_n_bins
                    )
            if self.strategy == "kmeans":
                raise ValueError(
                    " Currently, the strategy 'kmeans' is not supported when diverse_n_bins is not None."
                )
                ### The following code is correct for some cases, but it is not a general solution.
                # from ..cluster.kmeans import KMEANS
                # arrange_array = jnp.arange(n_bins + 1)
                # def bin_func(x, KMeans, diverse_n_bin):
                #     x = x[:, None]
                #     col_min = jnp.min(x)
                #     col_max = jnp.max(x)
                #     delta = (col_max - col_min) / diverse_n_bin
                #     def bin_element_func(x_inner):
                #         return jnp.where(x_inner <= diverse_n_bin, col_min + x_inner * delta, col_max)
                #     uniform_edges = jax.vmap(bin_element_func)(arrange_array)
                #     def uniform_func(x_inner):
                #         return jnp.where(x_inner < diverse_n_bin, (uniform_edges[x_inner] + uniform_edges[x_inner + 1]) * 0.5, (uniform_edges[diverse_n_bin - 1] + uniform_edges[diverse_n_bin]) * 0.5)
                #     init = jax.vmap(uniform_func)(arrange_array[:-1])[:, None]
                #     km = KMeans(n_clusters=n_bins, n_samples=x.shape[0], init=init, max_iter=10)
                #     km.fit(x)
                ### Though it seems to successfuly control the number of centers,
                ### the problem here is the KMENAS will ruturn invalid center as 0
                ### The idea now is to use predict to get the valid centers.
                #     centers = jnp.sort(km._centers[:, 0])
                #     return jnp.r_[col_min, (centers[1:] + centers[:-1]) * 0.5, col_max]
                # bin_edges = jax.vmap(bin_func, in_axes=(1, None, 0), out_axes=1)(X, KMEANS, diverse_n_bins)

            ### remove the small interval by iteratively comparing the adjacent bin edges
            ### and remove the small interval by setting the bin edge to the adjacent one.
            ### unqiue_count is used to record the number of unique bin edges for each feature
            ### which is used in transform function.
            if remove_bin == True and self.strategy in ("quantile", "kmeans"):

                def eliminate_func(x):
                    n = x.shape[0]

                    def loop_body(i, st):
                        count, x = st
                        ### Not sure whether to add abs here. Though the element in x bin_edges shuld be incremental, the precision problem of MPC may cuase addition unexpected behavior.
                        pred = (x[i] - x[i - 1]) >= remove_ref
                        x = jax.lax.cond(
                            pred, lambda _: x.at[count].set(x[i]), lambda _: x, count
                        )
                        count = jax.lax.cond(pred, lambda c: c + 1, lambda c: c, count)
                        return count, x

                    st = (1, x)
                    count, x = jax.lax.fori_loop(1, n, loop_body, st)
                    return x, count

                bin_edges, unqiue_count = jax.vmap(
                    eliminate_func, in_axes=1, out_axes=(1, 0)
                )(bin_edges)
                self.unqiue_count = unqiue_count
            else:
                self.unqiue_count = diverse_n_bins + 1

        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins

        return self

    def transform(self, X):
        """
        Discretize the data.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        bin_edges = self.bin_edges_

        if self.diverse_n_bins != None or (
            self.remove_bin == True and self.strategy in ("quantile", "kmeans")
        ):
            unqiue_count = self.unqiue_count

            def compute_row(bin, x, c):
                def compute_element(x):
                    encoding = jnp.where(x >= bin[1:-1], 1, 0)
                    return jnp.clip(jnp.sum(encoding), 0, c - 2)

                return jax.vmap(compute_element)(x)

            compute_rows_vmap = jax.vmap(compute_row, in_axes=(1, 1, 0), out_axes=1)(
                bin_edges, X, unqiue_count
            )
            return compute_rows_vmap
        else:

            def compute_row(bin, x):
                def compute_element(x):
                    encoding = jnp.where(x >= bin[1:-1], 1, 0)
                    return jnp.sum(encoding)

                return jax.vmap(compute_element)(x)

            compute_rows_vmap = jax.vmap(compute_row, in_axes=(1, 1), out_axes=1)(
                bin_edges, X
            )
            return compute_rows_vmap

    def fit_transform(
        self, X, sample_weight=None, *, remove_bin=False, remove_ref=1e-3
    ):
        """fit and transform with same input data.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Data to be discretized.

        sample_weight : {array-like} of shape (n_samples,)
            Contains weight values to be associated with each sample.
            Only possible when `strategy` is set to `"quantile"`.

        remove_bin : bool, default=False
            Set to True to remove bins with the small interval, which is less than
            remove_ref. This option is only available when `strategy` is set to
            `"quantile"` or `"kmeans"`.

        remove_ref : float, default=1e-3
            The reference value to remove the small interval. This option is only
            available when `strategy` is set to `"quantile"` or `"kmeans"` and
            remove_bin is set to True.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        return self.fit(
            X, sample_weight=sample_weight, remove_bin=remove_bin, remove_ref=remove_ref
        ).transform(X)

    def inverse_transform(self, X):
        """Transform discretized data back to original feature space.

        Note that this function does not regenerate the original data
        due to discretization rounding.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Transformed data in the binned space.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Data in the original feature space.
        """
        bin_edges = self.bin_edges_

        def bin_func(bin, x):
            bin_edges = bin
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            return bin_centers[(x).astype(jnp.int32)]

        return jax.vmap(bin_func, in_axes=(1, 1), out_axes=1)(bin_edges, X)
