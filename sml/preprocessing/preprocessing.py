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

class BrierScoreLoss:
    """
    Compute the Brier score loss using JAX.
    Ensures that calculations are performed in float64 for precision consistency with sklearn.
    """
    
    def __init__(self, pos_label=1):
        """
        Initialize the BrierScoreLoss class.
        
        Parameters
        ----------
        pos_label : int, default=1
            The positive label to be considered in the binary classification.
        """
        self.pos_label = pos_label

    def compute(self, y_true, y_proba, sample_weight=None):
        """
        Compute the Brier score loss.
        
        Parameters
        ----------
        y_true : array-like, shape (n_samples,)
            True binary labels.

        y_proba : array-like, shape (n_samples,)
            Predicted probabilities.

        sample_weight : array-like, shape (n_samples,), optional
            Sample weights.
        
        Returns
        -------
        float
            The computed Brier score loss.
        """
        y_true = jnp.asarray(y_true, dtype=jnp.float64)  # 强制 float64
        y_proba = jnp.asarray(y_proba, dtype=jnp.float64)
        
        y_true = jnp.where(y_true == self.pos_label, 1.0, 0.0)
        loss = (y_proba - y_true) ** 2
        
        if sample_weight is not None:
            sample_weight = jnp.asarray(sample_weight, dtype=jnp.float64)
            loss = loss * sample_weight
            return jnp.sum(loss) / jnp.sum(sample_weight)
        else:
            return jnp.mean(loss)

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
    eq_func = lambda x: classes == x
    result = jax.vmap(eq_func)(y).astype(jnp.int_)

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
    return (X > threshold).astype(jnp.int_)


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
            This option decides whether to use nanmin and nanmax to compute the minimum
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
            This option decides whether to use nanmin and nanmax to compute the minimum
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
            This option decides whether to use nanmin and nanmax to compute the minimum
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
            This option decides whether to use nanmin and nanmax to compute the minimum
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
            This option decides whether to use nanmin and nanmax to compute the minimum
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
            This option decides whether to use nanmin and nanmax to compute the minimum
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


def _weighted_percentile(x, q, w):
    """Compute weighted percentile

    Parameters
    ----------
    x : {array-like} of shape (n_samples,)
        Values to take the weighted percentile of.

    q : int
        Percentile to compute. Must be value between 0 and 100.

    w : {array-like} of shape (n_samples,)
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)`.

    Returns
    -------
    {array-like} of shape (n_bins + 1, )
        Weighted percentile.
    """
    x = x.reshape((-1, 1))
    if x.shape != w.shape and x.shape[0] == w.shape[0]:
        w = jnp.tile(w, (x.shape[1], 1)).T
    sorted_idx = jnp.argsort(x, axis=0)
    sorted_weights = jnp.take_along_axis(w, sorted_idx, axis=0)
    weight_cdf = jnp.cumsum(sorted_weights, axis=0)
    adjusted_percentile = q / 100 * weight_cdf[-1]

    def searchsorted_element(x_inner):
        encoding = x_inner >= weight_cdf[0:-1, 0]
        return jnp.sum(encoding)

    percentile_idx = jax.vmap(searchsorted_element)(adjusted_percentile)
    col_index = jnp.arange(x.shape[1])
    percentile_in_sorted = sorted_idx[percentile_idx, col_index]
    percentile = x[percentile_in_sorted, col_index]
    return percentile[0]


def _kmeans_bin_func(x, n_bins, minval, maxval, KMEANS):
    """Compute bins using k-means clustering."""
    uniform_edges = jnp.linspace(minval, maxval, n_bins + 1)
    init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
    km = KMEANS(n_clusters=n_bins, n_samples=x.shape[0], init=init, max_iter=10)
    km.fit(x)
    centers = jnp.sort(km._centers[:, 0])
    return jnp.r_[minval, (centers[1:] + centers[:-1]) * 0.5, maxval]


def _remove_bin_func(bin_edges, remove_ref):
    """
    Remove the small interval by iteratively comparing the adjacent bin edges
    and remove the small interval by setting the bin edge to the adjacent one.

    Parameters
    ----------
    bin_edges : {array-like} of shape (n_bins + 1, n_features)
        The bin edges for each feature.

    remove_ref : float
        The reference value to remove the small interval.

    Returns
    -------
    bin_edges : {array-like} of shape (n_bins + 1, n_features)
        The bin edges for each feature after removing the small interval.

    count : {array-like} of shape (n_features,)
        The number of unique bin edges for each feature.
    """
    n = bin_edges.shape[0]

    def eliminate_func(x):
        def loop_body(i, st):
            count, x = st
            pred = (x[i] - x[i - 1]) >= remove_ref
            x = jax.lax.cond(pred, lambda _: x.at[count].set(x[i]), lambda _: x, count)
            count = jax.lax.cond(pred, lambda c: c + 1, lambda c: c, count)
            return count, x

        st = (1, x)
        count, x = jax.lax.fori_loop(1, n, loop_body, st)
        return x, count

    return jax.vmap(eliminate_func, in_axes=1, out_axes=(1, 0))(bin_edges)


class KBinsDiscretizer:
    """Bin continuous data into intervals.

    Attribute encode is not implemented, since there is currently no OneHotEncoder
    in sml.
    Attribute subsample is not implemented, since random choice in SPU runtime does
    not work as expected.

    Parameters
    ----------
    n_bins : int, default=5
        The number of bins to produce. n_bins should be int >= 2.
        If diverse_n_bins is not None, n_bins should be int = max(diverse_n_bins).
        Though n_bins can be int > max(diverse_n_bins), it will introduce much redundant
        computation when removing bins with small interval. So it is not recommended but
        feasible for some specific requirement.

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

    def fit(
        self,
        X,
        sample_weight=None,
        vectorize=True,
        *,
        remove_bin=False,
        remove_ref=1e-3,
    ):
        """Fit the estimator.

        JAX does not support array with dynamic shape or type object, so the feature which
        has different number of n_bins will be presented by duplicated elements in bin_edges.

        Note that the remove_bin here will also influence transform function.
        Note that there is currently no support for handling constant values in a feature
        , since it introduces much redundant boolean computation because dynamic shape is
        not supported.
        In sklearn, feature with constant value will be replaced with 0 after transformation.
        (see https://github.com/scikit-learn/scikit-learn/blob/d139ff234b0f8ec30287e26e0bc801bdafdfbb1a/sklearn/preprocessing/tests/test_discretization.py#L192)

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Data to be discretized.

        sample_weight : {array-like} of shape (n_samples,)
            Contains weight values to be associated with each sample.
            Only possible when `strategy` is set to `"quantile"`.

        vectorize : bool, default=True
            Set to False to disable vectorization when diverse_n_bins is not
            None. The vectorized version supports computation with encrypted
            diverse_n_bins, in comparison, the non-vectorized version can only
            compute with diverse_n_bins of type numpy.ndarray (note that array
            created from jax.numpy will also create problems).
            There is currently no support for a vectorized form of kmeans.
            In addition, the vectorized version shows different performance and
            precision error with the non-vectorized version.
            Generally speaking, the vectorized version introduces more
            computation to enable more parallelization, which means more data
            sent and fewer send actions. However, when there is a great value
            gap between the n_bins and elements in diverse_n_bins, the
            vectorized version will introduce too much redundant computation
            that makes both data sent and send actions exceed the cost of the
            non-vectorized version. A special case is that vectorized form of
            quantile with sample_weight always shows more data sent and send
            actions than non-vectorized version.
            The vectorized version shows slightly less precision error than
            the non-vectorized version.
            Users can choose the version according to their specific requirements.

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

        if self.diverse_n_bins is None:
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
                        return jax.vmap(_weighted_percentile, (None, 0, None))(
                            x, quantiles, sample_weight
                        )

                    bin_edges = jax.vmap(bin_func, in_axes=1, out_axes=1)(X)

            if self.strategy == "kmeans":
                from ..cluster.kmeans import KMEANS

                def bin_func(x, KMEANS):
                    x = x[:, None]
                    minval = jnp.min(x)
                    maxval = jnp.max(x)
                    return _kmeans_bin_func(x, n_bins, minval, maxval, KMEANS)

                bin_edges = jax.vmap(bin_func, in_axes=(1, None), out_axes=1)(X, KMEANS)

            ### remove the small interval
            ### unqiue_count is used to record the number of unique bin edges for each feature
            ### which is used in transform function.
            if remove_bin == True and self.strategy in ("quantile", "kmeans"):
                bin_edges, unqiue_count = _remove_bin_func(bin_edges, remove_ref)
                self.unqiue_count = unqiue_count

        elif vectorize == True:
            diverse_n_bins = self.diverse_n_bins
            ### directly using jnp.linspace will cause dynamic shape problem,
            ### so we need to use jnp.arange with a public value n_bins
            if self.strategy == "uniform":
                arrange_array = jnp.arange(n_bins + 1)

                def bin_func(x, diverse_n_bin, arrange_array):
                    minval = jnp.min(x)
                    maxval = jnp.max(x)
                    delta = (maxval - minval) / diverse_n_bin

                    def bin_element_func(x_inner):
                        return minval + x_inner * delta

                    return jax.vmap(bin_element_func)(arrange_array)

                bin_edges = jax.vmap(bin_func, in_axes=(1, 0, None), out_axes=1)(
                    X, diverse_n_bins, arrange_array
                )

            ### Note that here we do not use length of qunaitle to control the number of bins,
            ### which will cause a dynamic shape problem.
            ### Here we use a duplicated number in qunaitle to control the number of bins
            ### Since there is precision problem in MPC, there may be undiscovered unexpected behavior.
            if self.strategy == "quantile":
                if sample_weight is None:
                    arrange_array = jnp.arange(n_bins + 1)

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
                    arrange_array = jnp.arange(n_bins + 1)

                    def bin_func(x, diverse_n_bin):
                        delta = 100 / diverse_n_bin

                        def quantiles_func(x_inner):
                            return jnp.where(
                                x_inner <= diverse_n_bin, x_inner * delta, 100
                            )

                        quantiles = jax.vmap(quantiles_func)(arrange_array)

                        return jax.vmap(_weighted_percentile, (None, 0, None))(
                            x, quantiles, sample_weight
                        )

                    bin_edges = jax.vmap(bin_func, in_axes=(1, 0), out_axes=1)(
                        X, diverse_n_bins
                    )

            if self.strategy == "kmeans":
                raise ValueError(
                    " Currently, the strategy 'kmeans' is not supported when diverse_n_bins is not None and vectorize=True."
                )
                ### The following code is correct for some cases, but it is not a general solution.
                # from ..cluster.kmeans import KMEANS
                # arrange_array = jnp.arange(n_bins + 1)
                # def bin_func(x, KMeans, diverse_n_bin):
                #     x = x[:, None]
                #     minval = jnp.min(x)
                #     maxval = jnp.max(x)
                #     delta = (maxval - minval) / diverse_n_bin
                #     def bin_element_func(x_inner):
                #         return jnp.where(x_inner <= diverse_n_bin, minval + x_inner * delta, maxval)
                #     uniform_edges = jax.vmap(bin_element_func)(arrange_array)
                #     def uniform_func(x_inner):
                #         return jnp.where(x_inner < diverse_n_bin, (uniform_edges[x_inner] + uniform_edges[x_inner + 1]) * 0.5, (uniform_edges[diverse_n_bin - 1] + uniform_edges[diverse_n_bin]) * 0.5)
                #     init = jax.vmap(uniform_func)(arrange_array[:-1])[:, None]
                #     km = KMeans(n_clusters=n_bins, n_samples=x.shape[0], init=init, max_iter=10)
                #     km.fit(x)
                ### Though it seems to successfuly control the number of centers,
                ### the problem here is the KMENAS will ruturn invalid center as 0
                ### The idea now is to use predict to get the valid centers.
                ### But it seems to be not an efficient solution
                #     centers = jnp.sort(km._centers[:, 0])
                #     return jnp.r_[minval, (centers[1:] + centers[:-1]) * 0.5, maxval]
                # bin_edges = jax.vmap(bin_func, in_axes=(1, None, 0), out_axes=1)(X, KMEANS, diverse_n_bins)

            ### remove the small interval
            ### unqiue_count is used to record the number of unique bin edges for each feature
            ### which is used in transform function.
            if remove_bin == True and self.strategy in ("quantile", "kmeans"):
                bin_edges, unqiue_count = _remove_bin_func(bin_edges, remove_ref)
                self.unqiue_count = unqiue_count
            else:
                self.unqiue_count = diverse_n_bins + 1
        else:
            ### the following code is the non-vectorized version
            ### they can only works with diverse_n_bins of type numpy.ndarray
            diverse_n_bins = self.diverse_n_bins
            if self.strategy == "uniform":
                for index_n_bin in range(diverse_n_bins.shape[0]):
                    x = X[:, index_n_bin]
                    diverse_n_bin = diverse_n_bins[index_n_bin]
                    ### According to emulation, add a branch on diverse_n_bin
                    ### will reduce communication cost
                    if diverse_n_bin < n_bins:
                        maxval = jnp.max(x)
                        diverse_bin_edges = jnp.linspace(
                            jnp.min(x), maxval, diverse_n_bin + 1
                        )
                        if index_n_bin == 0:
                            bin_edges = jnp.concatenate(
                                [
                                    diverse_bin_edges,
                                    jnp.full((n_bins - diverse_n_bin), maxval),
                                ]
                            ).reshape(-1, 1)
                        else:
                            bin_edges = jnp.concatenate(
                                [
                                    bin_edges,
                                    jnp.concatenate(
                                        [
                                            diverse_bin_edges,
                                            jnp.full((n_bins - diverse_n_bin), maxval),
                                        ]
                                    ).reshape(-1, 1),
                                ],
                                axis=1,
                            )
                    else:
                        diverse_bin_edges = jnp.linspace(
                            jnp.min(x), jnp.max(x), diverse_n_bin + 1
                        ).reshape(-1, 1)
                        if index_n_bin == 0:
                            bin_edges = diverse_bin_edges
                        else:
                            bin_edges = jnp.concatenate(
                                [bin_edges, diverse_bin_edges], axis=1
                            )
            if self.strategy == "quantile":
                if sample_weight is None:
                    quantiles = jnp.linspace(0, 100, n_bins + 1)
                    for index_n_bin in range(diverse_n_bins.shape[0]):
                        x = X[:, index_n_bin]
                        diverse_n_bin = diverse_n_bins[index_n_bin]
                        maxval = jnp.max(x)
                        diverse_quantiles = jnp.linspace(0, 100, diverse_n_bin + 1)
                        diverse_bin_edges = jnp.percentile(x, diverse_quantiles)
                        if index_n_bin == 0:
                            bin_edges = jnp.concatenate(
                                [
                                    diverse_bin_edges,
                                    jnp.full((n_bins - diverse_n_bin), maxval),
                                ]
                            ).reshape(-1, 1)
                        else:
                            bin_edges = jnp.concatenate(
                                [
                                    bin_edges,
                                    jnp.concatenate(
                                        [
                                            diverse_bin_edges,
                                            jnp.full((n_bins - diverse_n_bin), maxval),
                                        ]
                                    ).reshape(-1, 1),
                                ],
                                axis=1,
                            )
                else:
                    for index_n_bin in range(diverse_n_bins.shape[0]):
                        x = X[:, index_n_bin]
                        diverse_n_bin = diverse_n_bins[index_n_bin]
                        maxval = jnp.max(x)
                        diverse_quantiles = jnp.linspace(0, 100, diverse_n_bin + 1)
                        diverse_bin_edges = jax.vmap(
                            _weighted_percentile, (None, 0, None)
                        )(x, diverse_quantiles, sample_weight)
                        if index_n_bin == 0:
                            bin_edges = jnp.concatenate(
                                [
                                    diverse_bin_edges,
                                    jnp.full((n_bins - diverse_n_bin), maxval),
                                ]
                            ).reshape(-1, 1)
                        else:
                            bin_edges = jnp.concatenate(
                                [
                                    bin_edges,
                                    jnp.concatenate(
                                        [
                                            diverse_bin_edges,
                                            jnp.full((n_bins - diverse_n_bin), maxval),
                                        ]
                                    ).reshape(-1, 1),
                                ],
                                axis=1,
                            )
            if self.strategy == "kmeans":
                from ..cluster.kmeans import KMEANS

                for index_n_bin in range(diverse_n_bins.shape[0]):
                    x = X[:, index_n_bin]
                    diverse_n_bin = diverse_n_bins[index_n_bin]

                    x = x[:, None]
                    minval = jnp.min(x)
                    maxval = jnp.max(x)
                    diverse_bin_edges = _kmeans_bin_func(
                        x, diverse_n_bin, minval, maxval, KMEANS
                    )
                    if index_n_bin == 0:
                        bin_edges = jnp.concatenate(
                            [
                                diverse_bin_edges,
                                jnp.full((n_bins - diverse_n_bin), maxval),
                            ]
                        ).reshape(-1, 1)
                    else:
                        bin_edges = jnp.concatenate(
                            [
                                bin_edges,
                                jnp.concatenate(
                                    [
                                        diverse_bin_edges,
                                        jnp.full((n_bins - diverse_n_bin), maxval),
                                    ]
                                ).reshape(-1, 1),
                            ],
                            axis=1,
                        )

            ### remove the small interval
            ### unqiue_count is used to record the number of unique bin edges for each feature
            ### which is used in transform function.
            if remove_bin == True and self.strategy in ("quantile", "kmeans"):
                bin_edges, unqiue_count = _remove_bin_func(bin_edges, remove_ref)
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

        if self.diverse_n_bins is not None or (
            self.remove_bin == True and self.strategy in ("quantile", "kmeans")
        ):
            unqiue_count = self.unqiue_count

            def compute_row(bin, x, c):
                def compute_element(x):
                    encoding = x >= bin[1:-1]
                    return jnp.clip(jnp.sum(encoding), 0, c - 2)

                return jax.vmap(compute_element)(x)

            compute_rows_vmap = jax.vmap(compute_row, in_axes=(1, 1, 0), out_axes=1)(
                bin_edges, X, unqiue_count
            )
            return compute_rows_vmap
        else:

            def compute_row(bin, x):
                def compute_element(x):
                    encoding = x >= bin[1:-1]
                    return jnp.sum(encoding)

                return jax.vmap(compute_element)(x)

            compute_rows_vmap = jax.vmap(compute_row, in_axes=(1, 1), out_axes=1)(
                bin_edges, X
            )
            return compute_rows_vmap

    def fit_transform(
        self,
        X,
        sample_weight=None,
        vectorize=True,
        *,
        remove_bin=False,
        remove_ref=1e-3,
    ):
        """fit and transform with same input data.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Data to be discretized.

        sample_weight : {array-like} of shape (n_samples,)
            Contains weight values to be associated with each sample.
            Only possible when `strategy` is set to `"quantile"`.

        vectorize : bool, default=True
            Set to False to disable vectorization when diverse_n_bins is not
            None. The vectorized version supports computation with encrypted
            diverse_n_bins, in comparison, the non-vectorized version can only
            compute with diverse_n_bins of type numpy.ndarray (note that array
            created from jax.numpy will also create problems).
            There is currently no support for a vectorized form of kmeans.
            In addition, the vectorized version shows different performance and
            precision error with the non-vectorized version.
            Generally speaking, the vectorized version introduces more
            computation to enable more parallelization, which means more data
            sent and fewer send actions. However, when there is a great value
            gap between the n_bins and elements in diverse_n_bins, the
            vectorized version will introduce too much redundant computation
            that makes both data sent and send actions exceed the cost of the
            non-vectorized version. A special case is that vectorized form of
            quantile with sample_weight always shows more data sent and send
            actions than non-vectorized version.
            The vectorized version shows slightly less precision error than
            the non-vectorized version.
            Users can choose the version according to their specific requirements.

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
            X,
            sample_weight=sample_weight,
            vectorize=vectorize,
            remove_bin=remove_bin,
            remove_ref=remove_ref,
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
