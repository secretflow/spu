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

class MinMaxScaler():
    def __init__(self, feature_range=(0, 1), *, clip=False):
        self.feature_range = feature_range
        self.clip = clip
    
    def _reset(self):
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, *, zero_variance=False, contain_nan=False):
        ### if do not handle zero variance, though transform the matrix that used in fit will work correctly. Transform new matrix and inverse transform will work incorrectly.
        self._reset()
        return self.partial_fit(X, zero_variance=zero_variance, contain_nan=contain_nan)
    
    def partial_fit(self, X, *, zero_variance=False, contain_nan=False):
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
            self.scale_ = (feature_range[1] - feature_range[0]) / jnp.where(data_range==0, 1, data_range)

        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self
    
    def transform(self, X):
        X *= self.scale_
        X += self.min_
        if self.clip:
            X = jnp.clip(X, self.feature_range[0], self.feature_range[1])
        return X

    def fit_transform(self, X, *, zero_variance=False, contain_nan=False):
        return self.fit(X, zero_variance=zero_variance, contain_nan=contain_nan).transform(X)

    def inverse_transform(self, X):
        X -= self.min_
        X /= self.scale_
        return X

class MaxAbsScaler():
    def __init__(self):
        pass

    def _reset(self):
        if hasattr(self, "scale_"):
            del self.scale_
            del self.n_samples_seen_
            del self.max_abs_
    
    def fit(self, X, zero_maxabs=False, contain_nan=False):
        ### if do not handle zero maxabs, though transform the matrix that used in fit and inverse transform will work correctly. Transform new matrix and inverse transform will work incorrectly.
        self._reset()
        return self.partial_fit(X, zero_maxabs=zero_maxabs, contain_nan=contain_nan)
    
    def partial_fit(self, X, *, zero_maxabs=False, contain_nan=False):
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
            self.scale_ = jnp.where(max_abs==0, 1, max_abs)
        return self
    
    def transform(self, X):
        return X / self.scale_
    
    def fit_transform(self, X, *, zero_maxabs=False, contain_nan=False):
        return self.fit(X, zero_maxabs=zero_maxabs, contain_nan=contain_nan).transform(X)

    def inverse_transform(self, X):
        return X * self.scale_
    
class KBinsDiscretizer():
    ### subsample is not triggered by default
    # n_bins : array-like of shape (n_features,)
    #     The number of bins to produce. All the value in n_bins should be int >= 2. There is no check here to avoid redundent operation, so user should pay attention here.
    def __init__(self, n_bins, same_n_bins=True, *, encode="onehot", strategy="quantile", dtype=None, subsample=None, random_state=None):
        self.n_bins = n_bins
        self.same_n_bins = same_n_bins
        self.encode = encode
        self.strategy = strategy
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state
    
    def fit(self, X, sample_weight=None):
        ### subsample is not triggered by default
        if self.dtype in (jnp.float64, jnp.float32):
            output_dtype = self.dtype
        else:  # self.dtype is None
            output_dtype = X.dtype
        n_samples, n_features = X.shape
        # X = X.astype(output_dtype)
        if sample_weight is not None and self.strategy == "uniform":
            raise ValueError(
                "`sample_weight` was provided but it cannot be "
                "used with strategy='uniform'. Got strategy="
                f"{self.strategy!r} instead."
            )
        subsample = self.subsample
        n_bins = self.n_bins
        if subsample is not None and n_samples > subsample:
            ### this part is not implemented
            pass

        if self.same_n_bins == True:
            bin_func = lambda x: jnp.linspace(jnp.min(x), jnp.max(x), n_bins + 1)
            bin_edges = jax.vmap(bin_func, in_axes=1, out_axes=1)(X)
        else:
            bin_edges = []
            for jj in range(n_features):
                column = X[:, jj]
                col_min, col_max = jnp.min(column), jnp.max(column)

                if self.strategy == "uniform":
                    bin_edges.append(jnp.linspace(col_min, col_max, n_bins[jj] + 1))
        # column_min = jnp.min(X, axis=0)
        # column_max = jnp.max(X, axis=0)
        # bin_edges = jnp.linspace(column_min, column_max, n_bins[:, None])

        # bin_func = lambda x: jnp.linspace(jnp.min(x), jnp.max(x), 4)
        # bin_edges = jax.vmap(bin_func, in_axes=1, out_axes=0)(X)

        # bin_func = lambda x: jnp.linspace(jnp.min(x), jnp.max(x), 4)
        # bin_edges = jax.vmap(bin_func, in_axes=1, out_axes=0)(X)


        # for jj in range(n_features):
        #     column = X[:, jj]
        #     col_min, col_max = jnp.min(column), jnp.max(column)

        #     ### some problem related to dynamic shape

        #     # if col_min == col_max:
        #     #     # warnings.warn(
        #     #     #     "Feature %d is constant and will be replaced with 0." % jj
        #     #     # )
        #     #     n_bins[jj] = 1
        #     #     bin_edges[jj] = jnp.array([-jnp.inf, jnp.inf])
        #     #     continue
        #     if self.strategy == "uniform":
        #         bin_edges[jj] = jnp.linspace(col_min, col_max, 3)
        #     elif self.strategy == "quantile":
        #         quantiles = jnp.linspace(0, 100, n_bins[jj] + 1)
        #         if sample_weight is None:
        #             bin_edges[jj] = jnp.asarray(jnp.percentile(column, quantiles))
        #         ### not implemented
                    
        #         # else:
        #         #     bin_edges[jj] = jnp.asarray(
        #         #         [
        #         #             _weighted_percentile(column, sample_weight, q)
        #         #             for q in quantiles
        #         #         ],
        #         #         dtype=np.float64,
        #         #     )
                    
        #     if self.strategy in ("quantile", "kmeans"):
        #         mask = jnp.ediff1d(bin_edges[jj], to_begin=jnp.inf) > 1e-8
        #         bin_edges[jj] = bin_edges[jj][mask]
        #         if len(bin_edges[jj]) - 1 != n_bins[jj]:
        #             # warnings.warn(
        #             #     "Bins whose width are too small (i.e., <= "
        #             #     "1e-8) in feature %d are removed. Consider "
        #             #     "decreasing the number of bins." % jj
        #             # )
        #             n_bins[jj] = len(bin_edges[jj]) - 1
        
        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins

        ### not implemented
        # if "onehot" in self.encode:
        #     self._encoder = OneHotEncoder(
        #         categories=[np.arange(i) for i in self.n_bins_],
        #         sparse_output=self.encode == "onehot",
        #         dtype=output_dtype,
        #     )
        #     # Fit the OneHotEncoder with toy datasets
        #     # so that it's ready for use after the KBinsDiscretizer is fitted
        #     self._encoder.fit(np.zeros((1, len(self.n_bins_))))

        return self

    def transform(self, X):
        # check input and attribute dtypes
        dtype = (jnp.float64, jnp.float32) if self.dtype is None else self.dtype
        # Xt = self._validate_data(X, copy=True, dtype=dtype, reset=False)

        bin_edges = self.bin_edges_

        # def compute_row(x, y):
        #     jnp.searchsorted(x[1:-1], y, side="right")
        compute_row = lambda x, y: jnp.searchsorted(x[1:-1], y, side="right")
        compute_rows_vmap = jax.vmap(compute_row, in_axes=(1, 1), out_axes=1)(bin_edges, X)
        return compute_rows_vmap
        # for jj in range(Xt.shape[1]):
        #     Xt[:, jj] = np.searchsorted(bin_edges[jj][1:-1], Xt[:, jj], side="right")

        # if self.encode == "ordinal":
        #     return Xt

        # dtype_init = None
        # if "onehot" in self.encode:
        #     dtype_init = self._encoder.dtype
        #     self._encoder.dtype = Xt.dtype
        # try:
        #     Xt_enc = self._encoder.transform(Xt)
        # finally:
        #     # revert the initial dtype to avoid modifying self.
        #     self._encoder.dtype = dtype_init
        # return Xt_enc

