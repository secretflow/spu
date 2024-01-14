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

from posixpath import normcase

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
    n_samples = y.shape[0]
    indices = jnp.searchsorted(classes, y)
    result = jax.nn.one_hot(indices, n_classes, dtype=jnp.int_)
    # eq_func = lambda x: jnp.where(classes == x, 1, 0)
    # result = jax.vmap(eq_func)(y)

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

    Secondly, use transform() to convert the value to a one-hot label for classes.
    The input array needs to be 1d. In sklearn, the input array is automatically transformed into 1d,
    so more dimension seems to be meaningless. To avoid redundant operations used in MPC implementation,
    the automated transformation is canceled. Users can directly use the transformation method like jax.ravel to
    transform the input array into 1d then use LabelBinarizer to do further transformation.

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

    def fit(self, y, n_classes):
        """Fit label binarizer.

        Parameters
        ----------
        y : {array-like}, shape (n_samples,)
            Input data.

        n_classes : int
            Number of classes. SPU cannot support dynamic shape,
            so this parameter needs to be designated.

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
        # The output of jax needs to be tensor with known size.
        self.classes_ = jnp.unique(y, size=n_classes)
        self.n_classes_ = n_classes
        return self

    def fit_transform(self, y, n_classes):
        """Fit label binarizer/transform multi-class labels to binary labels.

        Parameters
        ----------
        y : {array-like}, shape (n_samples,)
            Input data.

        n_classes : int
            Number of classes. SPU cannot support dynamic shape,
            so this parameter needs to be designated.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems.
        """
        return self.fit(y, n_classes).transform(y)

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
