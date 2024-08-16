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

from typing import Tuple

import jax
import jax.numpy as jnp

from sml.preprocessing.preprocessing import label_binarize
from spu.ops.groupby import groupby, groupby_sum

from .auc import binary_clf_curve, binary_roc_auc


def roc_auc_score(y_true, y_pred):
    sorted_arr = create_sorted_label_score_pair(y_true, y_pred)
    return binary_roc_auc(sorted_arr)


def create_sorted_label_score_pair(y_true: jnp.array, y_score: jnp.array):
    """produce an n * 2 shaped array with the second column as the sorted scores, in decreasing order"""
    y_true = y_true.flatten()
    y_score = y_score.flatten()
    sorted_columns = jax.lax.sort([-y_score, y_true], num_keys=1)
    return jnp.hstack(
        [sorted_columns[1].reshape(-1, 1), -sorted_columns[0].reshape(-1, 1)]
    )


def bin_counts(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, thresholds: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """First, digitize y_pred using thresholds.
    Second, perform groupby sum to compute number of positive/negative samples in each bin.
    Being in bin i means thresholds[i-1] <= x < thresholds[i].

    Return:
        bin_ids: array (n,) padded with 0 in redundant rows.
        pos_counts: array (n,) number of positive counts, padded with 0 in redundant rows.
        neg_counts: array (n,) number of negative counts, padded with 0 in redundant rows.
        effective_rows: array. (n,) 0 or 1 array with each 1 indicates a valid row.
    """
    bins = jnp.digitize(y_pred, thresholds)
    y_true_negate = 1 - y_true
    bin_sorted, bin_count_cols, _, effective_rows = groupby(
        [-bins], [y_true, y_true_negate]
    )
    bin_count_matrix = groupby_sum(bin_count_cols, effective_rows)
    return (
        -bin_sorted[0],
        bin_count_matrix[:, 0],
        bin_count_matrix[:, 1],
        effective_rows,
    )


def equal_obs(x: jnp.ndarray, n_bin: int) -> jnp.ndarray:
    """
    Equal Frequency Split Point Search in x with bin size = n_bins
    In each bin, there is equal number of points in them

    Args:
        x: array
        n_bin: int

    Returns:
        jnp.array with size n_bin+1
    """
    n_len = len(x)
    return jnp.interp(
        x=jnp.linspace(0, n_len, n_bin + 1),
        xp=jnp.arange(n_len),
        fp=jnp.sort(x),
        right='extrapolate',
    )


def equal_range(x: jnp.ndarray, n_bin: int) -> jnp.ndarray:
    """
    Equal Range Search Split Points in x with bin size = n_bins
    Returns:
        jnp.array with size n_bin+1
    """
    min_val = jnp.min(x)
    max_val = jnp.max(x)
    segment = (max_val - min_val) / n_bin
    result = jnp.array([min_val + i * segment for i in range(n_bin)])
    result = jnp.r_[result, max_val + 0.01]
    return result


def _f1_score(y_true, y_pred):
    """Calculate the F1 score."""
    tp = jnp.sum(y_true * y_pred)
    fp = jnp.sum(y_pred) - tp
    fn = jnp.sum(y_true) - tp
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
    return f1


def _precision_score(y_true, y_pred):
    """Calculate the Precision score."""
    tp = jnp.sum(y_true * y_pred)
    fp = jnp.sum(y_pred) - tp
    precision = tp / (tp + fp + 1e-10)
    return precision


def _recall_score(y_true, y_pred):
    """Calculate the Recall score."""
    tp = jnp.sum(y_true * y_pred)
    fn = jnp.sum(y_true) - tp
    recall = tp / (tp + fn + 1e-10)
    return recall


def accuracy_score(y_true, y_pred):
    """Calculate the Accuracy score."""
    correct = jnp.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy


def transform_binary(y_true, y_pred, label):
    y_true_transform = jnp.where(y_true == label, 1, 0)
    y_pred_transform = jnp.where(y_pred != label, 0, 1)
    return y_true_transform, y_pred_transform


def f1_score(
    y_true, y_pred, average='binary', labels=None, pos_label=1, transform=True
):
    f1_result = fun_score(
        _f1_score, y_true, y_pred, average, labels, pos_label, transform
    )
    return f1_result


def precision_score(
    y_true, y_pred, average='binary', labels=None, pos_label=1, transform=True
):
    f1_result = fun_score(
        _precision_score, y_true, y_pred, average, labels, pos_label, transform
    )
    return f1_result


def recall_score(
    y_true, y_pred, average='binary', labels=None, pos_label=1, transform=True
):
    f1_result = fun_score(
        _recall_score, y_true, y_pred, average, labels, pos_label, transform
    )
    return f1_result


def fun_score(
    fun, y_true, y_pred, average='binary', labels=None, pos_label=1, transform=True
):
    """
    Compute precision, recall, f1.

    Args:
    fun : function, support '_precision_score' / '_recall_score' / '_f1_score'.

    y_true : 1d array-like, ground truth (correct) target values.

    y_pred : 1d array-like, estimated targets as returned by a classifier.

    average : {'binary'} or None, default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned.

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary

    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``.

    pos_label : int, float, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;

    transform : bool, default=True
        Binary classification only. If True, then the transformation of label to 0/1 will be done explicitly. Else, you can do it beforehand which decrease the costs of this function.

    Returns:
    -------
    precision : float, shape = [n_unique_labels] for multi-classification
        Precision score.

    recall : float, shape = [n_unique_labels] for multi-classification
        Recall score.

    f1 : float, shape = [n_unique_labels] for multi-classification
        F1 score.
    """

    if average is None:
        assert labels is not None, f"labels cannot be None"
        fun_result = []
        for i in labels:
            y_true_binary, y_pred_binary = transform_binary(y_true, y_pred, i)
            fun_result.append(fun(y_true_binary, y_pred_binary))
    elif average == 'binary':
        if transform:
            y_true_binary, y_pred_binary = transform_binary(y_true, y_pred, pos_label)
        else:
            y_true_binary, y_pred_binary = y_true, y_pred
        fun_result = fun(y_true_binary, y_pred_binary)
    else:
        raise ValueError("average should be None or 'binary'")
    return fun_result


def precision_recall_curve(
    y_true: jnp.ndarray, y_score: jnp.ndarray, pos_label=1, score_eps=1e-3
):
    """Compute precision-recall pairs for different probability thresholds.

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    y_true : 1d array-like of shape (n,). True binary labels.

    y_score : 1d array-like of shape (n,). Target scores, non-negative.

    pos_label : int, default=1. The label of the positive class.

    score_eps : float, default=1e-3. The lower bound for y_score.

    Returns
    -------
    precisions : ndarray of shape (n + 1,).
        Precision values where element i is the precision s.t.
        score >= thresholds[i] and the last element is 1.

    recalls : ndarray of shape (n + 1,).
        Increasing recall values where element i is the recall s.t.
        score >= thresholds[i] and the last element is 0.

    thresholds : ndarray of shape (n,).
        Decreasing thresholds used to compute precision and recall.
        Results might include trailing zeros.
    """

    # normalize the input
    y_true = jnp.where(y_true == pos_label, 1, 0)
    y_score = jnp.where(y_score < score_eps, score_eps, y_score) # to avoid wrong mask extraction

    # compute TP and FP
    pairs = jnp.stack([y_true, y_score], axis=1)
    sorted_pairs = pairs[jnp.argsort(pairs[:, 1], descending=True, stable=True)]
    fp, tp, thresholds = binary_clf_curve(sorted_pairs)

    # compute precision and recalls
    mask = jnp.where(thresholds > 0, 1, 0)  # tied value entries have mask=0
    last = jnp.max(
        jnp.where(mask, size=len(mask), fill_value=-1)[0]
    )  # equivalent of jnp.argwhere(mask)[-1], last index before tied value section
    precisions = jnp.where(mask, tp / (tp + fp + 1e-5), 0)
    recalls = jnp.where(tp[last] == 0, jnp.ones_like(tp), tp / tp[last])

    return (
        jnp.hstack((1, precisions)),
        jnp.hstack((0, recalls)),
        thresholds,
    )


def average_precision_score(
    y_true: jnp.ndarray,
    y_score: jnp.ndarray,
    classes=(0, 1),
    average="macro",
    pos_label=1,
    score_eps=1e-3,
):
    """Compute average precision (AP) from prediction scores.

    .. math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n

    Parameters
    -------
    y_true : array-like of shape (n_samples,)
             True labels.

    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
              Estimated target scores as returned by a classifier, non-negative.

    classes : 1d array-like, shape (n_classes,), default=(0,1) as for binary classification
              Uniquely holds the label for each class.
              SPU cannot support dynamic shape, so this parameter needs to be designated.

    average : {'macro', 'micro', None}, default='macro'
        This parameter is required for multiclass/multilabel targets and
        will be ignored when y_true is binary.

        'macro':
            Calculate metrics for each label, and find their unweighted mean.
        'micro':
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        None:
            Scores for each class are returned.

    pos_label : int, default=1
        The label of the positive class. Only applied to binary y_true.

    score_eps : float, default=1e-3. The lower bound for y_score.

    Returns
    -------
    average_precision : float
        Average precision score.
    """

    assert average in (
        'macro',
        'micro',
        None,
    ), 'average must be either "macro", "micro" or None'

    def binary_average_precision(y_true, y_score, pos_label=1):
        """Compute the average precision for binary classification."""
        precisions, recalls, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, score_eps=score_eps
        )

        return jnp.sum(jnp.diff(recalls) * precisions[1:])

    n_classes = len(classes)
    if n_classes <= 2:
        # binary classification
        # given y_true all the same is a special case considered as binary classification
        return binary_average_precision(y_true, y_score, pos_label=pos_label)
    else:
        # multi-class classification
        # binarize labels using one-vs-all scheme into multilabel-indicator
        y_true = label_binarize(y_true, classes=classes, n_classes=n_classes)

        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        elif average == "macro":
            pass

        # extend the classes dimension if needed
        if y_true.ndim == 1:
            y_true = y_true[:, jnp.newaxis]
        if y_score.ndim == 1:
            y_score = y_score[:, jnp.newaxis]

        # compute score for each class
        n_classes = y_score.shape[1]
        score = jnp.zeros((n_classes,))
        for c in range(n_classes):
            binary_ap = binary_average_precision(
                y_true[:, c].ravel(), y_score[:, c].ravel(), pos_label=pos_label
            )
            score = score.at[c].set(binary_ap)

        # average the scores
        return jnp.average(score) if average else score
