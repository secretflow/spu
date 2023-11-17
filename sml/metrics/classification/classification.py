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
from auc import binary_roc_auc

from spu.ops.groupby import groupby, groupby_sum


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


def f1_score(y_true, y_pred, average='binary', labels=None, pos_label=1, transform=True):
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
