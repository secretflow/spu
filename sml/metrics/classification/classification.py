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


# TODO: more evaluation tools


def compute_f1_score(
    true_positive: jnp.ndarray, false_positive: jnp.ndarray, false_negative: jnp.ndarray
):
    """Calculate the F1 score."""
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2 * precision * recall / (precision + recall)
