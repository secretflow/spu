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

from typing import Tuple, Union

import jax
import jax.numpy as jnp

from spu.ops.groupby import groupby_sorted


def binary_clf_curve(sorted_pairs: jnp.ndarray, return_seg_end_marks=False) -> Union[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Calculate true and false positives per binary classification
    threshold (can be used for roc curve or precision/recall curve).
    Results may include trailing zeros.
    Args:
        sorted_pairs: jnp.ndarray
            y_true y_score pairs sorted by y_score in decreasing order
        return_seg_end_marks: bool
            If true, the seg_end_marks array will be returned at the end
    Returns:
        fps: 1d ndarray
            False positives counts, index i records the number
            of negative samples that got assigned a
            score >= thresholds[i].
            The total number of negative samples is equal to
            fps[-1] (thus true negatives are given by fps[-1] - fps)
        tps: 1d ndarray
            True positives counts, index i records the number
            of positive samples that got assigned a
            score >= thresholds[i].
            The total number of positive samples is equal to
            tps[-1] (thus false negatives are given by tps[-1] - tps)
        thresholds : 1d ndarray
            predicted score sorted in decreasing order
        seg_end_marks: 1d ndarray
            marking the end of segment in result arrays
    References:
        Github: scikit-learn _binary_clf_curve.
    """
    # y_score typically consists of tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve

    tps = jnp.cumsum(sorted_pairs[:, 0])
    fps = jnp.arange(1, sorted_pairs.shape[0] + 1) - tps
    thresholds = sorted_pairs[:, 1]
    _, _, _, seg_end_marks = groupby_sorted([-thresholds], [-thresholds])
    tps = seg_end_marks * tps
    fps = seg_end_marks * fps
    thresholds = seg_end_marks * thresholds
    thresholds, fps, tps = jax.lax.sort([-thresholds] + [fps, tps], num_keys=1)

    if return_seg_end_marks:
        return fps, tps, -thresholds, seg_end_marks
    return fps, tps, -thresholds


def roc_curve(sorted_pairs: jnp.array) -> Tuple[jnp.array, jnp.array, jnp.array]:
    """Compute Receiver operating characteristic (ROC).

    Compared to sklearn implementation, this implementation eliminates most conditionals and ill-conditionals checking.
    Results may include trailng zeros.

    Args:
        sorted_pairs: jnp.array
            y_true y_score pairs sorted by y_score in decreasing order
    Returns:
        fpr: ndarray of shape (>2,)
            Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= `thresholds[i]`.
        tpr: ndarray of shape (>2,)
            Increasing true positive rates such that element `i` is the true
            positive rate of predictions with score >= `thresholds[i]`.
        thresholds: ndarray of shape = (n_thresholds,)
            Decreasing thresholds on the decision function used to compute
            fpr and tpr. `thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`.
    References:
        Github: scikit-learn roc_curve.
    """
    fps, tps, thresholds = binary_clf_curve(sorted_pairs)
    tps = jnp.r_[0, tps]
    fps = jnp.r_[0, fps]
    thresholds = jnp.r_[thresholds[0] + 1, thresholds]
    fpr = fps / jnp.max(fps)
    tpr = tps / jnp.max(tps)
    return fpr, tpr, thresholds


def auc(x, y):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.
    X must be monotonic, no checking inside function.

    Args:
        x: ndarray of shape (n,)
            monotonic X coordinates
        y: ndarray of shape, (n,)
            Y coordinates
    Returns:
        auc: float
            Area Under the Curve
    """
    x, y = jax.lax.sort([x, y], num_keys=1)
    area = jnp.abs(jax.scipy.integrate.trapezoid(y, x))
    return area


def binary_roc_auc(sorted_pairs: jnp.array) -> float:
    """
    Compute Area Under the Curve (AUC) for ROC from labels and prediction scores in sorted_pairs.

    Compared to sklearn implementation, this implementation is watered down with less options and
    eliminates most conditionals and ill-conditionals checking.

    Args:
        sorted_pairs: jnp.array
            y_true y_score pairs sorted by y_score in decreasing order,
            and it has shape n_samples * 2.
    Returns:
        roc_auc: float
    References:
        Github: scikit-learn _binary_roc_auc_score.
    """
    fpr, tpr, _ = roc_curve(sorted_pairs)
    return auc(fpr, tpr)
