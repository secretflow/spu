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

import jax.numpy as jnp


def contingency_matrix(
    labels_true: jnp.ndarray,
    labels_pred: jnp.ndarray,
    n_classes: int,
    n_clusters: int,
    eps=None,
    dtype=jnp.int32,
):
    """
    Compute the contingency matrix between two clusterings.

    The contingency matrix is a matrix where the element at (i, j) represents
    the number of samples that are in true class i and predicted cluster j.

    Parameters
    ----------
    labels_true : jnp.ndarray, shape (n_samples,)
        True class labels.

    labels_pred : jnp.ndarray, shape (n_samples,)
        Predicted cluster labels.

    n_classes : int
        Number of distinct true classes.
        SPU cannot support dynamic shape, so this parameter needs to be designated.

    n_clusters : int
        Number of distinct predicted clusters.
        SPU cannot support dynamic shape, so this parameter needs to be designated.

    eps : optional, default=None
        Small value to add to each entry of the contingency matrix (for smoothing).

    dtype : jnp.dtype, default=jnp.int32
        Data type of the returned contingency matrix.

    Returns
    -------
    contingency : jnp.ndarray, shape (n_classes, n_clusters)
        Contingency matrix counting co-occurrences of true and predicted labels.
    """

    classes, class_idx = jnp.unique(labels_true, return_inverse=True, size=n_classes)
    clusters, cluster_idx = jnp.unique(
        labels_pred, return_inverse=True, size=n_clusters
    )

    # Flatten (class, cluster) index into single integer
    idx = class_idx * jnp.int32(n_clusters) + cluster_idx

    # bincount with explicit length to fix output shape at compile time
    counts = jnp.bincount(idx, length=n_classes * n_clusters).astype(dtype)
    contingency = counts.reshape((n_classes, n_clusters))

    if eps is not None:
        contingency = contingency + jnp.array(eps, dtype=dtype)

    return contingency


def pair_confusion_matrix(
    labels_true: jnp.ndarray, labels_pred: jnp.ndarray, n_classes: int, n_clusters: int
):
    """
    Compute a 2x2 pair confusion matrix from clustering labels.

    The pair confusion matrix counts pairs of samples in the following categories:
        - True Negative (TN): pairs not in same true class nor predicted cluster
        - False Positive (FP): pairs not in same true class but same predicted cluster
        - False Negative (FN): pairs in same true class but not in same predicted cluster
        - True Positive (TP): pairs in same true class and same predicted cluster

    Parameters
    ----------
    labels_true : jnp.ndarray, shape (n_samples,)
        True class labels.

    labels_pred : jnp.ndarray, shape (n_samples,)
        Predicted cluster labels.

    n_classes : int
        Number of true classes.

    n_clusters : int
        Number of predicted clusters.

    Returns
    -------
    C : jnp.ndarray, shape (2, 2)
        Pair confusion matrix [[TN, FP],
                              [FN, TP]]
    """

    n_samples = jnp.int32(labels_true.shape[0])

    contingency = contingency_matrix(
        labels_true, labels_pred, n_classes, n_clusters, dtype=jnp.int32
    )

    n_c = jnp.ravel(contingency.sum(axis=1))
    n_k = jnp.ravel(contingency.sum(axis=0))

    sum_squares = (contingency**2).sum()

    C = jnp.zeros((2, 2), dtype=jnp.int32)
    C = C.at[1, 1].set(sum_squares - n_samples)
    C = C.at[0, 1].set((contingency.dot(n_k)).sum() - sum_squares)
    C = C.at[1, 0].set((contingency.T.dot(n_c)).sum() - sum_squares)
    C = C.at[0, 0].set(n_samples**2 - C[0, 1] - C[1, 0] - sum_squares)

    return C


def rand_score(
    labels_true: jnp.ndarray, labels_pred: jnp.ndarray, n_classes: int, n_clusters: int
):
    """
    Compute the Rand index, a measure of similarity between two clusterings.

    The Rand index is the ratio of the number of agreeing pairs (same or different clusters)
    over the total number of pairs.

    Parameters
    ----------
    labels_true : jnp.ndarray, shape (n_samples,)
        True class labels.

    labels_pred : jnp.ndarray, shape (n_samples,)
        Predicted cluster labels.

    n_classes : int
        Number of true classes.

    n_clusters : int
        Number of predicted clusters.

    Returns
    -------
    rand_index : float
        Rand index score in [0, 1]. 1 means perfect agreement.
    """

    C = pair_confusion_matrix(labels_true, labels_pred, n_classes, n_clusters)
    numerator = C.diagonal().sum()
    denominator = C.sum()

    return jnp.where(
        (numerator == 0) | (denominator == 0), 1.0, numerator / denominator
    )


def adjusted_rand_score(
    labels_true: jnp.ndarray, labels_pred: jnp.ndarray, n_classes: int, n_clusters: int
):
    """
    Compute the Adjusted Rand Index (ARI) between two clusterings.

    ARI corrects the Rand index for chance, with value 0 expected for random labelings,
    and 1 for perfect match.

    Parameters
    ----------
    labels_true : jnp.ndarray, shape (n_samples,)
        True class labels.

    labels_pred : jnp.ndarray, shape (n_samples,)
        Predicted cluster labels.

    n_classes : int
        Number of distinct true classes.
        SPU cannot support dynamic shape, so this parameter needs to be designated.

    n_clusters : int
        Number of distinct predicted clusters.
        SPU cannot support dynamic shape, so this parameter needs to be designated.

    Returns
    -------
    adjusted_rand_index : float
        Adjusted Rand Index score in [-1, 1], higher is better.
    """

    (tn, fp), (fn, tp) = pair_confusion_matrix(
        labels_true, labels_pred, n_classes=n_classes, n_clusters=n_clusters
    )

    return jnp.where(
        jnp.logical_and(fn == 0, fp == 0),
        1.0,
        2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)),
    )
