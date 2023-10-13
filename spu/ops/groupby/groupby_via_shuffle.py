# Copyright 2023 Ant Group Co., Ltd.
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

from spu.ops.groupby.aggregation import (
    groupby_max,
    groupby_mean,
    groupby_min,
    groupby_sum,
    groupby_var,
)
from spu.ops.groupby.shuffle import shuffle_matrix


def groupby_sum_via_shuffle(
    cols, seg_end_marks, segment_ids, secret_random_order: jnp.ndarray
):
    """Perform groupby sum operation and shuffle the results.
    The inputs are groupby operation's output.
    """
    group_agg_matrix = groupby_sum(cols, seg_end_marks)
    return shuffle_matrix(
        group_agg_matrix, seg_end_marks, segment_ids, secret_random_order
    )


def groupby_max_via_shuffle(
    cols, seg_end_marks, segment_ids, secret_random_order: jnp.ndarray
):
    """Perform groupby max operation and shuffle the results.
    The inputs are groupby operation's output.
    """
    group_agg_matrix = groupby_max(cols, seg_end_marks)
    return shuffle_matrix(
        group_agg_matrix, seg_end_marks, segment_ids, secret_random_order
    )


def groupby_min_via_shuffle(
    cols, seg_end_marks, segment_ids, secret_random_order: jnp.ndarray
):
    """Perform groupby min operation and shuffle the results.
    The inputs are groupby operation's output.
    """
    group_agg_matrix = groupby_min(cols, seg_end_marks)
    return shuffle_matrix(
        group_agg_matrix, seg_end_marks, segment_ids, secret_random_order
    )


def groupby_mean_via_shuffle(
    cols, seg_end_marks, segment_ids, secret_random_order: jnp.ndarray
):
    """Perform groupby mean operation and shuffle the results.
    The inputs are groupby operation's output.
    """
    group_mean_matrix = groupby_mean(cols, seg_end_marks)

    return shuffle_matrix(
        group_mean_matrix, seg_end_marks, segment_ids, secret_random_order
    )


def groupby_var_via_shuffle(
    cols, seg_end_marks, segment_ids, secret_random_order: jnp.ndarray
):
    """Perform groupby var operation and shuffle the results.
    The inputs are groupby operation's output.

    recall that var(X) = (x_i - x_mean)^2 / (N - 1)
    """

    group_var_matrix = groupby_var(cols, seg_end_marks)
    return shuffle_matrix(
        group_var_matrix, seg_end_marks, segment_ids, secret_random_order
    )
