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


import jax
import jax.numpy as jnp

from spu.ops.groupby.utils import cols_to_matrix, matrix_to_cols


def groupby_agg(
    cols,
    seg_end_marks,
    segment_aware_ops,
) -> jnp.ndarray:
    """Performs groupby aggregation operation.

    The returns of this function are NOT safe to open (count of group elements revealed)
    However, if user already asked about count statistics, then we can safely open it.

    return:
        group_agg_matrix:
            shape = (n_samples, n_cols)
            group aggregations
            padded with zeros.

    """
    group_mask = jnp.ones(seg_end_marks.shape) - jnp.roll(seg_end_marks, 1)

    X = jnp.vstack([group_mask] + list(cols)).T

    X_prefix_sum = jax.lax.associative_scan(segment_aware_ops, X, axis=0)
    X_prefix_sum_masked = seg_end_marks.reshape(-1, 1) * X_prefix_sum

    return X_prefix_sum_masked[:, 1:]


def groupby_transform(seg_end_marks, group_agg_matrix):
    """broadcast the result of groupby_agg in a group wise manner
    [[0,0,0,b1,0,0,0,b2],
     [0,0,0,c1, 0,0,0,c2]] -> [[b1, b1, b1, b1, b2,b2,b2,b2],[c1,c1,c1,c1,c2,c2,c2,c2]]
    """

    group_agg_matrix_offseted = group_agg_matrix
    # perform groupby sum
    group_mask = jnp.ones(seg_end_marks.shape) - seg_end_marks
    X = jnp.hstack([group_mask.reshape(-1, 1), group_agg_matrix_offseted])
    X_prefix_sum = jax.lax.associative_scan(
        segment_aware_addition, X, axis=0, reverse=True
    )
    # restore the offset
    return X_prefix_sum[:, 1:]


def groupby_sum(cols, seg_end_marks) -> jnp.ndarray:
    return groupby_agg(cols, seg_end_marks, segment_aware_addition)


def groupby_max(cols, seg_end_marks) -> jnp.ndarray:
    return groupby_agg(cols, seg_end_marks, segment_aware_max)


def groupby_min(cols, seg_end_marks) -> jnp.ndarray:
    return groupby_agg(cols, seg_end_marks, segment_aware_min)


def groupby_count(cols, seg_end_marks):
    """groupby count, it does not require cleartext data"""
    ones = jnp.ones(cols[0].shape)
    group_count_matrix = groupby_agg(
        [ones], seg_end_marks, segment_aware_ops=segment_aware_addition
    )
    return group_count_matrix


# the method is simple: open segment_ids and do count in cleartext
# It is supposed to be faster than the groupby_count which outputs ciphertext.
# in SPU all NaN values are encoded as 0, so count becomes trivial.
# cleartext function:
# further if a query includes count, the shuffle ops in groupby sum can be skipped
def groupby_count_cleartext(opened_segment_ids):
    """Count the number of elements in each group."""
    _, counts = jnp.unique(opened_segment_ids, return_counts=True)
    return counts


def groupby_mean(cols, seg_end_marks):
    assert len(cols) > 0, "at least one col is required"

    # note nan are zeros in SPU, count is the same for all columns
    group_count_matrix = groupby_count(cols, seg_end_marks)
    return grouby_mean_given_count(cols, seg_end_marks, group_count_matrix)


def grouby_mean_given_count(cols, seg_end_marks, group_count_matrix):
    group_sum_matrix = groupby_agg(
        cols,
        seg_end_marks,
        segment_aware_ops=segment_aware_addition,
    )
    return group_sum_matrix / group_count_matrix


def groupby_var(cols, seg_end_marks):
    """Perform groupby var operation and shuffle the results.
    The inputs are groupby operation's output.

    recall that var(X) = (x_i - x_mean)^2 / (N - 1)
    """

    group_count_matrix = groupby_count(cols, seg_end_marks)
    group_mean_matrix = grouby_mean_given_count(cols, seg_end_marks, group_count_matrix)
    mean_matrix = groupby_transform(seg_end_marks, group_mean_matrix)
    raw_matrix = cols_to_matrix(cols)
    residual_square_matrix = (raw_matrix - mean_matrix) ** 2
    group_rrs_matrix = groupby_sum(
        matrix_to_cols(residual_square_matrix), seg_end_marks
    )
    group_var_matrix = (
        seg_end_marks.reshape(-1, 1) * group_rrs_matrix / (group_count_matrix - 1)
    )
    return group_var_matrix


def segment_aware_addition(row1, row2):
    return segment_aware_ops(row1, row2, jnp.add)


def segment_aware_max(row1, row2):
    return segment_aware_ops(row1, row2, jnp.maximum)


def segment_aware_min(row1, row2):
    return segment_aware_ops(row1, row2, jnp.minimum)


def segment_aware_ops(row1, row2, ops):
    cum_part = jnp.where((row2[:, 0] == 1).reshape(-1, 1), ops(row1, row2), row2)[:, 1:]
    lead_part = (row1[:, 0] * row2[:, 0]).reshape(-1, 1)
    return jnp.c_[lead_part, cum_part]
