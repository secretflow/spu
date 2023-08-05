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

import functools
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Conceptually we want to do the following
# However, due to limitations of design, we cannot do this
# we will use other groupby accumulators to do groupby some
# and treat (target_columns_sorted: List[jnp.ndarray], segment_ids: List[jnp.ndarray]) as the groupby object.
# class GroupByObject:
#     def __init__(
#         self, target_columns_sorted: List[jnp.ndarray], segment_ids: List[jnp.ndarray]
#     ):
#         self.target_columns_sorted = target_columns_sorted
#         self.segment_ids = segment_ids

#     def sum(self, group_num: int):
#         """
#         group num should be revealed and accessed from segment ids and taken as a static int.
#         """
#         segment_ids = self.segment_ids
#         x = self.target_columns_sorted
#         return jax.ops.segment_sum(x, segment_ids, num_segments=group_num)


def segment_aware_addition(row1, row2):
    return segment_aware_ops(row1, row2, jnp.add)


def segment_aware_max(row1, row2):
    return segment_aware_ops(row1, row2, jnp.maximum)


def segment_aware_ops(row1, row2, ops):
    cum_part = jnp.where((row2[:, 0] == 1).reshape(-1, 1), ops(row1, row2), row2)[:, 1:]
    lead_part = (row1[:, 0] * row2[:, 0]).reshape(-1, 1)
    return jnp.c_[lead_part, cum_part]


def groupby_agg_via_shuffle(
    cols,
    seg_end_marks,
    segment_ids,
    secret_random_order: jnp.ndarray,
    segment_aware_ops,
):
    """Groupby Aggregation with shuffled outputs.

    trick: output segment_end_marks and group aggregations in shuffled state,
    filter to get the group aggregations in cleartext.

    shuffle to protect number of group elements.

    The returns of this function are supposed to be ok to be opened.
    return:
        segment_ids_shuffled:
            shuffled segment ids
        shuffled_group_end_masks:
            shuffled group end masks
        shuffled_group_agg_matrix:
            shape = (n_samples, n_cols)
            group aggregations shuffled
            padded with zeros.

    """
    group_mask = jnp.ones(seg_end_marks.shape) - jnp.roll(seg_end_marks, 1)

    X = jnp.vstack([group_mask] + list(cols)).T

    X_prefix_sum = jax.lax.associative_scan(segment_aware_ops, X, axis=0)
    X_prefix_sum_masked = seg_end_marks.reshape(-1, 1) * X_prefix_sum
    segment_ids_masked = seg_end_marks * segment_ids
    shuffled_cols = jax.lax.sort(
        [secret_random_order]
        + [segment_ids_masked]
        + [seg_end_marks]
        + [X_prefix_sum_masked[:, i] for i in range(1, X_prefix_sum_masked.shape[1])],
        num_keys=1,
    )
    return [
        shuffled_cols[1],
        shuffled_cols[2],
        jnp.vstack(shuffled_cols[3:]).T,
    ]


def groupby_sum_via_shuffle(
    cols, seg_end_marks, segment_ids, secret_random_order: jnp.ndarray
):
    return groupby_agg_via_shuffle(
        cols,
        seg_end_marks,
        segment_ids,
        secret_random_order,
        segment_aware_ops=segment_aware_addition,
    )


def groupby_max_via_shuffle(
    cols, seg_end_marks, segment_ids, secret_random_order: jnp.ndarray
):
    return groupby_agg_via_shuffle(
        cols,
        seg_end_marks,
        segment_ids,
        secret_random_order,
        segment_aware_ops=segment_aware_max,
    )


# cleartext function
def groupby_agg_postprocess(
    segment_ids, seg_end_marks, group_agg_matrix, group_num: int
):
    assert (
        isinstance(group_num, int) and group_num > 0
    ), f"group num must be a positve integer. got {group_num}, {type(group_num)}"
    if group_num > 1:
        filter_mask = seg_end_marks == 1
        segment_ids = segment_ids[filter_mask]
        group_agg_matrix = group_agg_matrix[filter_mask]
        sorted_results = jax.lax.sort(
            [segment_ids]
            + [group_agg_matrix[:, i] for i in range(group_agg_matrix.shape[1])],
            num_keys=1,
        )[1:]
        return jnp.vstack(sorted_results).T
    else:
        return group_agg_matrix[-1]


def batch_product(list_of_cols, multiplier_col):
    return list(
        map(
            lambda x: x * multiplier_col,
            list_of_cols,
        )
    )


def view_key(
    key_columns_sorted: List[jnp.ndarray],
    seg_end_marks: jnp.ndarray,
    secret_random_order: jnp.ndarray,
):
    """The secret_random_order must be secret to all parties
    trick: open a shuffled array and unique in cleartext
    """
    assert len(key_columns_sorted) > 0, "number of keys must be non-empty"
    keys = batch_product(key_columns_sorted, seg_end_marks)
    assert (
        secret_random_order.shape == key_columns_sorted[0].shape
    ), "the secret_random_order should be the same shape as each of the key columns."
    keys_shuffled = jax.lax.sort([secret_random_order] + keys, num_keys=1)[1:]
    return keys_shuffled


#  function operating on cleartext, used to postprocess opened results.
def view_key_postprocessing(keys, group_num: int):
    keys = np.unique(np.vstack(keys).T, axis=0)
    if keys.shape[0] > group_num:
        keys = keys[1:, :]
    return keys


def groupby(
    key_columns: List[jnp.ndarray],
    target_columns: List[jnp.ndarray],
) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
    """GroupBy
    Given a matrix X, it has multiple columns.
    We want to calculate some statistics of target columns grouped by some columns as keys.
    This operator completes the first step of GroupBy statistics: transfom the matrix x into a form,
    that is suitable for subsequent statistics.

    Parameters
    ----------

    key_columns : List[jnp.ndarray]
        List of columns that are used as keys, these should be arrays of the same shape.

    target_columns :  List[jnp.ndarray]
        List of columns that are used as keys, these should be arrays of the same shape as the shape in key columns.


    Returns
    -------
    key_columns_sorted : List[jnp.ndarray]
    target_columns_sorted : List[jnp.ndarray]
    segment_ids :  List[jnp.ndarray]
    """
    # parameter check.
    assert isinstance(key_columns, List)
    assert isinstance(target_columns, List)
    assert len(key_columns) > 0, "There should be at least one key_column."
    assert len(target_columns) > 0, "There should be at least one target_column."
    assert (
        len(set(map(lambda x: x.shape, key_columns + target_columns))) == 1
    ), f"Columns' shape should be consistant. {set(map(lambda x: x.shape, key_columns + target_columns))}"
    key_columns = key_columns
    target_columns = target_columns
    sorted_columns = jax.lax.sort(
        key_columns + target_columns, num_keys=len(key_columns)
    )
    key_columns_sorted = sorted_columns[: len(key_columns)]
    target_columns_sorted = sorted_columns[len(key_columns) :]
    key_columns_sorted_rolled = rotate_cols(key_columns_sorted)
    seg_end_marks = get_segment_marks(key_columns_sorted, key_columns_sorted_rolled)
    mark_accumulated = associative_scan(seg_end_marks)
    segment_ids = mark_accumulated - seg_end_marks
    return key_columns_sorted, target_columns_sorted, segment_ids, seg_end_marks


# the method is simple: open segment_ids and do count in cleartext
# in SPU all NaN values are encoded as 0, so count becomes trivial.
# cleartext function:
# further if a query includes count, the shuffle ops in groupby sum can be skipped
def groupby_count(opened_segment_ids):
    _, counts = jnp.unique(opened_segment_ids, return_counts=True)
    return counts


def rotate_cols(key_columns_sorted) -> List[jnp.ndarray]:
    return list(map(lambda x: jnp.roll(x, -1), key_columns_sorted))


def get_segment_marks(key_columns_sorted, key_columns_sorted_rolled):
    tuple_list = list(zip(key_columns_sorted, key_columns_sorted_rolled))
    equal = [a - b == 0 for (a, b) in tuple_list]
    c = ~functools.reduce(lambda x, y: x & y, equal)
    c = c.astype(int)
    result = jnp.r_[c[: c.size - 1], [1]]
    # try
    # result = c.at[c.size - 1].set(1)
    return result


def associative_scan(seg_end_marks):
    return jax.lax.associative_scan(jnp.add, seg_end_marks)
