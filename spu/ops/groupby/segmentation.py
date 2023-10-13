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

from spu.ops.groupby.utils import rotate_cols


def groupby(
    key_columns: List[jnp.ndarray],
    target_columns: List[jnp.ndarray],
) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
    """GroupBy
    Given a matrix X, it has multiple columns.
    We want to calculate some statistics of target columns grouped by some columns as keys.
    This operator completes the first step of GroupBy statistics: transform the matrix x into a form,
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
    segment_ids :  jnp.ndarray
    seg_end_marks : jnp.ndarray
    """
    # parameter check.
    assert isinstance(key_columns, List)
    assert isinstance(target_columns, List)
    assert len(key_columns) > 0, "There should be at least one key_column."
    assert len(target_columns) > 0, "There should be at least one target_column."
    assert (
        len(set(map(lambda x: x.shape, key_columns + target_columns))) == 1
    ), f"Columns' shape should be consistent. {set(map(lambda x: x.shape, key_columns + target_columns))}"
    key_columns = key_columns
    target_columns = target_columns
    sorted_columns = jax.lax.sort(
        key_columns + target_columns, num_keys=len(key_columns)
    )
    key_columns_sorted = sorted_columns[: len(key_columns)]
    target_columns_sorted = sorted_columns[len(key_columns) :]
    return groupby_sorted(key_columns_sorted, target_columns_sorted)


def groupby_sorted(
    key_columns_sorted: List[jnp.ndarray],
    target_columns_sorted: List[jnp.ndarray],
) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
    """Groupby on sorted data."""
    key_columns_sorted_rolled = rotate_cols(key_columns_sorted)
    seg_end_marks = get_segment_marks(key_columns_sorted, key_columns_sorted_rolled)
    mark_accumulated = associative_scan(seg_end_marks)
    segment_ids = mark_accumulated - seg_end_marks
    return key_columns_sorted, target_columns_sorted, segment_ids, seg_end_marks


def get_segment_marks(key_columns_sorted, key_columns_sorted_rolled):
    tuple_list = list(zip(key_columns_sorted, key_columns_sorted_rolled))
    equal = [a - b == 0 for (a, b) in tuple_list]
    c = ~functools.reduce(lambda x, y: x & y, equal)
    c = c.astype(int)
    result = jnp.r_[c[: c.size - 1], [1]]
    return result


def associative_scan(seg_end_marks):
    return jax.lax.associative_scan(jnp.add, seg_end_marks)
