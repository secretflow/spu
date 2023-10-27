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

from typing import List

import jax
import jax.numpy as jnp

from spu.ops.groupby.utils import batch_product


def shuffle_matrix(
    group_agg_matrix,
    seg_end_marks,
    segment_ids,
    secret_random_order: jnp.ndarray,
):
    """
    Shuffle the groupby matrix results for security
    """
    segment_ids_masked = seg_end_marks * segment_ids
    shuffled_cols = jax.lax.sort(
        [secret_random_order]
        + [segment_ids_masked]
        + [seg_end_marks]
        + [group_agg_matrix[:, i] for i in range(group_agg_matrix.shape[1])],
        num_keys=1,
    )
    return [
        shuffled_cols[1],
        shuffled_cols[2],
        jnp.vstack(shuffled_cols[3:]).T,
    ]


def shuffle_cols(
    cols_sorted: List[jnp.ndarray],
    seg_end_marks: jnp.ndarray,
    secret_random_order: jnp.ndarray,
):
    """Shuffle the cols sorted based on the secret random order.
    Often used to shuffle the key cols before revealing.
    We want to view the key without leaking the number of elements in each key.
    So we shuffle before revealing the key.
    """
    assert len(cols_sorted) > 0, "number of keys must be non-empty"
    keys = batch_product(cols_sorted, seg_end_marks)
    assert (
        secret_random_order.shape == cols_sorted[0].shape
    ), "the secret_random_order should be the same shape as each of the key columns."
    cols_shuffled = jax.lax.sort([secret_random_order] + keys, num_keys=1)[1:]
    return cols_shuffled
