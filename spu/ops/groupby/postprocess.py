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
import numpy as np


# cleartext function
def groupby_agg_postprocess(
    segment_ids, seg_end_marks, group_agg_matrix, group_num: int
):
    assert (
        isinstance(group_num, int) and group_num > 0
    ), f"group num must be a positive integer. got {group_num}, {type(group_num)}"
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


#  function operating on cleartext, used to postprocess opened results.
def view_key_postprocessing(keys, group_num: int):
    """We want to view the key in order."""
    keys = np.unique(np.vstack(keys).T, axis=0)
    if keys.shape[0] > group_num:
        keys = keys[1:, :]
    return keys
