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

import jax.numpy as jnp


def cols_to_matrix(cols):
    return jnp.vstack(cols).T


def matrix_to_cols(matrix):
    return [matrix[:, i] for i in range(matrix.shape[1])]


def batch_product(list_of_cols, multiplier_col):
    """apply multiplication of multiplier col to each column of list_of_cols"""
    return list(
        map(
            lambda x: x * multiplier_col,
            list_of_cols,
        )
    )


def rotate_cols(key_columns_sorted) -> List[jnp.ndarray]:
    return list(map(lambda x: jnp.roll(x, -1), key_columns_sorted))
