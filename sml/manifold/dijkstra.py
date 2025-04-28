# Copyright 2025 Ant Group Co., Ltd.
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
import jax.numpy as jnp
import numpy as np


def set_value(x, index, value):
    """Change the value at the specified index of array x to a given value, where the index is secretly shared.

    Args:
        x: The input array to be modified.
        index: The index at which the value should be set (secretly shared).
        value: The new value to set at the specified index.

    Returns:
        The modified array with the value updated at the specified index.
    """
    n = x.shape[0]
    perm = jnp.arange(n)
    perm_2 = jnp.ones(n) * index

    set_x = jnp.where(perm == perm_2, value, x)

    return set_x


def get_value_1(x, index):
    """Retrieve the value at the specified index of array x, where the index is secretly shared.

    Args:
        x: The input array from which to retrieve the value.
        index: The index to retrieve the value from (secretly shared).

    Returns:
        The value at the specified index.
    """

    n = x.shape[0]
    perm = jnp.arange(n)
    perm_2 = jnp.ones(n) * index
    flag = jnp.equal(perm, perm_2)
    return jnp.sum(flag * x)


def mpc_dijkstra(adj_matrix, start):
    """Use Dijkstra's algorithm to compute the shortest paths from the starting point to all other points.

    Parameters
    ----------
    adj_matrix : ndarray
        The adjacency matrix used to compute the shortest paths.

    start : int
        The starting point for which to compute the shortest paths.

    Returns
    -------
    distances : ndarray
        The shortest paths from the starting point to all other points.
    """

    num_samples = adj_matrix.shape[0]

    # Initialize with Inf value
    sinf = np.inf
    distances = jnp.full(num_samples, np.inf)

    # Calculate the shortest path from the starting point to other points using Dijkstra's algorithm
    distances = distances.at[start].set(0)
    visited = jnp.zeros(num_samples, dtype=bool)  # Initialize an array to False

    for i in range(num_samples):
        # Find the nearest node that is not currently visited

        min_distance = sinf
        min_index = -1
        for v in range(num_samples):
            flag = (visited[v] == 0) * (distances[v] < min_distance)
            min_distance = min_distance + flag * (distances[v] - min_distance)
            min_index = min_index + flag * (v - min_index)

        visited = set_value(visited, min_index, True)

        # Update the distance between adjacent nodes
        temp_dis = get_value_1(distances, min_index)

        for v in range(num_samples):
            temp_adj = get_value_1(adj_matrix[:, v], min_index)
            dist_new = temp_dis + temp_adj
            distances = distances.at[v].set(
                distances[v]
                + (temp_adj != 0)
                * (visited[v] == 0)
                * (dist_new < distances[v])
                * (dist_new - distances[v])
            )
    return distances
