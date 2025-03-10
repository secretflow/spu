# Copyright 2024 Ant Group Co., Ltd.
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


def set_value(x, index, value, n):
    """Change the value at the specified index of array x to a given value, where the index is secretly shared.

    Args:
        x: The input array to be modified.
        index: The index at which the value should be set (secretly shared).
        value: The new value to set at the specified index.
        n: The length of the array x.

    Returns:
        The modified array with the value updated at the specified index.
    """
    perm = jnp.arange(n)
    perm_2 = jnp.ones(n) * index

    set_x = jnp.where(perm == perm_2, value, x)

    return set_x


def get_value_1(x, index, n):
    """Retrieve the value at the specified index of array x, where the index is secretly shared.

    Args:
        x: The input array from which to retrieve the value.
        index: The index to retrieve the value from (secretly shared).
        n: The length of the array x.

    Returns:
        The value at the specified index.
    """

    perm = jnp.arange(n)
    perm_2 = jnp.ones(n) * index
    flag = jnp.equal(perm, perm_2)
    return jnp.sum(flag * x)


def get_value_2(x, index_1, index_2, n):
    """Retrieve the value at the specified 2D index of array x, where index_1 is secretly shared and index_2 is plaintext.

    Args:
        x: The input 2D array from which to retrieve the value.
        index_1: The row index (secretly shared).
        index_2: The column index (plaintext).
        n: The size of the array x (assuming it is square).

    Returns:
        The value at the specified 2D index.
    """

    # Initialize row index
    perm_1 = jnp.arange(n)[:, None]
    perm_1 = jnp.tile(perm_1, (1, index_2 + 1))

    perm_2_row = jnp.ones((n, index_2 + 1)) * index_1

    # Match rows
    flag_row = jnp.equal(perm_1, perm_2_row)

    # Extract column values directly using plaintext index_2
    flag = flag_row[:, index_2]
    # Return the value at the matching index
    return jnp.sum(flag * x[:, index_2])


def mpc_dijkstra(adj_matrix, num_samples, start, dist_inf):
    """Use Dijkstra's algorithm to compute the shortest paths from the starting point to all other points.

    Parameters
    ----------
    adj_matrix : ndarray
        The adjacency matrix used to compute the shortest paths.

    num_samples : int
        The size of the adjacency matrix (number of nodes).

    start : int
        The starting point for which to compute the shortest paths.

    dist_inf : ndarray
        The initialized shortest path array, usually set to infinity (inf).

    Returns
    -------
    distances : ndarray
        The shortest paths from the starting point to all other points.
    """

    # Initialize with Inf value
    sinf = dist_inf[0]
    distances = dist_inf

    # Calculate the shortest path from the starting point to other points using Dijkstra's algorithm
    distances = distances.at[start].set(0)
    # visited = [False] * num_samples
    visited = jnp.zeros(num_samples, dtype=bool)  # Initialize an array to False
    visited = jnp.array(visited)

    for i in range(num_samples):
        # Find the nearest node that is not currently visited

        min_distance = sinf
        min_index = -1
        for v in range(num_samples):
            flag = (visited[v] == 0) * (distances[v] < min_distance)
            min_distance = min_distance + flag * (distances[v] - min_distance)
            min_index = min_index + flag * (v - min_index)

        visited = set_value(visited, min_index, True, num_samples)

        # Update the distance between adjacent nodes
        temp_dis = get_value_1(distances, min_index, num_samples)

        for v in range(num_samples):
            temp_adj = get_value_2(adj_matrix, min_index, v, num_samples)
            dist_new = temp_dis + temp_adj
            distances = distances.at[v].set(
                distances[v]
                + (temp_adj != 0)
                * (visited[v] == 0)
                * (dist_new < distances[v])
                * (dist_new - distances[v])
            )
    return distances
