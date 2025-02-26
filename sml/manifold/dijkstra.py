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
    # Change the value at the index of array x to value, where index is secretly shared
    # n: the length of array x

    perm = jnp.zeros(n, dtype=jnp.int16)
    perm_2 = jnp.zeros(n, dtype=jnp.int16)
    for i in range(n):
        perm = perm.at[i].set(i)
        perm_2 = perm_2.at[i].set(index)
    flag = jnp.equal(perm, perm_2)
    set_x = jnp.select([flag], [value], x)

    return set_x


def get_value_1(x, index, n):
    # Obtain the value at the x[index] index, where index is a secret shared value
    # n: the length of array x

    perm = jnp.zeros(n, dtype=jnp.int16)
    perm_2 = jnp.zeros(n, dtype=jnp.int16)
    for i in range(n):
        perm = perm.at[i].set(i)
        perm_2 = perm_2.at[i].set(index)
    flag = jnp.equal(perm, perm_2)
    return jnp.sum(flag * x)


def get_value_2(x, index_1, index_2, n):
    # Obtain the value at index x[index_1][index_2], where index_2 is plaintext and index_1 is secret shared
    # n: the length of array x

    # Initialize row index
    perm_1 = jnp.zeros((n, n), dtype=jnp.int16)
    perm_2_row = jnp.zeros((n, n), dtype=jnp.int16)

    for i in range(n):
        for j in range(n):
            perm_1 = perm_1.at[i, j].set(i)
            perm_2_row = perm_2_row.at[i, j].set(index_1)

    # Match rows
    flag_row = jnp.equal(perm_1, perm_2_row)

    # Extract column values directly using plaintext index_2
    flag = flag_row[:, index_2]

    # Return the value at the matching index
    return jnp.sum(flag * x[:, index_2])


def mpc_dijkstra(adj_matrix, num_samples, start, dist_inf):
    # adj_matrix： the adjacency matrix for calculating shortest path
    # num_samples：The size of the adjacency matrix
    # start：To calculate the shortest path for all point-to-point starts
    # dis_inf：The initial shortest path for all point-to-point starts, set as inf

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
            # min_distance = jax.lax.cond(flag, lambda _: distances[v], lambda _: min_distance)
            # min_index = jax.lax.cond(flag, lambda _: v, lambda _: min_index)

        # Mark as visited
        # jax.lax.dynamic_update_slice(visited, 1, (min_index,))
        # visited[min_index] = True
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
