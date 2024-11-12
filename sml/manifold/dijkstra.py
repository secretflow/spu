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
import jax
import jax.numpy as jnp


def set_value(x, index, value, n):
    # 将数组x的index索引处的值修改为value，其中index是秘密共享的
    perm = jnp.zeros(n, dtype=jnp.int16)
    perm_2 = jnp.zeros(n, dtype=jnp.int16)
    for i in range(n):
        perm = perm.at[i].set(i)
        perm_2 = perm_2.at[i].set(index)
    flag = jnp.equal(perm, perm_2)
    set_x = jnp.select([flag], [value], x)

    return set_x


def get_value_1(x, index, n):
    # 获得x[index]索引处的值，其中index是秘密共享的
    perm = jnp.zeros(n, dtype=jnp.int16)
    perm_2 = jnp.zeros(n, dtype=jnp.int16)
    for i in range(n):
        perm = perm.at[i].set(i)
        perm_2 = perm_2.at[i].set(index)
    flag = jnp.equal(perm, perm_2)
    return jnp.sum(flag * x)


def get_value_2(x, index_1, index_2, n):
    # 获得x[index_1][index_2]索引处的值，其中index_2是明文，index_1是秘密共享的
    # 初始化行索引
    perm_1 = jnp.zeros((n, n), dtype=jnp.int16)
    perm_2_row = jnp.zeros((n, n), dtype=jnp.int16)

    for i in range(n):
        for j in range(n):
            perm_1 = perm_1.at[i, j].set(i)
            perm_2_row = perm_2_row.at[i, j].set(index_1)

    # 行匹配
    flag_row = jnp.equal(perm_1, perm_2_row)

    # 使用明文 index_2 直接提取列的值
    flag = flag_row[:, index_2]

    # 返回匹配索引处的值
    return jnp.sum(flag * x[:, index_2])


def mpc_dijkstra(adj_matrix, num_samples, start, dist_inf):
    # adj_matrix：要求最短路径的邻接矩阵
    # num_samples：邻接矩阵的大小
    # start：要计算所有点到点start的最短路径
    # dis_inf：所有点到点start的初始最短路径，设置为inf

    # 用inf值初始化
    
    sinf = dist_inf[0]
    distances = dist_inf

    # 使用 Dijkstra 算法计算从起始点到其他点的最短路径
    distances = distances.at[start].set(0)
    # visited = [False] * num_samples
    visited = jnp.zeros(num_samples, dtype=bool)  # 初始化为 False 的数组
    visited = jnp.array(visited)
    
    for i in range(num_samples):
        # 找到当前未访问的最近节点
        
        min_distance = sinf
        min_index = -1
        for v in range(num_samples):
            flag = (visited[v] == 0) * (distances[v] < min_distance)
            min_distance = min_distance + flag * (distances[v] - min_distance)
            min_index = min_index + flag * (v - min_index)
            # min_distance = jax.lax.cond(flag, lambda _: distances[v], lambda _: min_distance)
            # min_index = jax.lax.cond(flag, lambda _: v, lambda _: min_index)
        
        # 标记为已访问
        # jax.lax.dynamic_update_slice(visited, 1, (min_index,))
        # visited[min_index] = True
        visited = set_value(visited, min_index, True, num_samples)

        # 更新邻接节点的距离
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
