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
import numpy as np

import sml.utils.emulation as emulation
from sml.manifold.dijkstra import mpc_dijkstra
from sml.manifold.floyd import floyd
from sml.manifold.floyd import floyd_opt


def emul_cpz(mode: emulation.Mode.MULTIPROCESS):
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        def dijkstra_all_pairs(
            Knn,
            mpc_dist_inf,
            num_samples,
        ):
    
            def compute_distances_for_sample(i, Knn, num_samples, mpc_dist_inf):
                return mpc_dijkstra(Knn, num_samples, i, mpc_dist_inf)

            # 使用 vmap 来并行化计算每个样本的最短路径
            compute_distances = jax.vmap(
                compute_distances_for_sample, in_axes=(0, None, None, None)
            )

            # 并行化执行所有样本的最短路径计算
            indices = jnp.arange(num_samples)  # 样本索引
            mpc_shortest_paths = compute_distances(
                indices, Knn, num_samples, mpc_dist_inf
            )
            
            return mpc_shortest_paths

        # 设置样本数量和维度
        num_samples = 20
        dist_inf = jnp.full(num_samples, np.inf)

        # 初始化邻接矩阵
        Knn = np.random.rand(num_samples, num_samples)
        # Knn = (Knn + Knn.T) * 100
        Knn = (Knn + Knn.T) / 2
        Knn[Knn == 0] = np.inf
        np.fill_diagonal(Knn, 0)
        # print("\nadjacency_matrix:\n")
        # for row in Knn:
        #     print(row)
        mpc_dist_inf = emulator.seal(dist_inf)

        # dijkstra_all_pairs
        Knn=emulator.seal(Knn)
        shortest_paths_dijkstra= emulator.run(
            dijkstra_all_pairs, static_argnums=(2,)
        )(
            Knn,
            mpc_dist_inf,
            num_samples
        )
        
        # floyd_opt
        # shortest_paths_floyd= emulator.run(floyd)(Knn)
        shortest_paths_opt_floyd= emulator.run(floyd_opt)(Knn)
        # are_equal = np.array_equal(shortest_paths_dijkstra, shortest_paths_floyd)
        # if are_equal:
        #     print("计算结果相同。")
        # else:
        #     print("计算结果不同!")

        print("\nshortest_paths_dijkstra:\n")
        for row in shortest_paths_dijkstra:
            print(row)

        # print("\nshortest_paths_floyd:\n")
        # for row in shortest_paths_floyd:
        #     print(row)

        print("\nshortest_paths_opt_floyd:\n")
        for row in shortest_paths_opt_floyd:
            print(row)







        # # sklearn test
        # affinity_matrix = kneighbors_graph(
        #     X, n_neighbors=k, mode="distance", include_self=False
        # )
        # # print('affinity_matrix1: \n',affinity_matrix.toarray())
        # # 使矩阵对称
        # affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        # # print('affinity_matrix2: \n',affinity_matrix.toarray())

        # # dist_matrix = shortest_path(affinity_matrix, method="D", directed=False)
        # # print('dist_matrix: \n',dist_matrix)

        # affinity_matrix = affinity_matrix.toarray()
        # for i in range(1, num_samples):
        #     for j in range(i):
        #         if affinity_matrix[i][j] == 0:
        #             affinity_matrix[i][j] = 10000
        #             affinity_matrix[j][i] = 10000

        # embedding = Isomap(n_components=num_components, metric='precomputed')
        # X_transformed = embedding.fit_transform(affinity_matrix)
        # print('X_transformed: \n', X_transformed)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_cpz(emulation.Mode.MULTIPROCESS)
