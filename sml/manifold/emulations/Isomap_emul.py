import jax
import jax.numpy as jnp
import numpy as np
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph

import sml.utils.emulation as emulation
from sml.manifold.dijkstra import mpc_dijkstra
from sml.manifold.kneighbors import mpc_kneighbors_graph
from sml.manifold.MDS import mds


def emul_cpz(mode: emulation.Mode.MULTIPROCESS):
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        def mpc_isomap(
            sX,
            mpc_dist_inf,
            mpc_shortest_paths,
            num_samples,
            num_features,
            k,
            num_components,
        ):
            Knn = mpc_kneighbors_graph(sX, num_samples, num_features, k)

            # for i in range(num_samples):
            #     distances = mpc_dijkstra(Knn, num_samples, i, mpc_dist_inf)
            #     mpc_shortest_paths = mpc_shortest_paths.at[i].set(distances)
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

            B, ans, values, vectors = mds(
                mpc_shortest_paths, num_samples, num_components
            )
            return Knn, mpc_shortest_paths, B, ans, values, vectors


        # 设置样本数量和维度
        num_samples = 6
        num_features = 3
        k = 3
        num_components = 2
        X = jnp.array(
            [
                [0.122, 0.114, 0.64],
                [0.136, 0.204, 0.25],
                [0.11, 0.145, 0.24],
                [0.16, 0.81, 0.91],
                [0.209, 0.122, 0.76],
                [0.148, 0.119, 0.15],
            ]
        )
        dist_inf = jnp.full(num_samples, np.inf)
        shortest_paths = jnp.zeros((num_samples, num_samples))

        sX, mpc_dist_inf, mpc_shortest_paths = emulator.seal(
            X, dist_inf, shortest_paths
        )
        Knn, mpc_shortest_paths, B, ans, values, vectors = emulator.run(
            mpc_isomap, static_argnums=(3, 4, 5, 6)
        )(
            sX,
            mpc_dist_inf,
            mpc_shortest_paths,
            num_samples,
            num_features,
            k,
            num_components,
        )

        # print('shortest_paths: \n',shortest_paths)
        # print('Knn: \n',Knn)
        # print('B: \n', B)
        print('ans: \n', ans)
        # print('values: \n', values)
        # print('vectors: \n', vectors)

        # sklearn test
        affinity_matrix = kneighbors_graph(
            X, n_neighbors=k, mode="distance", include_self=False
        )
        # print('affinity_matrix1: \n',affinity_matrix.toarray())
        # 使矩阵对称
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        # print('affinity_matrix2: \n',affinity_matrix.toarray())

        # dist_matrix = shortest_path(affinity_matrix, method="D", directed=False)
        # print('dist_matrix: \n',dist_matrix)

        affinity_matrix = affinity_matrix.toarray()
        for i in range(1, num_samples):
            for j in range(i):
                if affinity_matrix[i][j] == 0:
                    affinity_matrix[i][j] = 10000
                    affinity_matrix[j][i] = 10000

        embedding = Isomap(n_components=num_components, metric='precomputed')
        X_transformed = embedding.fit_transform(affinity_matrix)
        print('X_transformed: \n', X_transformed)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_cpz(emulation.Mode.MULTIPROCESS)
