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
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import MDS, Isomap
from sklearn.neighbors import kneighbors_graph

import sml.utils.emulation as emulation
from sml.manifold.dijkstra import mpc_dijkstra
from sml.manifold.floyd import floyd_opt
from sml.manifold.kneighbors import mpc_kneighbors_graph
from sml.manifold.MDS import mds


def emul_cpz(mode: emulation.Mode.MULTIPROCESS):
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # The shortest path method in Isomap uses floyd
        def mpc_isomap_floyd(
            sX,
            mpc_shortest_paths,
            num_samples,
            num_features,
            k,
            num_components,
        ):

            Knn = mpc_kneighbors_graph(sX, num_samples, num_features, k)

            Knn = (Knn == 0) * jnp.inf + Knn
            Knn = jnp.where(jnp.eye(Knn.shape[0]), 0, Knn)
            flag = Knn <= Knn.T
            Knn = flag * Knn + (1 - flag) * Knn.T

            mpc_shortest_paths = floyd_opt(Knn)

            B, ans, values, vectors = mds(
                mpc_shortest_paths, num_samples, num_components
            )
            return Knn, mpc_shortest_paths, B, ans, values, vectors

        # Set sample size and dimensions
        num_samples = 6  # Number of samples
        num_features = 3  # Sample dimension
        k = 3  # Number of nearest neighbors
        num_components = 2  # Dimension after dimensionality reduction

        # Generate random input
        seed = int(time.time())
        key = jax.random.PRNGKey(seed)
        X = jax.random.uniform(
            key, shape=(num_samples, num_features), minval=0.0, maxval=1.0
        )

        shortest_paths = jnp.zeros((num_samples, num_samples))

        sX, mpc_shortest_paths = emulator.seal(X, shortest_paths)

        Knn, mpc_shortest_paths, B, ans, values, vectors = emulator.run(
            mpc_isomap_floyd, static_argnums=(2, 3, 4, 5)
        )(
            sX,
            mpc_shortest_paths,
            num_samples,
            num_features,
            k,
            num_components,
        )
        # print('Knn: \n',Knn)
        # print('mpc_shortest_paths: \n', mpc_shortest_paths)
        # print('values: \n',values)
        # print('vectors: \n',vectors)

        print('ans: \n', ans)

        # sklearn test
        embedding = Isomap(n_components=num_components, n_neighbors=k)
        X_transformed = embedding.fit_transform(X)
        print('X_transformed: \n', X_transformed)

        # Calculate the maximum difference between the results of SE and sklearn test, i.e. accuracy
        max_abs_diff = jnp.max(jnp.abs(jnp.abs(X_transformed) - jnp.abs(ans)))
        print(max_abs_diff)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_cpz(emulation.Mode.MULTIPROCESS)
