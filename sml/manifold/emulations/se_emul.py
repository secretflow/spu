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
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph

import sml.utils.emulation as emulation
from sml.manifold.kneighbors import mpc_kneighbors_graph
from sml.manifold.SE import normalization, se


def emul_cpz(mode: emulation.Mode.MULTIPROCESS):
    try:
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        def SE(sX, num_samples, num_features, k, num_components):
            Knn,Knn1 = mpc_kneighbors_graph(sX, num_samples, num_features, k)
            # Knn = 0.5 * (Knn + Knn.T)
            # D, L = normalization(Knn)
            # ans = se(L, num_samples, D, num_components)
            return Knn,Knn1


        # Set sample size and dimensions
        num_samples = 10  # Number of samples
        num_features = 4  # Sample dimension
        k = 6  # Number of nearest neighbors
        num_components = 3  # Dimension after dimensionality reduction

        # Generate random input
        seed = int(time.time())
        key = jax.random.PRNGKey(seed)
        X = jax.random.uniform(
            key, shape=(num_samples, num_features), minval=0.0, maxval=1.0
        )

        sX = emulator.seal(X)
        Knn ,Knn1= emulator.run(
            SE,
            static_argnums=(
                1,
                2,
                3,
                4,
            ),
        )(sX, num_samples, num_features, k, num_components)
        # print('ans: \n', ans)
        print('Knn: \n',Knn)
        print('Knn1: \n',Knn1)
        
        
        # # sklearn test
        # affinity_matrix = kneighbors_graph(
        #     X, n_neighbors=k, mode="distance", include_self=False
        # )

        # # Make the matrix symmetric
        # affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        # # print(affinity_matrix)
        # embedding = spectral_embedding(
        #     affinity_matrix, n_components=num_components, random_state=None
        # )
        # # print('embedding: \n', embedding)

        # # Calculate the maximum difference between the results of SE and sklearn test, i.e. accuracy
        # # max_abs_diff = jnp.max(jnp.abs(jnp.abs(embedding) - jnp.abs(ans)))
        # # print(max_abs_diff)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_cpz(emulation.Mode.MULTIPROCESS)
