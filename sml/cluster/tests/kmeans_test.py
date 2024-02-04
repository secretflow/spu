# Copyright 2023 Ant Group Co., Ltd.
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

import unittest

import jax.numpy as jnp
from sklearn.datasets import make_blobs

import spu.spu_pb2 as spu_pb2  # type: ignore
import spu.utils.simulation as spsim
from sml.cluster.kmeans import KMEANS


class UnitTests(unittest.TestCase):
    # def test_kmeans(self):
    #     sim = spsim.Simulator.simple(
    #         3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
    #     )

    #     def proc(x1, x2):
    #         x = jnp.concatenate((x1, x2), axis=1)
    #         model = KMEANS(n_clusters=2, n_samples=x.shape[0], max_iter=10)

    #         return model.fit(x).predict(x)

    #     def load_data():
    #         n_samples = 1000
    #         n_features = 100
    #         X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=2)
    #         split_index = n_features // 2
    #         return X[:, :split_index], X[:, split_index:]

    #     x1, x2 = load_data()
    #     X = jnp.concatenate((x1, x2), axis=1)
    #     result = spsim.sim_jax(sim, proc)(x1, x2)
    #     # print("result\n", result)

    #     # Compare with sklearn
    #     from sklearn.cluster import KMeans

    #     model = KMeans(n_clusters=2)
    #     # print("sklearn:\n", model.fit(X).predict(X))
    
    def test_kmeans(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )
        def proc(x):
            model = KMEANS(n_clusters=3, n_samples=x.shape[0], n_init=10, max_iter=10)
            model.fit(x)
            return model._centers
            return model._inertia
        # X = jnp.array([[-4, -3, -2, -1]]).T
        X = jnp.array([[-4, -3, -2, -1], [-4, -3, -2, -1]]).T
        result = spsim.sim_jax(sim, proc)(X)
        print("result\n", result)

        # Compare with sklearn
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=3)
        model.fit(X)
        print("sklearn:\n", model.cluster_centers_)




if __name__ == "__main__":
    unittest.main()
