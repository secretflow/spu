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

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_blobs

import spu.spu_pb2 as spu_pb2  # type: ignore
import spu.utils.simulation as spsim
from sml.cluster.kmeans import KMEANS


class UnitTests(unittest.TestCase):
    def test_kmeans(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def proc(x1, x2):
            x = jnp.concatenate((x1, x2), axis=1)
            model = KMEANS(
                n_clusters=2, n_samples=x.shape[0], init="random", max_iter=10
            )

            return model.fit(x).predict(x)

        def load_data():
            n_samples = 1000
            n_features = 100
            X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=2)
            split_index = n_features // 2
            return X[:, :split_index], X[:, split_index:]

        x1, x2 = load_data()
        X = jnp.concatenate((x1, x2), axis=1)
        result = spsim.sim_jax(sim, proc)(x1, x2)
        print("result\n", result)

        # Compare with sklearn
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=2)
        print("sklearn:\n", model.fit(X).predict(X))

    def test_kmeans_kmeans_plus_plus(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        X = jnp.array([[-4, -3, -2, -1], [-4, -3, -2, -1]]).T

        # define model in outer scope
        # then __init__ will be computed in plaintext
        # this is necessary since k-means++ needs to
        # generate random numbers in __init__
        # jax.random.uniform will cause great error
        # in SPU runtime
        model = KMEANS(
            n_clusters=4,
            n_samples=X.shape[0],
            init="k-means++",
            n_init=1,
            max_iter=10,
        )

        def proc(x):
            model.fit(x)
            return model._centers.sort(axis=0)

        result = spsim.sim_jax(sim, proc)(X)
        # print("result\n", result)

        # Compare with sklearn
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=4, n_init=1, max_iter=10)
        model.fit(X)
        sk_result = model.cluster_centers_
        sk_result.sort(axis=0)
        # print("sklearn:\n", sk_result)

        np.testing.assert_allclose(result, sk_result, rtol=0, atol=1e-4)

    def test_kmeans_init_array(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def proc(x, init):
            model = KMEANS(
                n_clusters=4, n_samples=x.shape[0], init=init, n_init=1, max_iter=10
            )
            model.fit(x)
            return model._centers

        X = jnp.array([[-4, -3, -2, -1]]).T
        uniform_edges = np.linspace(np.min(X), np.max(X), 5)
        init_array = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
        result = spsim.sim_jax(sim, proc)(X, init_array)
        # print("result\n", result)

        # Compare with sklearn
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=4, init=init_array, n_init=1)
        model.fit(X)
        sk_result = model.cluster_centers_
        # print("sklearn:\n", sk_result)
        np.testing.assert_allclose(result, sk_result, rtol=0, atol=1e-4)

    def test_kmeans_random(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        X = jnp.array([[-4, -3, -2, -1], [-4, -3, -2, -1]]).T

        # define model in outer scope
        # then __init__ will be computed in plaintext
        # this is better since random init needs to
        # randomly choose numbers in __init__
        # define model in SPU runtime won't cause error
        # but it requires much larger n_init
        # to get the correct result in some cases
        # (since jax.random.choice did not work well in SPU runtime)
        model = KMEANS(
            n_clusters=4,
            n_samples=X.shape[0],
            init="random",
            n_init=5,
            max_iter=10,
        )

        def proc(x):
            model.fit(x)
            return model._centers.sort(axis=0)

        result = spsim.sim_jax(sim, proc)(X)
        # print("result\n", result)

        # Compare with sklearn
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=4, init="random", n_init=5, max_iter=10)
        model.fit(X)
        sk_result = model.cluster_centers_
        sk_result.sort(axis=0)
        # print("sklearn:\n", sk_result)

        np.testing.assert_allclose(result, sk_result, rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
