import unittest
import jax.numpy as jnp

import spu.utils.simulation as spsim
import spu.spu_pb2 as spu_pb2  # type: ignore

from sml.kmeans.kmeans import KMEANS
from sklearn.datasets import make_blobs


class UnitTests(unittest.TestCase):
    def test_kmeans(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def proc(x1, x2):
            x = jnp.concatenate((x1, x2), axis=1)
            model = KMEANS(n_clusters=2, n_samples=x.shape[0], max_iter=10)

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


if __name__ == "__main__":
    unittest.main()
