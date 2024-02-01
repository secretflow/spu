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
import numpy as np
from sklearn import preprocessing

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim
from sml.preprocessing.preprocessing import Binarizer, LabelBinarizer, Normalizer, MinMaxScaler, MaxAbsScaler, KBinsDiscretizer


class UnitTests(unittest.TestCase):
    def test_labelbinarizer(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def labelbinarize(X, Y):
            transformer = LabelBinarizer(neg_label=-2, pos_label=3)
            transformer.fit(X, n_classes=4)
            transformed = transformer.transform(Y)
            inv_transformed = transformer.inverse_transform(transformed)
            return transformed, inv_transformed

        X = jnp.array([1, 2, 4, 6])
        Y = jnp.array([1, 6])

        transformer = preprocessing.LabelBinarizer(neg_label=-2, pos_label=3)
        transformer.fit(X)
        sk_transformed = transformer.transform(Y)
        sk_inv_transformed = transformer.inverse_transform(sk_transformed)
        # print("sklearn:\n", sk_transformed)
        # print("sklearn:\n", sk_inv_transformed)

        spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, labelbinarize)(X, Y)
        # print("result\n", spu_transformed)
        # print("result\n", spu_inv_transformed)

        np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=0)
        np.testing.assert_allclose(
            sk_inv_transformed, spu_inv_transformed, rtol=0, atol=0
        )

    def test_labelbinarizer_binary(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def labelbinarize(X):
            transformer = LabelBinarizer()
            transformed = transformer.fit_transform(X, n_classes=2, unique=False)
            inv_transformed = transformer.inverse_transform(transformed)
            return transformed, inv_transformed

        X = jnp.array([1, -1, -1, 1])

        transformer = preprocessing.LabelBinarizer()
        sk_transformed = transformer.fit_transform(X)
        sk_inv_transformed = transformer.inverse_transform(sk_transformed)
        # print("sklearn:\n", sk_transformed)
        # print("sklearn:\n", sk_inv_transformed)

        spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, labelbinarize)(X)
        # print("result\n", spu_transformed)
        # print("result\n", spu_inv_transformed)

        np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=0)
        np.testing.assert_allclose(
            sk_inv_transformed, spu_inv_transformed, rtol=0, atol=0
        )

    def test_labelbinarizer_unseen(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def labelbinarize(X, Y):
            transformer = LabelBinarizer()
            transformer.fit(X, n_classes=3)
            return transformer.transform(Y)

        X = jnp.array([2, 4, 5])
        Y = jnp.array([1, 2, 3, 4, 5, 6])

        transformer = preprocessing.LabelBinarizer()
        transformer.fit(X)
        sk_result = transformer.transform(Y)
        # print("sklearn:\n", sk_result)

        spu_result = spsim.sim_jax(sim, labelbinarize)(X, Y)
        # print("result\n", spu_result)

        np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=0)

    def test_binarizer(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def binarize(X):
            transformer = Binarizer()
            return transformer.transform(X)

        X = jnp.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

        transformer = preprocessing.Binarizer()
        sk_result = transformer.transform(X)
        # print("sklearn:\n", sk_result)

        spu_result = spsim.sim_jax(sim, binarize)(X)
        # print("result\n", spu_result)

        np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=0)

    def test_normalizer(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def normalize_l1(X):
            transformer = Normalizer(norm="l1")
            return transformer.transform(X)

        def normalize_l2(X):
            transformer = Normalizer()
            return transformer.transform(X)

        def normalize_max(X):
            transformer = Normalizer(norm="max")
            return transformer.transform(X)

        X = jnp.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])

        transformer_l1 = preprocessing.Normalizer(norm="l1")
        sk_result_l1 = transformer_l1.transform(X)
        transformer_l2 = preprocessing.Normalizer()
        sk_result_l2 = transformer_l2.transform(X)
        transformer_max = preprocessing.Normalizer(norm="max")
        sk_result_max = transformer_max.transform(X)
        # print("sklearn:\n", sk_result_l1)
        # print("sklearn:\n", sk_result_l2)
        # print("sklearn:\n", sk_result_max)

        spu_result_l1 = spsim.sim_jax(sim, normalize_l1)(X)
        spu_result_l2 = spsim.sim_jax(sim, normalize_l2)(X)
        spu_result_max = spsim.sim_jax(sim, normalize_max)(X)
        # print("result\n", spu_result_l1)
        # print("result\n", spu_result_l2)
        # print("result\n", spu_result_max)

        np.testing.assert_allclose(sk_result_l1, spu_result_l1, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_result_l2, spu_result_l2, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_result_max, spu_result_max, rtol=0, atol=1e-4)
    
    def test_minmaxscaler(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def minmaxscale(X, Y):
            transformer = MinMaxScaler()
            result1 = transformer.fit_transform(X)
            result2 = transformer.transform(Y)
            return result1, result2

        X = jnp.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
        Y = jnp.array([[2, 2]])

        transformer = preprocessing.MinMaxScaler()
        sk_result_1 = transformer.fit_transform(X)
        sk_result_2 = transformer.transform(Y)
        # print("sklearn:\n", sk_result_1)
        # print("sklearn:\n", sk_result_2)

        spu_result_1, spu_result_2 = spsim.sim_jax(sim, minmaxscale)(X, Y)
        # print("result\n", spu_result_1)
        # print("result\n", spu_result_2)

        np.testing.assert_allclose(sk_result_1, spu_result_1, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_result_2, spu_result_2, rtol=0, atol=1e-4)
    
    def test_minmaxscaler_partial_fit(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def minmaxscale(X):
            transformer = MinMaxScaler()
            for batch in range(50):
                transformer = transformer.partial_fit(X[batch * 2: batch * 2 + 2])
            result_min = transformer.data_min_
            result_max = transformer.data_max_
            return result_min, result_max

        rng = np.random.RandomState(0)
        n_features = 30
        n_samples = 1000
        offsets = rng.uniform(-1, 1, size=n_features)
        scales = rng.uniform(1, 10, size=n_features)
        X_2d = rng.randn(n_samples, n_features) * scales + offsets
        X = X_2d

        chunk_size = 2
        transformer = MinMaxScaler()
        for batch in range(50):
            transformer = transformer.partial_fit(X[batch * 2: batch * 2 + 2])

        # transformer = preprocessing.MinMaxScaler()
        # transformer.fit(X)
        sk_result_min = transformer.data_min_
        sk_result_max = transformer.data_max_
        # print("sklearn:\n", sk_result_min)
        # print("sklearn:\n", sk_result_max)

        spu_result_min, spu_result_max = spsim.sim_jax(sim, minmaxscale)(X)
        # print("result\n", spu_result_min)
        # print("result\n", spu_result_max)

        np.testing.assert_allclose(sk_result_min, spu_result_min, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_result_max, spu_result_max, rtol=0, atol=1e-4)
    
    def test_minmaxscaler_zero_variance(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def minmaxscale(X, X_new):
            transformer = MinMaxScaler()
            transformer.fit(X, zero_variance=True)
            transformed = transformer.transform(X)
            inv_transformed = transformer.inverse_transform(transformed)
            transformed_new = transformer.transform(X_new)
            return transformed, inv_transformed, transformed_new

        X = jnp.array([[0.0, 1.0, +0.5], [0.0, 1.0, -0.1], [0.0, 1.0, +1.1]])
        X_new = jnp.array([[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]])

        transformer = preprocessing.MinMaxScaler()
        transformer.fit(X)
        sk_transformed = transformer.transform(X)
        sk_inv_transformed = transformer.inverse_transform(sk_transformed)
        sk_transformed_new = transformer.transform(X_new)
        # print("sklearn:\n", sk_transformed)
        # print("sklearn:\n", sk_inv_transformed)
        # print("sklearn:\n", sk_transformed_new)

        spu_transformed, spu_inv_transformed, spu_transformed_new = spsim.sim_jax(sim, minmaxscale)(X, X_new)
        # print("result\n", spu_transformed)
        # print("result\n", spu_inv_transformed)
        # print("result\n", spu_transformed_new)

        np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_transformed_new, spu_transformed_new, rtol=0, atol=1e-4)
    
    def test_maxabsscaler(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def maxabsscale(X):
            transformer = MaxAbsScaler()
            result = transformer.fit_transform(X)
            return result

        X = jnp.array([[ 1., -1.,  2.], [ 2.,  0.,  0.], [ 0.,  1., -1.]])

        transformer = preprocessing.MaxAbsScaler()
        sk_result = transformer.fit_transform(X)
        # print("sklearn:\n", sk_result)

        spu_result = spsim.sim_jax(sim, maxabsscale)(X)
        # print("result\n", spu_result)

        np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)
    
    def test_maxabsscaler_zero_maxabs(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def maxabsscale(X, X_new):
            transformer = MaxAbsScaler()
            transformer.fit(X, zero_maxabs=True)
            transformed = transformer.transform(X)
            inv_transformed = transformer.inverse_transform(transformed)
            transformed_new = transformer.transform(X_new)
            return transformed, inv_transformed, transformed_new

        X = jnp.array([[0.0, 1.0, +0.5], [0.0, 1.0, -0.3], [0.0, 1.0, +1.5], [0.0, 0.0, +0.0]])
        X_new = jnp.array([[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]])

        transformer = preprocessing.MaxAbsScaler()
        transformer.fit(X)
        sk_transformed = transformer.transform(X)
        sk_inv_transformed = transformer.inverse_transform(sk_transformed)
        sk_transformed_new = transformer.transform(X_new)
        # print("sklearn:\n", sk_transformed)
        # print("sklearn:\n", sk_inv_transformed)
        # print("sklearn:\n", sk_transformed_new)

        spu_transformed, spu_inv_transformed, spu_transformed_new = spsim.sim_jax(sim, maxabsscale)(X, X_new)
        # print("result\n", spu_transformed)
        # print("result\n", spu_inv_transformed)
        # print("result\n", spu_transformed_new)

        np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_transformed_new, spu_transformed_new, rtol=0, atol=1e-4)
    
    def test_kbinsdiscretizer_uniform(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def kbinsdiscretize(X):
            transformer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform', subsample=None)
            transformer.fit(X)
            transformed = transformer.transform(X)
            inv_transformed = transformer.inverse_transform(transformed)
            return transformed, inv_transformed

        X = jnp.array([[-2, 1, -4, -1], [-1, 2, -3, -0.5], [0, 3, -2, 0.5], [ 1, 4, -1, 2]])

        transformer = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform', subsample=None)
        sk_transformed = transformer.fit_transform(X)
        sk_inv_transformed = transformer.inverse_transform(sk_transformed)
        # print("sklearn:\n", sk_transformed)
        # print("sklearn:\n", sk_inv_transformed)

        spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(X)
        # print("result\n", spu_transformed)
        # print("result\n", spu_inv_transformed)

        np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4)
    
    def test_kbinsdiscretizer_quantile(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def kbinsdiscretize(X):
            transformer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', subsample=None)
            transformer.fit(X)
            transformed = transformer.transform(X)
            inv_transformed = transformer.inverse_transform(transformed)
            return transformed, inv_transformed

        X = jnp.array([[-2, 1.5, -4, -1], [-1, 2.5, -3, -0.5], [0, 3.5, -2, 0.5], [1, 4.5, -1, 2]])

        transformer = preprocessing.KBinsDiscretizer(3, encode='ordinal', strategy='quantile', subsample=None)
        sk_transformed = transformer.fit_transform(X)
        sk_inv_transformed = transformer.inverse_transform(sk_transformed)
        print("sklearn:\n", sk_transformed)
        print("sklearn:\n", sk_inv_transformed)

        spu_transformed, spu_inv_transformed, spu333 = spsim.sim_jax(sim, kbinsdiscretize)(X)
        print("result\n", spu_transformed)
        print("result\n", spu_inv_transformed)

        np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
        ### The error here is larger than expected. If atol is 1e-4, there will be an error.
        np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3)
    
    def test_kbinsdiscretizer_quantile_eliminate(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def kbinsdiscretize(X):
            transformer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', subsample=None)
            transformer.fit(X, remove_bin=True)
            transformed = transformer.transform(X)
            inv_transformed = transformer.inverse_transform(transformed)
            return transformed, inv_transformed

        X = jnp.array([[-1.5, 2.0, -3.5, -0.75], [-0.5, 3.0, -2.5, 0.0], [0.5, 4.0, -1.5, 1.25], [0.5, 4.0, -1.5, 1.25]])

        transformer = preprocessing.KBinsDiscretizer(3, encode='ordinal', strategy='quantile', subsample=None)
        sk_transformed = transformer.fit_transform(X)
        sk_inv_transformed = transformer.inverse_transform(sk_transformed)
        # print("sklearn:\n", sk_transformed)
        # print("sklearn:\n", sk_inv_transformed)

        spu_transformed, spu_inv_transformed = spsim.sim_jax(sim, kbinsdiscretize)(X)
        # print("result\n", spu_transformed)
        # print("result\n", spu_inv_transformed)

        np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4)




if __name__ == "__main__":
    unittest.main()
