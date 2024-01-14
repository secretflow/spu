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
from sml.preprocessing.preprocessing import Binarizer, LabelBinarizer, Normalizer


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

        X = jnp.array([1, 2, 6, 4, 2])
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

        def labelbinarize(X, Y):
            transformer = LabelBinarizer()
            # transformer.fit(X, n_classes=4)
            transformed = transformer.fit_transform(X, n_classes=2)
            inv_transformed = transformer.inverse_transform(transformed)
            return transformed, inv_transformed

        X = jnp.array([1, -1, -1, 1])
        Y = jnp.array([1, 6])

        transformer = preprocessing.LabelBinarizer()
        sk_transformed = transformer.fit_transform(X)
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

        spu_result_l1 = spsim.sim_jax(sim, normalize_l1)(X)
        spu_result_l2 = spsim.sim_jax(sim, normalize_l2)(X)
        spu_result_max = spsim.sim_jax(sim, normalize_max)(X)

        transformer_l1 = preprocessing.Normalizer(norm="l1")
        sk_result_l1 = transformer_l1.transform(X)
        transformer_l2 = preprocessing.Normalizer()
        sk_result_l2 = transformer_l2.transform(X)
        transformer_max = preprocessing.Normalizer(norm="max")
        sk_result_max = transformer_max.transform(X)
        # print("sklearn:\n", sk_result_l1)
        # print("sklearn:\n", sk_result_l2)
        # print("sklearn:\n", sk_result_max)

        # print("result\n", spu_result_l1)
        # print("result\n", spu_result_l2)
        # print("result\n", spu_result_max)

        np.testing.assert_allclose(sk_result_l1, spu_result_l1, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_result_l2, spu_result_l2, rtol=0, atol=1e-4)
        np.testing.assert_allclose(sk_result_max, spu_result_max, rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
