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

import jax.numpy as jnp
import numpy as np
from sklearn import preprocessing

import sml.utils.emulation as emulation
from sml.preprocessing.preprocessing import Binarizer, LabelBinarizer, Normalizer


def emul_labelbinarizer():
    def labelbinarize(X, Y):
        transformer = LabelBinarizer(neg_label=-2, pos_label=3)
        transformer.fit(X, n_classes=4)
        transformed = transformer.transform(Y)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([1, 2, 6, 4, 2])
    Y = jnp.array([1, 6])

    spu_transformed, spu_inv_transformed = emulator.run(labelbinarize)(X, Y)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    transformer = preprocessing.LabelBinarizer(neg_label=-2, pos_label=3)
    transformer.fit(X)
    sk_transformed = transformer.transform(Y)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)
    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=0)
    np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=0)


def emul_labelbinarizer_binary():
    def labelbinarize(X, Y):
        transformer = LabelBinarizer()
        transformed = transformer.fit_transform(X, n_classes=2)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([1, -1, -1, 1])
    Y = jnp.array([1, 6])

    spu_transformed, spu_inv_transformed = emulator.run(labelbinarize)(X, Y)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    transformer = preprocessing.LabelBinarizer()
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)
    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=0)
    np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=0)


def emul_labelbinarizer_unseen():
    def labelbinarize(X, Y):
        transformer = LabelBinarizer()
        transformer.fit(X, n_classes=3)
        return transformer.transform(Y, unseen=True)

    X = jnp.array([2, 4, 5])
    Y = jnp.array([1, 2, 3, 4, 5, 6])

    spu_result = emulator.run(labelbinarize)(X, Y)
    # print("result\n", spu_result)

    transformer = preprocessing.LabelBinarizer()
    transformer.fit(X)
    sk_result = transformer.transform(Y)
    # print("sklearn:\n", sk_result)
    np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=0)


def emul_binarizer():
    def binarize(X):
        transformer = Binarizer()
        return transformer.transform(X)

    X = jnp.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

    spu_result = emulator.run(binarize)(X)
    # print("result\n", spu_result)

    transformer = preprocessing.Binarizer()
    sk_result = transformer.transform(X)
    # print("sklearn:\n", sk_result)
    np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=0)


def emul_normalizer():
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

    spu_result_l1 = emulator.run(normalize_l1)(X)
    spu_result_l2 = emulator.run(normalize_l2)(X)
    spu_result_max = emulator.run(normalize_max)(X)
    # print("result\n", spu_result_l1)
    # print("result\n", spu_result_l2)
    # print("result\n", spu_result_max)

    transformer_l1 = preprocessing.Normalizer(norm="l1")
    sk_result_l1 = transformer_l1.transform(X)
    transformer_l2 = preprocessing.Normalizer()
    sk_result_l2 = transformer_l2.transform(X)
    transformer_max = preprocessing.Normalizer(norm="max")
    sk_result_max = transformer_max.transform(X)
    # print("sklearn:\n", sk_result_l1)
    # print("sklearn:\n", sk_result_l2)
    # print("sklearn:\n", sk_result_max)
    np.testing.assert_allclose(sk_result_l1, spu_result_l1, rtol=0, atol=1e-4)
    np.testing.assert_allclose(sk_result_l2, spu_result_l2, rtol=0, atol=1e-4)
    np.testing.assert_allclose(sk_result_max, spu_result_max, rtol=0, atol=1e-4)


if __name__ == "__main__":
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC,
            emulation.Mode.MULTIPROCESS,
            bandwidth=300,
            latency=20,
        )
        emulator.up()
        emul_labelbinarizer()
        emul_labelbinarizer_binary()
        emul_labelbinarizer_unseen()
        emul_binarizer()
        emul_normalizer()
    finally:
        emulator.down()
