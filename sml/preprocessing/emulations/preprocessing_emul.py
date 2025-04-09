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
import sml.utils.emulation as emulation
from sklearn import preprocessing
from sml.preprocessing.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    LabelBinarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    RobustScaler,
)


def emul_labelbinarizer():
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

    X, Y = emulator.seal(X, Y)
    spu_transformed, spu_inv_transformed = emulator.run(labelbinarize)(X, Y)
    # print("spu:\n", spu_transformed)
    # print("spu:\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=0)
    np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=0)


def emul_labelbinarizer_binary():
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

    X = emulator.seal(X)
    spu_transformed, spu_inv_transformed = emulator.run(labelbinarize)(X)
    # print("spu:\n", spu_transformed)
    # print("spu:\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=0)
    np.testing.assert_allclose(sk_inv_transformed, spu_inv_transformed, rtol=0, atol=0)


def emul_labelbinarizer_unseen():
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

    X, Y = emulator.seal(X, Y)
    spu_result = emulator.run(labelbinarize)(X, Y)
    # print("spu:\n", spu_result)

    np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=0)


def emul_binarizer():
    def binarize(X):
        transformer = Binarizer()
        return transformer.transform(X)

    X = jnp.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

    transformer = preprocessing.Binarizer()
    sk_result = transformer.transform(X)
    # print("sklearn:\n", sk_result)

    X = emulator.seal(X)
    spu_result = emulator.run(binarize)(X)
    # print("spu:\n", spu_result)

    np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=0)


def emul_onehotEncoder():
    manual_categories = [[1, 1.1, 3.25], [2.0, 4.32, 6.10]]

    X = jnp.array([[1, 2.0], [3.25, 4.32], [1.1, 6.10]], dtype=jnp.float64)
    Y = jnp.array([[1, 2.1], [3.21, 4.32], [1.1, 6.10]], dtype=jnp.float64)

    sk_X = np.array([[1, 2.0], [3.25, 4.32], [1.1, 6.10]], dtype=np.float64)
    sk_Y = np.array([[1, 2.1], [3.21, 4.32], [1.1, 6.10]], dtype=np.float64)

    def onehotEncode(X, Y):
        onehotEncoder = OneHotEncoder(categories=manual_categories)
        onehotEncoder.fit(X)
        encoded = onehotEncoder.transform(Y)
        inverse_v = onehotEncoder.inverse_transform(encoded)
        return encoded, inverse_v

    sk_onehotEncoder = preprocessing.OneHotEncoder(
        categories=manual_categories, handle_unknown="ignore", sparse_output=False
    )
    sk_onehotEncoder.fit(sk_X)
    sk_transformed = sk_onehotEncoder.transform(sk_Y)
    sk_inv_transformed = sk_onehotEncoder.inverse_transform(sk_transformed)
    sk_inv_transformed = np.where(sk_inv_transformed == None, 0.0, sk_inv_transformed)

    X, Y = emulator.seal(X, Y)
    spu_transformed, spu_inv_transformed = emulator.run(onehotEncode)(X, Y)

    sk_inv_transformed = sk_inv_transformed.astype(np.float64)
    spu_inv_transformed = spu_inv_transformed.astype(np.float64)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=1e-4, atol=1e-4
    )


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

    transformer_l1 = preprocessing.Normalizer(norm="l1")
    sk_result_l1 = transformer_l1.transform(X)
    transformer_l2 = preprocessing.Normalizer()
    sk_result_l2 = transformer_l2.transform(X)
    transformer_max = preprocessing.Normalizer(norm="max")
    sk_result_max = transformer_max.transform(X)
    # print("sklearn:\n", sk_result_l1)
    # print("sklearn:\n", sk_result_l2)
    # print("sklearn:\n", sk_result_max)

    X = emulator.seal(X)
    spu_result_l1 = emulator.run(normalize_l1)(X)
    spu_result_l2 = emulator.run(normalize_l2)(X)
    spu_result_max = emulator.run(normalize_max)(X)
    # print("spu:\n", spu_result_l1)
    # print("spu:\n", spu_result_l2)
    # print("spu:\n", spu_result_max)

    np.testing.assert_allclose(sk_result_l1, spu_result_l1, rtol=0, atol=1e-4)
    np.testing.assert_allclose(sk_result_l2, spu_result_l2, rtol=0, atol=1e-4)
    np.testing.assert_allclose(sk_result_max, spu_result_max, rtol=0, atol=1e-4)


def emul_robustscaler():

    def robustscale(X, Y):
        transformer = RobustScaler(quantile_range=(25.0, 75.0))
        result1 = transformer.fit_transform(X)
        result2 = transformer.transform(Y)
        result1_restore = transformer.inverse_transform(result1)
        result2_restore = transformer.inverse_transform(result2)
        return result1, result2, result1_restore, result2_restore

    X = jnp.array([[-2, 0.5], [-0.5, 1.5], [0, 10.0], [1, 15.0], [5, 20.0]])
    Y = jnp.array([[3, 2]])

    # sklearn基准计算
    sk_transformer = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
    sk_result_1 = sk_transformer.fit_transform(X)
    sk_result_2 = sk_transformer.transform(Y)
    sk_restore_1 = sk_transformer.inverse_transform(sk_result_1)
    sk_restore_2 = sk_transformer.inverse_transform(sk_result_2)

    # SPU模拟计算
    X, Y = emulator.seal(X, Y)
    spu_result_1, spu_result_2, spu_restore_1, spu_restore_2 = emulator.run(
        robustscale
    )(X, Y)

    # 正向变换断言
    np.testing.assert_allclose(sk_result_1, spu_result_1, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sk_result_2, spu_result_2, rtol=1e-4, atol=1e-4)

    # 逆变换断言
    np.testing.assert_allclose(X, spu_restore_1, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(Y, spu_restore_2, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sk_restore_1, spu_restore_1, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sk_restore_2, spu_restore_2, rtol=1e-4, atol=1e-4)


def emul_minmaxscaler():
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

    X, Y = emulator.seal(X, Y)
    spu_result_1, spu_result_2 = emulator.run(minmaxscale)(X, Y)
    # print("result\n", spu_result_1)
    # print("result\n", spu_result_2)

    np.testing.assert_allclose(sk_result_1, spu_result_1, rtol=0, atol=1e-4)
    np.testing.assert_allclose(sk_result_2, spu_result_2, rtol=0, atol=1e-4)


def emul_minmaxscaler_partial_fit():
    def minmaxscale(X):
        transformer = MinMaxScaler()
        for batch in range(50):
            transformer = transformer.partial_fit(X[batch * 2 : batch * 2 + 2])
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
        transformer = transformer.partial_fit(X[batch * 2 : batch * 2 + 2])

    # transformer = preprocessing.MinMaxScaler()
    # transformer.fit(X)
    sk_result_min = transformer.data_min_
    sk_result_max = transformer.data_max_
    # print("sklearn:\n", sk_result_min)
    # print("sklearn:\n", sk_result_max)

    X = emulator.seal(X)
    spu_result_min, spu_result_max = emulator.run(minmaxscale)(X)
    # print("result\n", spu_result_min)
    # print("result\n", spu_result_max)

    np.testing.assert_allclose(sk_result_min, spu_result_min, rtol=0, atol=1e-4)
    np.testing.assert_allclose(sk_result_max, spu_result_max, rtol=0, atol=1e-4)


def emul_minmaxscaler_zero_variance():
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

    X, X_new = emulator.seal(X, X_new)
    spu_transformed, spu_inv_transformed, spu_transformed_new = emulator.run(
        minmaxscale
    )(X, X_new)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)
    # print("result\n", spu_transformed_new)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        sk_transformed_new, spu_transformed_new, rtol=0, atol=1e-4
    )


def emul_maxabsscaler():
    def maxabsscale(X):
        transformer = MaxAbsScaler()
        result = transformer.fit_transform(X)
        return result

    X = jnp.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

    transformer = preprocessing.MaxAbsScaler()
    sk_result = transformer.fit_transform(X)
    # print("sklearn:\n", sk_result)

    X = emulator.seal(X)
    spu_result = emulator.run(maxabsscale)(X)
    # print("result\n", spu_result)

    np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)


def emul_maxabsscaler_zero_maxabs():
    def maxabsscale(X, X_new):
        transformer = MaxAbsScaler()
        transformer.fit(X, zero_maxabs=True)
        transformed = transformer.transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        transformed_new = transformer.transform(X_new)
        return transformed, inv_transformed, transformed_new

    X = jnp.array(
        [[0.0, 1.0, +0.5], [0.0, 1.0, -0.3], [0.0, 1.0, +1.5], [0.0, 0.0, +0.0]]
    )
    X_new = jnp.array([[+0.0, 2.0, 0.5], [-1.0, 1.0, 0.0], [+0.0, 1.0, 1.5]])

    transformer = preprocessing.MaxAbsScaler()
    transformer.fit(X)
    sk_transformed = transformer.transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    sk_transformed_new = transformer.transform(X_new)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)
    # print("sklearn:\n", sk_transformed_new)

    X, X_new = emulator.seal(X, X_new)
    spu_transformed, spu_inv_transformed, spu_transformed_new = emulator.run(
        maxabsscale
    )(X, X_new)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)
    # print("result\n", spu_transformed_new)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        sk_transformed_new, spu_transformed_new, rtol=0, atol=1e-4
    )


def emul_kbinsdiscretizer_uniform():
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(n_bins=3, strategy="uniform")
        transformed = transformer.fit_transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[-2, 1, -4, -1], [-1, 2, -3, -0.5], [0, 3, -2, 0.5], [1, 4, -1, 2]])

    transformer = preprocessing.KBinsDiscretizer(
        n_bins=3, encode="ordinal", strategy="uniform", subsample=None
    )
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X = emulator.seal(X)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(X)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def emul_kbinsdiscretizer_uniform_diverse_n_bins():
    def kbinsdiscretize(X, n_bins):
        transformer = KBinsDiscretizer(
            n_bins=3, diverse_n_bins=n_bins, strategy="uniform"
        )
        transformed = transformer.fit_transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([2, 3, 3, 3])

    transformer = preprocessing.KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="uniform", subsample=None
    )
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X, n_bins = emulator.seal(X, n_bins)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(X, n_bins)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def emul_kbinsdiscretizer_uniform_diverse_n_bins_no_vectorize():
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(
            n_bins=3, diverse_n_bins=np.array([2, 3, 3, 3]), strategy="uniform"
        )
        transformed = transformer.fit_transform(X, vectorize=False)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([2, 3, 3, 3])

    transformer = preprocessing.KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="uniform", subsample=None
    )
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X = emulator.seal(X)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(X)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def emul_kbinsdiscretizer_quantile():
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(n_bins=3, strategy="quantile")
        transformed = transformer.fit_transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array(
        [[-2, 1.5, -4, -1], [-1, 2.5, -3, -0.5], [0, 3.5, -2, 0.5], [1, 4.5, -1, 2]]
    )

    transformer = preprocessing.KBinsDiscretizer(
        3, encode="ordinal", strategy="quantile", subsample=None
    )
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X = emulator.seal(X)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(X)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3
    )


def emul_kbinsdiscretizer_quantile_diverse_n_bins():
    def kbinsdiscretize(X, n_bins):
        transformer = KBinsDiscretizer(
            n_bins=3, diverse_n_bins=n_bins, strategy="quantile"
        )
        transformed = transformer.fit_transform(X, remove_bin=True)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([2, 3, 3, 3])

    transformer = preprocessing.KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
    )
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X, n_bins = emulator.seal(X, n_bins)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(X, n_bins)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3
    )


def emul_kbinsdiscretizer_quantile_diverse_n_bins2():
    def kbinsdiscretize(X, n_bins):
        transformer = KBinsDiscretizer(
            n_bins=4, diverse_n_bins=n_bins, strategy="quantile"
        )
        transformed = transformer.fit_transform(X, remove_bin=True)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    n_bins = jnp.array([2, 4, 4, 4])

    transformer = preprocessing.KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
    )
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X, n_bins = emulator.seal(X, n_bins)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(X, n_bins)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3
    )


def emul_kbinsdiscretizer_quantile_diverse_n_bins_no_vectorize():
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(
            n_bins=3, diverse_n_bins=np.array([2, 3, 3, 3]), strategy="quantile"
        )
        transformed = transformer.fit_transform(X, vectorize=False, remove_bin=True)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([2, 3, 3, 3])

    transformer = preprocessing.KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
    )
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X = emulator.seal(X)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(X)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3
    )


def emul_kbinsdiscretizer_quantile_eliminate():
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(n_bins=3, strategy="quantile")
        transformed = transformer.fit_transform(X, remove_bin=True)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array(
        [
            [-1.5, 2.0, -3.5, -0.75],
            [-0.5, 3.0, -2.5, 0.0],
            [0.5, 4.0, -1.5, 1.25],
            [0.5, 4.0, -1.5, 1.25],
        ]
    )

    transformer = preprocessing.KBinsDiscretizer(
        3, encode="ordinal", strategy="quantile", subsample=None
    )
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X = emulator.seal(X)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(X)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def emul_kbinsdiscretizer_quantile_sample_weight():
    def kbinsdiscretize(X, sample_weight):
        transformer = KBinsDiscretizer(n_bins=3, strategy="quantile")
        transformed = transformer.fit_transform(
            X, sample_weight=sample_weight, remove_bin=True
        )
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]])
    sample_weight = jnp.array([1, 1, 3, 1])

    transformer = preprocessing.KBinsDiscretizer(
        3, encode="ordinal", strategy="quantile", subsample=None
    )
    transformer.fit(X, sample_weight=sample_weight)
    sk_transformed = transformer.transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X, sample_weight = emulator.seal(X, sample_weight)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(
        X, sample_weight
    )
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def emul_kbinsdiscretizer_quantile_sample_weight_diverse_n_bins():
    def kbinsdiscretize(X, n_bins, sample_weight):
        transformer = KBinsDiscretizer(
            n_bins=3, diverse_n_bins=n_bins, strategy="quantile"
        )
        transformed = transformer.fit_transform(
            X, sample_weight=sample_weight, remove_bin=True
        )
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2]])
    n_bins = jnp.array([2, 3, 3, 3])
    sample_weight = jnp.array([1, 1, 3, 1])

    transformer = preprocessing.KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
    )
    transformer.fit(X, sample_weight=sample_weight)
    sk_transformed = transformer.transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X, sample_weight, n_bins = emulator.seal(X, sample_weight, n_bins)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(
        X, n_bins, sample_weight
    )
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def emul_kbinsdiscretizer_quantile_sample_weight_diverse_n_bins2():
    def kbinsdiscretize(X, n_bins, sample_weight):
        transformer = KBinsDiscretizer(
            n_bins=4, diverse_n_bins=n_bins, strategy="quantile"
        )
        transformed = transformer.fit_transform(
            X, sample_weight=sample_weight, remove_bin=True
        )
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    n_bins = jnp.array([2, 4, 4, 4])
    sample_weight = jnp.array([1, 1, 3, 1])

    transformer = preprocessing.KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
    )
    transformer.fit(X, sample_weight=sample_weight)
    sk_transformed = transformer.transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X, sample_weight, n_bins = emulator.seal(X, sample_weight, n_bins)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(
        X, n_bins, sample_weight
    )
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def emul_kbinsdiscretizer_quantile_sample_weight_diverse_n_bins_no_vectorize():
    def kbinsdiscretize(X, sample_weight):
        transformer = KBinsDiscretizer(
            n_bins=3, diverse_n_bins=np.array([2, 3, 3, 3]), strategy="quantile"
        )
        transformed = transformer.fit_transform(
            X, vectorize=False, sample_weight=sample_weight, remove_bin=True
        )
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([2, 3, 3, 3])
    sample_weight = jnp.array([1, 1, 3, 1])

    transformer = preprocessing.KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
    )
    sk_transformed = transformer.fit_transform(X, sample_weight=sample_weight)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X, sample_weight = emulator.seal(X, sample_weight)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(
        X, sample_weight
    )
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-3)
    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-3
    )


def emul_kbinsdiscretizer_kmeans():
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(n_bins=4, strategy="kmeans")
        transformed = transformer.fit_transform(X)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array(
        [[-4, -4, -4, -4], [-3, -3, -3, -3], [-2, -2, -2, -2], [-1, -1, -1, -1]]
    )

    transformer = preprocessing.KBinsDiscretizer(
        4, encode="ordinal", strategy="kmeans", subsample=None
    )
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X = emulator.seal(X)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(X)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


def emul_kbinsdiscretizer_kmeans_diverse_n_bins_no_vectorize():
    def kbinsdiscretize(X):
        transformer = KBinsDiscretizer(
            n_bins=3, diverse_n_bins=np.array([2, 3, 3, 3]), strategy="kmeans"
        )
        transformed = transformer.fit_transform(X, vectorize=False, remove_bin=True)
        inv_transformed = transformer.inverse_transform(transformed)
        return transformed, inv_transformed

    X = jnp.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 2, 2]])
    n_bins = jnp.array([2, 3, 3, 3])

    transformer = preprocessing.KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="kmeans", subsample=None
    )
    sk_transformed = transformer.fit_transform(X)
    sk_inv_transformed = transformer.inverse_transform(sk_transformed)
    # print("sklearn:\n", sk_transformed)
    # print("sklearn:\n", sk_inv_transformed)

    X = emulator.seal(X)
    spu_transformed, spu_inv_transformed = emulator.run(kbinsdiscretize)(X)
    # print("result\n", spu_transformed)
    # print("result\n", spu_inv_transformed)

    ### The error here is larger than expected. If atol is 1e-4, there will be an error.
    np.testing.assert_allclose(sk_transformed, spu_transformed, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        sk_inv_transformed, spu_inv_transformed, rtol=0, atol=1e-4
    )


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
        emul_minmaxscaler()
        emul_minmaxscaler_partial_fit()
        emul_minmaxscaler_zero_variance()
        emul_maxabsscaler()
        emul_maxabsscaler_zero_maxabs()
        emul_kbinsdiscretizer_uniform()
        emul_kbinsdiscretizer_uniform_diverse_n_bins()
        emul_kbinsdiscretizer_uniform_diverse_n_bins_no_vectorize()
        emul_kbinsdiscretizer_quantile()
        emul_kbinsdiscretizer_quantile_diverse_n_bins()
        emul_kbinsdiscretizer_quantile_diverse_n_bins2()
        emul_kbinsdiscretizer_quantile_diverse_n_bins_no_vectorize()
        emul_kbinsdiscretizer_quantile_eliminate()
        emul_kbinsdiscretizer_quantile_sample_weight()
        emul_kbinsdiscretizer_quantile_sample_weight_diverse_n_bins()
        emul_kbinsdiscretizer_quantile_sample_weight_diverse_n_bins2()
        emul_kbinsdiscretizer_quantile_sample_weight_diverse_n_bins_no_vectorize()
        emul_kbinsdiscretizer_kmeans()
        emul_kbinsdiscretizer_kmeans_diverse_n_bins_no_vectorize()
        emul_onehotEncoder()
        emul_robustscaler()
    finally:
        emulator.down()
