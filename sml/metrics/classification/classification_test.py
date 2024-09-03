# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import unittest

import jax.numpy as jnp
import numpy as np
from sklearn import metrics

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# add ops dir to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from sklearn.metrics import average_precision_score as sk_average_precision_score
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

from sml.metrics.classification.classification import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    bin_counts,
    equal_obs,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)


class UnitTests(unittest.TestCase):

    def test_balanced_accuracy(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM128
        )

        def proc(y_true: jnp.ndarray, y_pred: jnp.ndarray, labels: jnp.ndarray):
            balanced_score = balanced_accuracy_score(y_true, y_pred, labels)
            return balanced_score

        def sklearn_proc(y_true, y_pred):
            balanced_score = metrics.balanced_accuracy_score(y_true, y_pred)
            return balanced_score

        def check(spu_result, sk_result):
            np.testing.assert_allclose(spu_result, sk_result, rtol=1, atol=1e-5)

        # Test binary
        y_true = jnp.array([0, 1, 1, 0, 1, 1])
        y_pred = jnp.array([0, 0, 1, 0, 1, 1])
        labels = jnp.array([0, 1])
        spu_result = spsim.sim_jax(sim, proc)(y_true, y_pred, labels)
        sk_result = sklearn_proc(y_true, y_pred)
        check(spu_result, sk_result)

        # Test multiclass
        y_true = jnp.array([0, 1, 1, 0, 2, 1])
        y_pred = jnp.array([0, 0, 1, 0, 2, 1])
        labels = jnp.array([0, 1, 2])
        spu_result = spsim.sim_jax(sim, proc)(y_true, y_pred, labels)
        sk_result = sklearn_proc(y_true, y_pred)
        check(spu_result, sk_result)

    def test_top_k_accuracy(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM128
        )

        def proc(
            y_true: jnp.ndarray,
            y_pred: jnp.ndarray,
            k,
            normalize,
            sample_weight,
            labels,
        ):
            top_k_score = top_k_accuracy_score(
                y_true,
                y_pred,
                k=k,
                normalize=normalize,
                sample_weight=sample_weight,
                labels=labels,
            )
            return top_k_score

        def sklearn_proc(y_true, y_pred, k, labels):
            top_k_score = metrics.top_k_accuracy_score(
                y_true, y_pred, k=k, labels=labels
            )
            return top_k_score

        def check(spu_result, sk_result):
            np.testing.assert_allclose(spu_result, sk_result, rtol=1, atol=1e-5)

        # Test multiclass
        y_true = jnp.array([0, 1, 2, 2, 0])
        y_score = jnp.array(
            [
                [0.8, 0.1, 0.1],
                [0.3, 0.4, 0.3],
                [0.1, 0.1, 0.8],
                [0.2, 0.2, 0.6],
                [0.7, 0.2, 0.1],
            ]
        )
        spu_result = spsim.sim_jax(sim, proc, static_argnums=(2, 3))(
            y_true, y_score, 2, True, None, None
        )
        sk_result = sklearn_proc(y_true, y_score, k=2, labels=None)
        check(spu_result, sk_result)

    def test_auc(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def bin_count(y_true, y_pred, bin_size):
            thresholds = equal_obs(y_pred, bin_size)
            return bin_counts(y_true, y_pred, thresholds)

        def thresholds(y_pred, bin_size):
            return equal_obs(y_pred, bin_size)

        def digitize(y_pred, thresholds):
            return jnp.digitize(y_pred, thresholds)

        row = 500
        y_true = np.random.randint(0, 2, (row,))
        y_pred = np.random.random((row,))

        bin_size = 4
        start = time.perf_counter()
        _ = spsim.sim_jax(sim, bin_count, static_argnums=(2,))(y_true, y_pred, bin_size)
        end = time.perf_counter()
        print("bin count takes time", end - start)

        start = time.perf_counter()
        thresholds_result = spsim.sim_jax(sim, thresholds, static_argnums=(1,))(
            y_pred, bin_size
        )
        end = time.perf_counter()
        print("thresholds takes time", end - start)
        start = time.perf_counter()
        _ = spsim.sim_jax(sim, digitize)(y_pred, thresholds_result)
        end = time.perf_counter()
        print("digitize count takes time", end - start)

        start = time.perf_counter()
        score = spsim.sim_jax(sim, roc_auc_score)(y_true, y_pred)
        end = time.perf_counter()
        print("auc takes time", end - start)
        true_score = sk_roc_auc_score(y_true, y_pred)

        np.testing.assert_almost_equal(true_score, score, decimal=2)

    def test_classification(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM128
        )

        def proc(
            y_true, y_pred, average='binary', labels=None, pos_label=1, transform=1
        ):
            f1 = f1_score(
                y_true,
                y_pred,
                average=average,
                labels=labels,
                pos_label=pos_label,
                transform=transform,
            )
            precision = precision_score(
                y_true,
                y_pred,
                average=average,
                labels=labels,
                pos_label=pos_label,
                transform=transform,
            )
            recall = recall_score(
                y_true,
                y_pred,
                average=average,
                labels=labels,
                pos_label=pos_label,
                transform=transform,
            )
            accuracy = accuracy_score(y_true, y_pred)
            return f1, precision, recall, accuracy

        def sklearn_proc(y_true, y_pred, average='binary', labels=None, pos_label=1):
            f1 = metrics.f1_score(
                y_true, y_pred, average=average, labels=labels, pos_label=pos_label
            )
            precision = metrics.precision_score(
                y_true, y_pred, average=average, labels=labels, pos_label=pos_label
            )
            recall = metrics.recall_score(
                y_true, y_pred, average=average, labels=labels, pos_label=pos_label
            )
            accuracy = metrics.accuracy_score(y_true, y_pred)
            return f1, precision, recall, accuracy

        def check(spu_result, sk_result):
            for pair in zip(spu_result, sk_result):
                np.testing.assert_allclose(pair[0], pair[1], rtol=1, atol=1e-5)

        # Test binary
        y_true = jnp.array([0, 1, 1, 0, 1, 1])
        y_pred = jnp.array([0, 0, 1, 0, 1, 1])
        spu_result = spsim.sim_jax(sim, proc, static_argnums=(2, 5))(
            y_true, y_pred, 'binary', None, 1, False
        )
        sk_result = sklearn_proc(y_true, y_pred)
        check(spu_result, sk_result)

        # Test multiclass
        y_true = jnp.array([0, 1, 1, 0, 2, 1])
        y_pred = jnp.array([0, 0, 1, 0, 2, 1])
        spu_result = spsim.sim_jax(sim, proc, static_argnums=(2, 5))(
            y_true, y_pred, None, [0, 1, 2], 1, True
        )
        sk_result = sklearn_proc(y_true, y_pred, average=None, labels=[0, 1, 2])
        check(spu_result, sk_result)

    def test_average_precision_score(self):
        sim = spsim.Simulator.simple(
            2, spu_pb2.ProtocolKind.SEMI2K, spu_pb2.FieldType.FM64
        )

        def proc(y_true, y_score, **kwargs):
            sk_res = sk_average_precision_score(y_true, y_score, **kwargs)
            spu_res = spsim.sim_jax(sim, average_precision_score)(
                y_true, y_score, **kwargs
            )
            return sk_res, spu_res

        def check(res1, res2):
            return np.testing.assert_allclose(res1, res2, rtol=1e-3, atol=1e-3)

        # --- Test binary classification ---
        # 0-1 labels, no tied value
        y_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([0.1, 0.4, 0.35, 0.8], dtype=jnp.float32)
        check(*proc(y_true, y_score))
        # 0-1 labels, with tied value, even length
        y_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([0.4, 0.4, 0.4, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score))
        # 0-1 labels, with tied value, odd length
        y_true = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([0.4, 0.4, 0.4, 0.25, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score))
        # customized labels
        y_true = jnp.array([2, 2, 3, 3], dtype=jnp.int32)
        y_score = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
        check(*proc(y_true, y_score, pos_label=3))
        # larger random dataset
        y_true = jnp.array(np.random.randint(0, 2, 100), dtype=jnp.int32)
        y_score = jnp.array(np.hstack((0, 1, np.random.random(98))), dtype=jnp.float32)
        check(*proc(y_true, y_score))
        # single label edge case
        y_true = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        y_score = jnp.array([0.4, 0.25, 0.4, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score))
        y_true = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([0.4, 0.25, 0.4, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score))
        # zero score edge case
        y_true = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([0, 0, 0, 0.25, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score))
        # score > 1 edge case
        y_true = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([1.5, 1.5, 1.5, 0.25, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score))

        # --- Test multiclass classification ---
        y_true = np.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
        y_score = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
                [0.4, 0.4, 0.2],
                [0.1, 0.2, 0.7],
            ],
            dtype=jnp.float32,
        )
        classes = jnp.unique(y_true)
        # test over three supported average options
        for average in ["macro", "micro", None]:
            sk_res = sk_average_precision_score(y_true, y_score, average=average)
            spu_res = spsim.sim_jax(sim, average_precision_score, static_argnums=(3,))(
                y_true, y_score, classes, average
            )
            check(sk_res, spu_res)


if __name__ == "__main__":
    unittest.main()
