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
    bin_counts,
    equal_obs,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class UnitTests(unittest.TestCase):
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
    
        def thresholds_digitize(y_pred, bin_size):
            thresholds = equal_obs(y_pred, bin_size)
            return jnp.digitize(y_pred, thresholds)

        row = 500
        y_true = np.random.randint(0, 2, (row,))
        y_pred = np.random.random((row,))

        bin_size = 4
        # start = time.perf_counter()
        copts = spu_pb2.CompilerOptions()
        
        spu_fn = spsim.sim_jax(sim, bin_count, static_argnums=(2,), copts=copts, pphlo_ref=(None, None)) #bin_count
        result = spu_fn(y_true, y_pred, bin_size)
        skip_1 = False
        try:
            if result == "skipped":
                skip_1 = True
        except:
            pass
        if not skip_1:
            print(spu_fn.pphlo)
            # end = time.perf_counter()
            # print("bin count takes time", end - start)

        # start = time.perf_counter()
        copts = spu_pb2.CompilerOptions()

        spu_fn = spsim.sim_jax(sim, thresholds_digitize, static_argnums=(1,), copts=copts, pphlo_ref=(None, None)) #thresholds_digitize
        result = spu_fn(y_pred, bin_size)
        skip_2 = False
        try:
            if result == "skipped":
                skip_2 = True
        except:
            pass
        if not skip_2:
            print(spu_fn.pphlo)
            # end = time.perf_counter()
            # print("thresholds and digitize takes time", end - start)

        # start = time.perf_counter()
        copts = spu_pb2.CompilerOptions()
        
        spu_fn = spsim.sim_jax(sim, roc_auc_score, copts=copts, pphlo_ref=(None, None)) #roc_auc_score
        score = spu_fn(y_true, y_pred)
        try:
            if score == "skipped":
                return True
        except:
            pass
        print(spu_fn.pphlo)
        # end = time.perf_counter()
        # print("auc takes time", end - start)
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
        copts = spu_pb2.CompilerOptions()
        
        proc.__name__ = "test_classification_binary"
        spu_fn = spsim.sim_jax(sim, proc, static_argnums=(2, 5), copts=copts, pphlo_ref=(None, None)) #test_classification_binary
        spu_result = spu_fn(
            y_true, y_pred, 'binary', None, 1, False
        )
        to_skip = False
        try:
            if spu_result == "skipped":
                to_skip = True
        except:
            pass
        if not to_skip:
            print(spu_fn.pphlo)
            sk_result = sklearn_proc(y_true, y_pred)
            check(spu_result, sk_result)

        # Test multiclass
        y_true = jnp.array([0, 1, 1, 0, 2, 1])
        y_pred = jnp.array([0, 0, 1, 0, 2, 1])
        copts = spu_pb2.CompilerOptions()
        
        proc.__name__ = "test_classification_multiclass"
        spu_fn = spsim.sim_jax(sim, proc, static_argnums=(2, 5), copts=copts, pphlo_ref=(None, None)) #test_classification_multiclass
        spu_result = spu_fn(
            y_true, y_pred, None, [0, 1, 2], 1, True
        )
        try:
            if spu_result == "skipped":
                return True
        except:
            pass
        print(spu_fn.pphlo)
        sk_result = sklearn_proc(y_true, y_pred, average=None, labels=[0, 1, 2])
        check(spu_result, sk_result)

    def test_average_precision_score(self):
        sim = spsim.Simulator.simple(
            2, spu_pb2.ProtocolKind.CHEETAH, spu_pb2.FieldType.FM64
        )

        def proc(y_true, y_score, test_i, pphlo_ref, skip = False, skip_list = None, not_skip_list = None, hlo_log = False, **kwargs):
            sk_res = sk_average_precision_score(y_true, y_score, **kwargs)
            copts = spu_pb2.CompilerOptions()
        
            average_precision_score.__name__ = f"average_precision_score_test{test_i}"
            spu_fn = spsim.sim_jax(sim, average_precision_score, copts=copts, pphlo_ref=pphlo_ref, skip=skip, skip_list=skip_list, not_skip_list=not_skip_list, hlo_log=hlo_log)
            spu_res = spu_fn(
                y_true, y_score, **kwargs
            )
            try:
                if spu_res == "skipped":
                    return 0, 0
            except:
                pass
            print(spu_fn.pphlo)
            return sk_res, spu_res

        def check(res1, res2):
            return np.testing.assert_allclose(res1, res2, rtol=1e-3, atol=1e-3)

        # --- Test binary classification ---
        # 0-1 labels, no tied value
        y_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([0.1, 0.4, 0.35, 0.8], dtype=jnp.float32)
        check(*proc(y_true, y_score, 0, (None, None))) #average_precision_score_test0
        # 0-1 labels, with tied value, even length
        y_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([0.4, 0.4, 0.4, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score, 1, (None, None))) #average_precision_score_test1
        # 0-1 labels, with tied value, odd length
        y_true = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([0.4, 0.4, 0.4, 0.25, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score, 2, (None, None))) #average_precision_score_test2
        # customized labels
        y_true = jnp.array([2, 2, 3, 3], dtype=jnp.int32)
        y_score = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
        check(*proc(y_true, y_score, 3, (None, None), pos_label=3)) #average_precision_score_test3
        # larger random dataset
        y_true = jnp.array(np.random.randint(0, 2, 100), dtype=jnp.int32)
        y_score = jnp.array(np.hstack((0, 1, np.random.random(98))), dtype=jnp.float32)
        check(*proc(y_true, y_score, 4, (None, None))) #average_precision_score_test4
        # single label edge case
        y_true = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        y_score = jnp.array([0.4, 0.25, 0.4, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score, 5, (None, None))) #average_precision_score_test5
        y_true = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([0.4, 0.25, 0.4, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score, 6, (None, None))) #average_precision_score_test6
        # zero score edge case
        y_true = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([0, 0, 0, 0.25, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score, 7, (None, None))) #average_precision_score_test7
        # score > 1 edge case
        y_true = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)
        y_score = jnp.array([1.5, 1.5, 1.5, 0.25, 0.25], dtype=jnp.float32)
        check(*proc(y_true, y_score, 8, (None, None))) #average_precision_score_test8

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
            copts = spu_pb2.CompilerOptions()
            
            if average == "macro":
                average_precision_score.__name__ = f"average_precision_score_testmacro"
                spu_fn = spsim.sim_jax(sim, average_precision_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #average_precision_score_testmacro
            elif average == "micro":
                average_precision_score.__name__ = f"average_precision_score_testmicro"
                spu_fn = spsim.sim_jax(sim, average_precision_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #average_precision_score_testmicro
            else:
                average_precision_score.__name__ = f"average_precision_score_testNone"
                spu_fn = spsim.sim_jax(sim, average_precision_score, static_argnums=(3,), copts=copts, pphlo_ref=(None, None)) #average_precision_score_testNone
            spu_res = spu_fn(
                y_true, y_score, classes, average
            )
            try:
                if spu_res == "skipped":
                    continue
            except:
                pass
            print(spu_fn.pphlo)
            check(sk_res, spu_res)


if __name__ == "__main__":
    unittest.main()
