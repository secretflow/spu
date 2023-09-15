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
# See the License for the specifi

import os
import sys
import time
import unittest

import numpy as np
import jax.numpy as jnp

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# add ops dir to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from sml.metrics.classification.classification import (
    roc_auc_score,
    equal_obs,
    bin_counts,
)

from sklearn.metrics import roc_auc_score as sk_roc_auc_score


class UnitTests(unittest.TestCase):
    def test_simple(self):
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


if __name__ == "__main__":
    unittest.main()
