# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import jax.numpy as jnp
import numpy as np
from sklearn.metrics import adjusted_rand_score as sk_adjusted_rand_score
from sklearn.metrics import rand_score as sk_rand_score

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.metrics.cluster.cluster import adjusted_rand_score, rand_score


class UnitTests(unittest.TestCase):
    def test_Cluster(self):
        sim = spsim.Simulator.simple(
            2, libspu.ProtocolKind.SEMI2K, libspu.FieldType.FM64
        )

        def proc(labels_true, labels_pred):
            sk_ri = sk_rand_score(labels_true, labels_pred)
            sk_ari = sk_adjusted_rand_score(labels_true, labels_pred)
            n_classes = int(jnp.unique(labels_true).shape[0])
            n_clusters = int(jnp.unique(labels_pred).shape[0])
            spu_ri = spsim.sim_jax(sim, rand_score, static_argnums=(2, 3))(
                labels_true, labels_pred, n_classes, n_clusters
            )
            spu_ari = spsim.sim_jax(sim, adjusted_rand_score, static_argnums=(2, 3))(
                labels_true, labels_pred, n_classes, n_clusters
            )
            return (sk_ri, sk_ari), (spu_ri, spu_ari)

        def check(spu_result, sk_result):
            for pair in zip(spu_result, sk_result):
                np.testing.assert_allclose(pair[0], pair[1], rtol=1e-3, atol=1e-3)

        # --- Test perfect match ---
        labels_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        labels_pred = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        check(*proc(labels_true, labels_pred))

        # --- Test another perfect match ---
        labels_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        labels_pred = jnp.array([2, 2, 0, 0], dtype=jnp.int32)
        check(*proc(labels_true, labels_pred))

        # --- Test total mismatch ---
        labels_true = jnp.array([0, 1, 2], dtype=jnp.int32)
        labels_pred = jnp.array([0, 0, 0], dtype=jnp.int32)
        check(*proc(labels_true, labels_pred))

        labels_true = jnp.array([0, 1], dtype=jnp.int32)
        labels_pred = jnp.array([0, 0], dtype=jnp.int32)
        check(*proc(labels_true, labels_pred))

        # --- Test partial match ---
        labels_true = jnp.array([0, 0, 1, 2], dtype=jnp.int32)
        labels_pred = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        check(*proc(labels_true, labels_pred))

        # --- Test with more than 2 clusters ---
        labels_true = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        labels_pred = jnp.array([1, 2, 3, 0], dtype=jnp.int32)
        check(*proc(labels_true, labels_pred))

        labels_true = jnp.array([0, 0, 1, 2], dtype=jnp.int32)
        labels_pred = jnp.array([2, 0, 2, 5], dtype=jnp.int32)
        check(*proc(labels_true, labels_pred))


if __name__ == "__main__":
    unittest.main()
