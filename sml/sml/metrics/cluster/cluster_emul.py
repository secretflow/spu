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

import os
import sys

import jax.numpy as jnp
import numpy as np
from sklearn.metrics import adjusted_rand_score as sk_adjusted_rand_score
from sklearn.metrics import rand_score as sk_rand_score

# add ops dir to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import sml.utils.emulation as emulation
from sml.metrics.cluster.cluster import adjusted_rand_score, rand_score


def emul_rand_score(mode: emulation.Mode.MULTIPROCESS):
    def proc(labels_true, labels_pred):
        sk_res = sk_rand_score(labels_true, labels_pred)
        n_classes = int(jnp.unique(labels_true).shape[0])
        n_clusters = int(jnp.unique(labels_pred).shape[0])
        spu_res = emulator.run(rand_score, static_argnums=(2, 3))(
            *emulator.seal(labels_true, labels_pred), n_classes, n_clusters
        )
        return sk_res, spu_res

    def check(res1, res2):
        return np.testing.assert_allclose(res1, res2, rtol=1e-3, atol=1e-3)

    # --- Test perfect match ---
    labels_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    labels_pred = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    check(*proc(labels_true, labels_pred))

    # --- Test another perfect match ---
    labels_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    labels_pred = jnp.array([2, 2, 0, 0], dtype=jnp.int32)
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


def emul_adjusted_rand_score(mode: emulation.Mode.MULTIPROCESS):
    def proc(labels_true, labels_pred):
        sk_res = sk_adjusted_rand_score(labels_true, labels_pred)
        n_classes = int(jnp.unique(labels_true).shape[0])
        n_clusters = int(jnp.unique(labels_pred).shape[0])
        spu_res = emulator.run(adjusted_rand_score, static_argnums=(2, 3))(
            *emulator.seal(labels_true, labels_pred), n_classes, n_clusters
        )
        return sk_res, spu_res

    def check(res1, res2):
        return np.testing.assert_allclose(res1, res2, rtol=1e-3, atol=1e-3)

    # --- Test perfect match ---
    labels_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    labels_pred = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    check(*proc(labels_true, labels_pred))

    # --- Test another perfect match ---
    labels_true = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    labels_pred = jnp.array([2, 2, 0, 0], dtype=jnp.int32)
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
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC,
            emulation.Mode.MULTIPROCESS,
            bandwidth=300,
            latency=20,
        )
        emulator.up()
        emul_rand_score(emulation.Mode.MULTIPROCESS)
        emul_adjusted_rand_score(emulation.Mode.MULTIPROCESS)
    finally:
        emulator.down()
