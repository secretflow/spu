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

import jax.numpy as jnp
import numpy as np
from sklearn import metrics

# add ops dir to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))


import sml.utils.emulation as emulation
from sml.metrics.regression.regression import (
    explained_variance_score,
    mean_squared_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    d2_tweedie_score,
)


def emul_Regression(mode: emulation.Mode.MULTIPROCESS):
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Test d2_tweedie_score
        y_true = jnp.array([0.5, 1, 2.5, 7])
        y_pred = jnp.array([1, 1, 5, 3.5])
        weight = None
        sk_result = metrics.d2_tweedie_score(y_true, y_pred, sample_weight=weight, power=1)
        spu_result = emulator.run(d2_tweedie_score, static_argnums=(3, ))(y_true, y_pred, weight, 1)
        np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)

        # Test explained_variance_score
        y_true = jnp.array([3, -0.5, 2, 7])
        y_pred = jnp.array([2.5, 0.0, 2, 8])
        weight = None
        sk_result = metrics.explained_variance_score(y_true, y_pred, sample_weight=weight, multioutput="variance_weighted", force_finite=True)
        spu_result = emulator.run(explained_variance_score, static_argnums=(3, ))(y_true, y_pred, weight, "variance_weighted")
        np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)

        # Test mean_squared_error
        y_true = jnp.array([3, -0.5, 2, 7])
        y_pred = jnp.array([2.5, 0.0, 2, 8])
        weight = None
        sk_result = metrics.mean_squared_error(y_true, y_pred, sample_weight=None, squared=False)
        spu_result = emulator.run(mean_squared_error, static_argnums=(3, 4))(y_true, y_pred, weight, "uniform_average", False)
        np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)

        # Test mean_poisson_deviance
        y_true = jnp.array([2, 0, 1, 4])
        y_pred = jnp.array([0.5, 0.5, 2., 2.])
        weight = None
        sk_result = metrics.mean_poisson_deviance(y_true, y_pred, sample_weight=weight)
        spu_result = emulator.run(mean_poisson_deviance)(y_true, y_pred, weight)
        np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)

        # Test mean_gamma_deviance
        y_true = jnp.array([2, 0.5, 1, 4])
        y_pred = jnp.array([0.5, 0.5, 2., 2.])
        weight = None
        sk_result = metrics.mean_gamma_deviance(y_true, y_pred, sample_weight=weight)
        spu_result = emulator.run(mean_gamma_deviance)(y_true, y_pred, weight)
        np.testing.assert_allclose(sk_result, spu_result, rtol=0, atol=1e-4)

    finally:
        emulator.down()

if __name__ == "__main__":
    emul_Regression(emulation.Mode.MULTIPROCESS)
