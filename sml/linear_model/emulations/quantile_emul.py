# Copyright 2024 Ant Group Co., Ltd.
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
import time

import jax.numpy as jnp
from sklearn.linear_model import QuantileRegressor as SklearnQuantileRegressor

import sml.utils.emulation as emulation
from sml.linear_model.quantile import QuantileRegressor as SmlQuantileRegressor

CONFIG_FILE = emulation.CLUSTER_ABY3_3PC


def emul_quantile(mode=emulation.Mode.MULTIPROCESS):
    def proc_wrapper(
        quantile,
        alpha,
        fit_intercept,
        lr,
        max_iter,
    ):
        quantile_custom = SmlQuantileRegressor(
            quantile=quantile,
            alpha=alpha,
            fit_intercept=fit_intercept,
            lr=lr,
            max_iter=max_iter,
        )

        def proc(X, y):
            quantile_custom_fit = quantile_custom.fit(X, y)
            result = quantile_custom_fit.predict(X)
            return result

        return proc

    def generate_data():
        import numpy as np

        np.random.seed(42)
        X = np.random.normal(size=(100, 2))
        y = 5 * X[:, 0] + 2 * X[:, 1] + np.random.normal(size=(100,)) * 0.1
        return X, y

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(CONFIG_FILE, mode, bandwidth=300, latency=20)
        emulator.up()

        # load mock data
        X, y = generate_data()

        # compare with sklearn
        quantile_sklearn = SklearnQuantileRegressor(
            quantile=0.7, alpha=0.1, fit_intercept=True, solver='highs'
        )
        start = time.time()
        quantile_sklearn_fit = quantile_sklearn.fit(X, y)
        score_plain = jnp.mean(y <= quantile_sklearn_fit.predict(X))
        end = time.time()
        print(f"Running time in SKlearn: {end - start:.2f}s")

        # mark these data to be protected in SPU
        X_spu, y_spu = emulator.seal(X, y)

        # run
        proc = proc_wrapper(
            quantile=0.7, alpha=0.1, fit_intercept=True, lr=0.01, max_iter=10
        )
        start = time.time()
        result = emulator.run(proc)(X_spu, y_spu)
        end = time.time()
        score_encrpted = jnp.mean(y <= result)
        print(f"Running time in SPU: {end - start:.2f}s")

        # print acc
        print(f"Accuracy in SKlearn: {score_plain:.2f}")
        print(f"Accuracy in SPU: {score_encrpted:.2f}")

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_quantile(emulation.Mode.MULTIPROCESS)
