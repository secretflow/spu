# Copyright 2023 Ant Group Co., Ltd.
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
import jax.numpy as jnp
from sklearn.linear_model import Ridge as skRidge

import examples.python.utils.dataset_utils as dsutil
import sml.utils.emulation as emulation
from sml.linear_model.ridge import Ridge


def emul_Ridge(mode: emulation.Mode.MULTIPROCESS):
    def proc(x1, x2, y, solver):
        model = Ridge(alpha=1.0, max_iter=100, solver=solver)
        x = jnp.concatenate((x1, x2), axis=1)
        y = y.reshape((y.shape[0], 1))
        result = model.fit(x, y).predict(x)
        return result

    def load_data():
        dataset_config = {
            "use_mock_data": False,
            "problem_type": "regression",
            "builtin_dataset_name": "diabetes",
            "left_slice_feature_ratio": 0.5,
        }

        x1, x2, y = dsutil.load_dataset_by_config(dataset_config)

        return x1, x2, y

    try:
        solver_list = ['cholesky', 'svd']
        print(f"solver_list={solver_list}")
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # load mock data
        x1, x2, y = load_data()

        for i in range(len(solver_list)):
            solver = solver_list[i]
            # mark these data to be protected in SPU
            X1, X2, Y = emulator.seal(x1, x2, y)

            # run
            result = emulator.run(proc, static_argnums=(3,))(X1, X2, Y, solver)
            print(f"[emul_{solver}_result]--------------------------------------------")
            print(result[:10])

            # sklearn test
            x = jnp.concatenate((x1, x2), axis=1)
            sklearn_result = (
                skRidge(alpha=1, solver=solver_list[i], fit_intercept=True)
                .fit(x, y)
                .predict(x)
            )
            print(f"[sklearn_{solver}_result]-----------------------------------------")
            print(sklearn_result[:10])

            # absolute_error
            print(f"[absolute_{solver}_error]-----------------------------------------")
            print(jnp.round(jnp.abs(result - sklearn_result)[:20], 5))
    finally:
        emulator.down()


if __name__ == "__main__":
    emul_Ridge(emulation.Mode.MULTIPROCESS)
