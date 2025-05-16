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
import unittest

import jax.numpy as jnp
from sklearn.linear_model import Ridge as skRidge

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.linear_model.ridge import Ridge
from sml.utils.dataset_utils import load_open_source_datasets


class UnitTests(unittest.TestCase):
    def test_ridge(self):
        solver_list = ['cholesky', 'svd']
        print(f"solver_list={solver_list}")

        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

        def proc(x, y, solver):
            model = Ridge(alpha=1.0, max_iter=100, solver=solver)
            y = y.reshape((y.shape[0], 1))
            result = model.fit(x, y).predict(x)
            return result

        x, y = load_open_source_datasets(name="diabetes", need_split_train_test=False)

        for i in range(len(solver_list)):
            solver = solver_list[i]
            result = spsim.sim_jax(sim, proc, static_argnums=(2,))(x, y, solver)

            print(f"[spsim_{solver}_result]-------------------------------------------")
            print(result[:10])

            # sklearn test
            sklearn_result = (
                skRidge(alpha=1, solver=solver, fit_intercept=True).fit(x, y).predict(x)
            )
            print(f"[sklearn_{solver}_result]-----------------------------------------")
            print(sklearn_result[:10])

            # absolute_error
            print(f"[absolute_{solver}_error]-----------------------------------------")
            print(jnp.round(jnp.abs(result - sklearn_result)[:20], 5))


if __name__ == "__main__":
    unittest.main()
