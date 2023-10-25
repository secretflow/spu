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

import os
import sys
import unittest

import jax
import jax.numpy as jnp
from sklearn.datasets import load_iris

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from sml.gaussian_process._gpc import GaussianProcessClassifier


class UnitTests(unittest.TestCase):
    def test_gpc(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM128
        )

        # Test GaussianProcessClassifier
        @jax.jit
        def proc(x, y, x_pred):
            model = GaussianProcessClassifier(max_iter_predict=10, n_classes_=3)
            model = model.fit(x, y)

            pred = model.predict(x_pred)
            return pred
            # return model

        # Create dataset
        x, y = load_iris(return_X_y=True)

        idx = list(range(45, 55)) + list(range(100, 105))
        prd_idx = list(range(0, 5)) + list(range(55, 60)) + list(range(110, 115))
        x_pred = x[prd_idx, :]
        y_pred = y[prd_idx]
        x = x[idx, :]
        y = y[idx]

        # Run
        result = spsim.sim_jax(sim, proc)(x, y, x_pred)
        # result = proc(x, y, x_pred)

        print(result)
        print(y_pred)
        print("Accuracy: ", jnp.sum(result == y_pred) / len(y_pred))


if __name__ == "__main__":
    unittest.main()
