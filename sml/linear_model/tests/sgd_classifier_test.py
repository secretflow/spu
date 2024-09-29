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

import json
import unittest

import jax.numpy as jnp

# TODO: unify this.
import examples.python.utils.dataset_utils as dsutil
import spu.spu_pb2 as spu_pb2  # type: ignore
import spu.utils.simulation as spsim
from sml.linear_model.sgd_classifier import SGDClassifier


class UnitTests(unittest.TestCase):
    def test_sgd(self):
        sim = spsim.Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, 64)

        def proc(x1, x2, y):
            model = SGDClassifier(
                epochs=1,
                learning_rate=0.1,
                batch_size=1024,
                reg_type='logistic',
                penalty='None',
                l2_norm=0.0,
            )

            x = jnp.concatenate((x1, x2), axis=1)
            y = y.reshape((y.shape[0], 1))

            return model.fit(x, y).predict_proba(x)

        DATASET_CONFIG_FILE = "examples/python/conf/ds_mock_regression_basic.json"
        with open(DATASET_CONFIG_FILE, "r") as f:
            dataset_config = json.load(f)

        x1, x2, y = dsutil.load_dataset_by_config(dataset_config)
        result = spsim.sim_jax(sim, proc)(x1, x2, y)
        print(result)


if __name__ == "__main__":
    unittest.main()
