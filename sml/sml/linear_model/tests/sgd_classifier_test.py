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


import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.linear_model.sgd_classifier import SGDClassifier
from sml.utils.dataset_utils import load_mock_datasets


class UnitTests(unittest.TestCase):
    def test_sgd(self):
        sim = spsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)

        def proc(x, y):
            model = SGDClassifier(
                epochs=1,
                learning_rate=0.1,
                batch_size=1024,
                reg_type='logistic',
                penalty='None',
                l2_norm=0.0,
            )

            y = y.reshape((y.shape[0], 1))

            return model.fit(x, y).predict_proba(x)

        x, y = load_mock_datasets(
            n_samples=50000,
            n_features=100,
            task_type="bi_classification",
            need_split_train_test=False,
        )
        result = spsim.sim_jax(sim, proc)(x, y)
        print(result)


if __name__ == "__main__":
    unittest.main()
