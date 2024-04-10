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
import os
import unittest
import sys
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_iris

import spu.spu_pb2 as spu_pb2  # type: ignore
import spu.utils.simulation as spsim

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from sml.feature_selection.univariate_selection import chi2


class UnitTests(unittest.TestCase):
    def test_chi2(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        def proc(x, y, label_lst):
            chi2_stats, p_value = chi2(x, y, label_lst)
            return chi2_stats, p_value

        # create dataset
        x, y = load_iris(return_X_y=True)
        label_lst = np.unique(y)
        num_class = len(label_lst)
        y = np.eye(num_class)[y]
        result = spsim.sim_jax(sim, proc)(x, y, label_lst)
        print("result: ", result)
        from sklearn.feature_selection import chi2 as chi2_sklearn

        sklearn_result = chi2_sklearn(x, y)
        print("sklearn result:", sklearn_result)


if __name__ == "__main__":
    unittest.main()
