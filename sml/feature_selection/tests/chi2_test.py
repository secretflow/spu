# Copyright 2024 Ant Group Co., Ltd.
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
import unittest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2 as chi2_sklearn

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from sml.feature_selection.univariate_selection import chi2


class UnitTests(unittest.TestCase):
    def test_chi2(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM128
        )

        def proc(x, y, num_class, max_iter, compute_p_value):
            chi2_stats, p_value = chi2(x, y, num_class, max_iter, compute_p_value)
            return chi2_stats, p_value

        # create dataset
        x, y = load_iris(return_X_y=True)
        label_lst = np.unique(y)
        num_class = len(label_lst)
        max_iter = 1
        compute_p_value = True
        chi2_stats, p_value = spsim.sim_jax(sim, proc, static_argnums=(2, 3, 4))(
            x, y, num_class, max_iter, compute_p_value
        )
        sklearn_chi2_stats, sklearn_p_value = chi2_sklearn(x, y)
        print("Chi2 stats result:")
        print(chi2_stats)
        print("Sklearn chi2 stats result:")
        print(sklearn_chi2_stats)
        assert np.allclose(
            sklearn_chi2_stats,
            sklearn_chi2_stats,
            rtol=1.0e-5,
            atol=1.0e-2,
        )
        print("P value result:")
        print(p_value)
        print("Sklearn p value result:")
        print(sklearn_p_value)
        assert np.allclose(
            p_value, sklearn_p_value, rtol=1.0e-5, atol=1.0e-2, equal_nan=True
        )


if __name__ == "__main__":
    unittest.main()
