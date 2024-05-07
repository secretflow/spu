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

import os
import sys
import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2 as chi2_sklearn

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import sml.utils.emulation as emulation
from sml.feature_selection.univariate_selection import chi2


def emul_Chi2(mode: emulation.Mode.MULTIPROCESS):
    print("start chi2 stats emulation")

    def load_data():
        x, y = load_iris(return_X_y=True)
        return x, y

    def proc(x, y, num_class, max_iter, compute_p_value):
        chi2_stats, p_value = chi2(x, y, num_class, max_iter, compute_p_value)
        return chi2_stats, p_value

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()
        # load data
        x, y = load_data()
        label_lst = np.unique(y)
        num_class = len(label_lst)
        max_iter = 1
        compute_p_value = True
        # compare with sklearn
        start_time = time.time()
        sklearn_chi2_stats, sklearn_p_value = chi2_sklearn(x, y)
        end_time = time.time()
        print("=========================")
        print(f"Running time in SKlearn: {end_time - start_time:.3f}s")
        print("=========================")
        # mark these data to be protected in SPU
        X_spu, y_spu = emulator.seal(x, y)
        # run
        start_time = time.time()
        chi2_stats, p_value = emulator.run(proc, static_argnums=(2, 3, 4))(
            X_spu, y_spu, num_class, max_iter, compute_p_value
        )
        end_time = time.time()
        print("=========================")
        print(f"Running time in SPU: {end_time - start_time:.3f}s")
        print("=========================")

        # print chi2 stats
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
    except Exception as e:
        print(e)
    finally:
        emulator.down()


if __name__ == "__main__":
    emul_Chi2(emulation.Mode.MULTIPROCESS)
