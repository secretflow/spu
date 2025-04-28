# Copyright 2025 Ant Group Co., Ltd.
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
import time
import unittest

import numpy as np
from sklearn.feature_selection import f_classif as f_classif_sklearn

import spu.libspu as libspu
import spu.utils.simulation as spsim

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from sml.feature_selection.anova_f import f_classif


class AnovaFTest(unittest.TestCase):
    def test_anova_f(self):
        """Tests ANOVA F-statistic and p-value calculation using multi-feature API."""
        start = time.time()

        sim = spsim.Simulator.simple(
            3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM128
        )

        # Fixed-point precision parameters
        fxp_fraction_bits = 26
        epsilon = 2 ** (-fxp_fraction_bits)
        fpmin = 2 ** (-fxp_fraction_bits)

        def proc_multi(x_all_features, y_labels, k):
            f_stat, p_val = f_classif(
                x_all_features,
                y_labels,
                k,
                p_value_iter=100,
                epsilon=epsilon,
                fpmin=fpmin,
            )
            return f_stat, p_val

        np.random.seed(0)
        n_features = 20
        x = np.vstack(
            [
                np.random.normal(loc=0.0, scale=1.0, size=(10, n_features)),
                np.random.normal(loc=0.5, scale=1.0, size=(10, n_features)),
                np.random.normal(loc=1.0, scale=1.0, size=(10, n_features)),
            ]
        )
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        x = x.astype(np.float64)
        y = y.astype(np.int64)
        num_classes = len(np.unique(y))

        sklearn_f_stats, sklearn_p_values = f_classif_sklearn(x, y)

        spu_exec = spsim.sim_jax(sim, proc_multi, static_argnums=(2,))
        f_spu_res, p_spu_res = spu_exec(x, y, num_classes)

        rtol = 1e-1
        atol = 1e-1

        for idx in range(x.shape[1]):
            f_ref_val = sklearn_f_stats[idx]
            p_ref_val = sklearn_p_values[idx]
            f_res = f_spu_res[idx]
            p_res = p_spu_res[idx]

            print(f"\n--- Feature {idx} ---")
            print(f"SPU Result: F={f_res}, p={p_res}")
            print(f"SKL Result: F={f_ref_val}, p={p_ref_val}")

            if np.isinf(f_ref_val):
                self.assertTrue(
                    np.isinf(f_res) or f_res > 1e10,
                    f"Feature {idx}: F-stat mismatch (Ref: Inf, SPU: {f_res})",
                )
                self.assertAlmostEqual(
                    p_res,
                    0.0,
                    delta=atol,
                    msg=f"Feature {idx}: P-value mismatch (Ref: 0.0, SPU: {p_res})",
                )
            else:
                np.testing.assert_allclose(
                    f_res,
                    f_ref_val,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Feature {idx}: F-stat mismatch (Ref: {f_ref_val}, SPU: {f_res})",
                )
                try:
                    np.testing.assert_allclose(
                        p_res,
                        p_ref_val,
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"Feature {idx}: P-value mismatch (Ref: {p_ref_val}, SPU: {p_res})",
                    )
                except AssertionError as e:
                    if p_ref_val < atol and p_res < atol:
                        print(
                            f"Note: P-value comparison failed strict tolerance but both close to zero. Accepting."
                        )
                    else:
                        raise e

        end = time.time()
        print(f"\n[Time] test_anova_f took {end - start:.3f} seconds")
        print("\nANOVA F-test comparison completed.")


if __name__ == "__main__":
    unittest.main()
