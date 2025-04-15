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
import time
import unittest

import numpy as np

# Use sklearn for dataset loading and reference calculation
from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif as f_classif_sklearn

# Use libspu enums and spsim directly as in the chi2_test example
import spu
import spu.libspu as libspu
import spu.utils.simulation as spsim

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
# Import the JAX-based logic function
from sml.feature_selection.anova_f import f_classif_logic, f_classif_multi


class AnovaFTest(unittest.TestCase):
    def test_anova_f(self):
        """Tests ANOVA F-statistic and p-value calculation using iris dataset."""
        start = time.time()

        # Set up SPU Simulator
        sim = spsim.Simulator.simple(
            3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM128
        )

        # Define the wrapper function for the JAX logic
        def proc(x_feature, y_labels, k):
            f_stat, p_val = f_classif_logic(x_feature, y_labels, k)
            return f_stat, p_val

        # Load Dataset (Iris)
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

        # Calculate Sklearn Reference (per feature)
        sklearn_f_stats, sklearn_p_values = f_classif_sklearn(x, y)

        # Run SPU computation and compare (per feature)
        num_features = x.shape[1]
        for idx in range(num_features):
            x_feature_plain = x[:, idx : idx + 1]
            y_plain = y

            # Wrap 'proc' using sim_jax
            spu_exec = spsim.sim_jax(sim, proc, static_argnums=(2,))

            # Execute by passing NumPy arrays directly
            f_spu_res, p_spu_res = spu_exec(x_feature_plain, y_plain, num_classes)

            f_ref_val = sklearn_f_stats[idx]
            p_ref_val = sklearn_p_values[idx]

            print(f"\n--- Feature {idx} ---")
            print(f"SPU Result: F={f_spu_res[0]}, p={p_spu_res[0]}")
            print(f"SKL Result: F={f_ref_val}, p={p_ref_val}")

            rtol = 1e-1
            atol = 1e-1

            if np.isnan(f_ref_val):
                self.assertTrue(
                    np.isnan(f_spu_res[0]),
                    f"Feature {idx}: F-stat mismatch (Ref: NaN, SPU: {f_spu_res[0]})",
                )
                self.assertTrue(
                    np.isnan(p_spu_res[0]),
                    f"Feature {idx}: P-value mismatch (Ref: NaN, SPU: {p_spu_res[0]})",
                )
            elif np.isinf(f_ref_val):
                self.assertTrue(
                    np.isinf(f_spu_res[0]) or f_spu_res[0] > 1e10,
                    f"Feature {idx}: F-stat mismatch (Ref: Inf, SPU: {f_spu_res[0]})",
                )
                self.assertAlmostEqual(
                    p_spu_res[0],
                    0.0,
                    delta=atol,
                    msg=f"Feature {idx}: P-value mismatch (Ref: 0.0, SPU: {p_spu_res[0]})",
                )
            else:
                np.testing.assert_allclose(
                    f_spu_res[0],
                    f_ref_val,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Feature {idx}: F-stat mismatch (Ref: {f_ref_val}, SPU: {f_spu_res[0]})",
                )
                try:
                    np.testing.assert_allclose(
                        p_spu_res[0],
                        p_ref_val,
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"Feature {idx}: P-value mismatch (Ref: {p_ref_val}, SPU: {p_spu_res[0]})",
                    )
                except AssertionError as e:
                    if p_ref_val < atol and p_spu_res[0] < atol:
                        print(
                            f"Note: P-value comparison failed strict tolerance but both values are close to zero (Ref: {p_ref_val}, SPU: {p_spu_res[0]}). Accepting."
                        )
                    else:
                        raise e
        end = time.time()
        print(f"\n[Time] test_anova_f took {end - start:.3f} seconds")
        print("\nANOVA F-test comparison completed.")

    def test_anova_f_multi(self):
        """Tests ANOVA F-statistic and p-value calculation using multi-feature API."""
        start = time.time()

        sim = spsim.Simulator.simple(
            3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM128
        )

        def proc_multi(x_all_features, y_labels, k):
            f_stat, p_val = f_classif_multi(x_all_features, y_labels, k)
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

        # Sklearn reference
        sklearn_f_stats, sklearn_p_values = f_classif_sklearn(x, y)

        # Run all features at once via sim_jax
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

            if np.isnan(f_ref_val):
                self.assertTrue(
                    np.isnan(f_res),
                    f"Feature {idx}: F-stat mismatch (Ref: NaN, SPU: {f_res})",
                )
                self.assertTrue(
                    np.isnan(p_res),
                    f"Feature {idx}: P-value mismatch (Ref: NaN, SPU: {p_res})",
                )
            elif np.isinf(f_ref_val):
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
        print(f"\n[Time] test_anova_f_multi took {end - start:.3f} seconds")
        print("\nANOVA F-test (multi-column) comparison completed.")


if __name__ == "__main__":
    unittest.main()
