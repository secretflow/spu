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

import jax.random as random
import numpy as np
from scipy.spatial import procrustes
from sklearn.datasets import load_digits, load_iris, make_blobs
from sklearn.manifold import TSNE as SklearnTSNE
from sklearn.manifold import trustworthiness

import spu.libspu as libspu
import spu.utils.simulation as spsim

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from sml.decomposition.tsne import TSNE


class TestTSNEComparison(unittest.TestCase):
    def test_tsne_similarity_init_pca(self):
        print("\nStarting t-SNE comparison test (PCA initialization)...")

        def proc_transform(
            X_in,
            n_components_in,
            perplexity_in,
            max_iter_in,
            spu_early_exaggeration_iter,
        ):

            model = TSNE(
                n_components=n_components_in,
                perplexity=perplexity_in,
                max_iter=max_iter_in,
                early_exaggeration_iter=spu_early_exaggeration_iter,
                init='pca',
                max_attempts=20,
                sigma_maxs=1e6,
                sigma_mins=1e-6,
            )
            Y_spu_out = model.fit_transform(X_in)
            kl_spu_out = model.kl_divergence_
            return Y_spu_out, kl_spu_out

        X, y = make_blobs(n_samples=50, n_features=4, centers=3, random_state=42)
        n_samples = X.shape[0]
        print(f"Loaded dataset with {n_samples} samples.")

        n_components = 2
        perplexity = 10
        max_iter = 300
        spu_max_iter = 50  # Set to 50 for faster unit test execution; to achieve better results, it needs to be set to 150 or above.
        spu_early_exaggeration_iter = 25
        random_state = 42
        n_neighbors_trustworthiness = 15

        print(
            f"Parameters: n_components={n_components}, perplexity={perplexity}, max_iter={max_iter}, random_state={random_state}"
        )

        print("Running scikit-learn t-SNE...")
        start_time_sklearn = time.time()
        sklearn_tsne = SklearnTSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
            init='pca',
        )

        Y_sklearn = sklearn_tsne.fit_transform(X.astype(np.float64))
        end_time_sklearn = time.time()

        kl_sklearn = sklearn_tsne.kl_divergence_
        print(
            f"Scikit-learn t-SNE finished in {end_time_sklearn - start_time_sklearn:.2f} seconds."
        )
        print(f"Scikit-learn final KL divergence: {kl_sklearn:.4f}")

        # --- Run SPU t-SNE ---
        print("Running SPU t-SNE...")
        start_time_spu = time.time()

        config = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.ABY3,
            field=libspu.FieldType.FM64,
            fxp_fraction_bits=18,
        )
        config.fxp_exp_mode = libspu.RuntimeConfig.ExpMode.EXP_PADE

        sim = spsim.Simulator(3, config)

        X_spu_input = X.astype(np.float32)
        Y_spu, kl_spu = spsim.sim_jax(sim, proc_transform, static_argnums=(1, 2, 3, 4))(
            X_spu_input,
            n_components,
            perplexity,
            spu_max_iter,
            spu_early_exaggeration_iter,
        )

        Y_spu = np.array(Y_spu)
        kl_spu = float(kl_spu)

        end_time_spu = time.time()
        print(f"SPU t-SNE finished in {end_time_spu - start_time_spu:.2f} seconds.")
        print(f"SPU final KL divergence: {kl_spu:.4f}")

        self.assertEqual(Y_sklearn.shape, Y_spu.shape)
        print(f"\nOutput shapes are consistent: {Y_sklearn.shape}")

        # --- Trustworthiness Analysis ---
        print(
            f"\nPerforming Trustworthiness Analysis (n_neighbors={n_neighbors_trustworthiness})..."
        )
        trust_sklearn = 0.0
        trust_spu = 0.0
        try:
            trust_sklearn = trustworthiness(
                X, Y_sklearn, n_neighbors=n_neighbors_trustworthiness
            )
            print(f"Trustworthiness (Scikit-learn): {trust_sklearn:.4f}")
        except ValueError as e:
            print(f"Could not calculate trustworthiness for Scikit-learn: {e}")

        try:
            trust_spu = trustworthiness(
                X, Y_spu, n_neighbors=n_neighbors_trustworthiness
            )
            print(f"Trustworthiness (SPU): {trust_spu:.4f}")
        except ValueError as e:
            print(f"Could not calculate trustworthiness for SPU: {e}")

        # --- Procrustes Analysis ---
        print("\nPerforming Procrustes Analysis...")

        try:
            mtx1, mtx2, disparity_val = procrustes(
                Y_sklearn.astype(np.float64), Y_spu.astype(np.float64)
            )
            print(f"Procrustes Disparity: {disparity_val:.4f}")
        except ValueError as e:
            print(f"Could not perform Procrustes analysis: {e}")
            print(f"Sklearn shape: {Y_sklearn.shape}, dtype: {Y_sklearn.dtype}")
            print(f"SPU shape: {Y_spu.shape}, dtype: {Y_spu.dtype}")

        print("\n--- Comparison Summary ---")
        print(f"Metric              | Scikit-learn | SPU")
        print(f"--------------------|--------------|------")
        print(f"KL Divergence       | {kl_sklearn:^12.4f} | {kl_spu:^4.4f}")
        print(f"Trustworthiness     | {trust_sklearn:^12.4f} | {trust_spu:^4.4f}")
        print(
            f"Execution Time (s)  | {end_time_sklearn - start_time_sklearn:^12.2f} | {end_time_spu - start_time_spu:^4.2f}"
        )
        print("\nTest finished.")

    def test_tsne_similarity_init_random(self):
        print("\nStarting t-SNE comparison test (random initialization)...")

        def proc_transform_random(
            X_in,
            Y_init_in,
            n_components_in,
            perplexity_in,
            max_iter_in,
            spu_early_exaggeration_iter,
        ):

            model = TSNE(
                n_components=n_components_in,
                perplexity=perplexity_in,
                max_iter=max_iter_in,
                early_exaggeration_iter=spu_early_exaggeration_iter,
                init='random',
                max_attempts=20,
                sigma_maxs=1e6,
                sigma_mins=1e-6,
            )
            Y_spu_out = model.fit_transform(X_in, Y_init=Y_init_in)
            kl_spu_out = model.kl_divergence_
            return Y_spu_out, kl_spu_out

        X, y = make_blobs(n_samples=50, n_features=4, centers=3, random_state=42)
        n_samples = X.shape[0]
        print(f"Loaded dataset with {n_samples} samples.")

        n_components = 2
        perplexity = 10
        max_iter = 300
        spu_max_iter = 50  # Set to 50 for faster unit test execution; to achieve better results, it needs to be set to 150 or above.
        spu_early_exaggeration_iter = 25
        random_state = 42
        n_neighbors_trustworthiness = 15

        print(
            f"Parameters: n_components={n_components}, perplexity={perplexity}, max_iter={max_iter}, random_state={random_state}"
        )

        print("Running scikit-learn t-SNE...")
        start_time_sklearn = time.time()
        sklearn_tsne = SklearnTSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
            init='random',
        )

        Y_sklearn = sklearn_tsne.fit_transform(X.astype(np.float64))
        end_time_sklearn = time.time()

        kl_sklearn = sklearn_tsne.kl_divergence_
        print(
            f"Scikit-learn t-SNE finished in {end_time_sklearn - start_time_sklearn:.2f} seconds."
        )
        print(f"Scikit-learn final KL divergence: {kl_sklearn:.4f}")

        # --- Run SPU t-SNE ---
        print("Running SPU t-SNE...")
        start_time_spu = time.time()

        X_spu_input = X.astype(np.float32)

        key = random.PRNGKey(random_state)

        Y_init_jax = 1e-4 * random.normal(key, (n_samples, n_components)).astype(
            np.float32
        )

        config = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.ABY3,
            field=libspu.FieldType.FM64,
            fxp_fraction_bits=18,
        )
        config.fxp_exp_mode = libspu.RuntimeConfig.ExpMode.EXP_PADE

        sim = spsim.Simulator(3, config)

        Y_spu, kl_spu = spsim.sim_jax(
            sim, proc_transform_random, static_argnums=(2, 3, 4, 5)
        )(
            X_spu_input,
            Y_init_jax,
            n_components,
            perplexity,
            spu_max_iter,
            spu_early_exaggeration_iter,
        )

        Y_spu = np.array(Y_spu)
        kl_spu = float(kl_spu)
        end_time_spu = time.time()
        print(f"SPU t-SNE finished in {end_time_spu - start_time_spu:.2f} seconds.")
        print(f"SPU final KL divergence: {kl_spu:.4f}")

        self.assertEqual(Y_sklearn.shape, Y_spu.shape)
        print(f"\nOutput shapes are consistent: {Y_sklearn.shape}")

        # --- Trustworthiness Analysis ---
        print(
            f"\nPerforming Trustworthiness Analysis (n_neighbors={n_neighbors_trustworthiness})..."
        )
        trust_sklearn = 0.0
        trust_spu = 0.0
        try:
            trust_sklearn = trustworthiness(
                X, Y_sklearn, n_neighbors=n_neighbors_trustworthiness
            )
            print(f"Trustworthiness (Scikit-learn): {trust_sklearn:.4f}")
        except ValueError as e:
            print(f"Could not calculate trustworthiness for Scikit-learn: {e}")

        try:
            trust_spu = trustworthiness(
                X, Y_spu, n_neighbors=n_neighbors_trustworthiness
            )
            print(f"Trustworthiness (SPU): {trust_spu:.4f}")
        except ValueError as e:
            print(f"Could not calculate trustworthiness for SPU: {e}")

        # --- Procrustes Analysis ---
        print("\nPerforming Procrustes Analysis...")

        try:
            mtx1, mtx2, disparity_val = procrustes(
                Y_sklearn.astype(np.float64), Y_spu.astype(np.float64)
            )
            print(f"Procrustes Disparity: {disparity_val:.4f}")
        except ValueError as e:
            print(f"Could not perform Procrustes analysis: {e}")
            print(f"Sklearn shape: {Y_sklearn.shape}, dtype: {Y_sklearn.dtype}")
            print(f"SPU shape: {Y_spu.shape}, dtype: {Y_spu.dtype}")

        print("\n--- Comparison Summary ---")
        print(f"Metric              | Scikit-learn | SPU")
        print(f"--------------------|--------------|------")
        print(f"KL Divergence       | {kl_sklearn:^12.4f} | {kl_spu:^4.4f}")
        print(f"Trustworthiness     | {trust_sklearn:^12.4f} | {trust_spu:^4.4f}")
        print(
            f"Execution Time (s)  | {end_time_sklearn - start_time_sklearn:^12.2f} | {end_time_spu - start_time_spu:^4.2f}"
        )
        print("\nTest finished.")


if __name__ == "__main__":
    unittest.main()
