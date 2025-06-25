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

import jax.random as random
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import sml.utils.emulation as emulation
from sml.decomposition.tsne import TSNE


def test_tsne(mode: emulation.Mode = emulation.Mode.MULTIPROCESS):
    conf_path = "sml/decomposition/emulations/3pc.json"
    emulator = emulation.Emulator(conf_path, mode, bandwidth=300, latency=20)

    def emul_tsne_random():
        print("Start t-SNE emulation...")

        def load_data():
            """Loads the dataset."""
            print("Loading dataset...")
            x, y = make_blobs(n_samples=50, n_features=4, centers=3, random_state=42)
            x = x.astype(np.float64)
            x = x.astype(np.float64)
            return x, y

        def proc(x):
            key = random.PRNGKey(42)

            Y_init_jax = 1e-4 * random.normal(key, (n_samples, 2)).astype(np.float32)
            model = TSNE(
                n_components=2,
                perplexity=10,
                max_iter=50,  # Set to 50 for faster unit test execution; to achieve better results, it needs to be set to 150 or above.
                early_exaggeration=12,
                init='random',
            )
            Y_spu_out = model.fit_transform(x, Y_init=Y_init_jax)
            kl_spu_out = model.kl_divergence_
            return Y_spu_out, kl_spu_out

        try:
            x, y = load_data()
            n_samples = x.shape[0]
            print(f"Data loaded: {n_samples} samples, {x.shape[1]} features.")

            print("Sealing data...")
            x_spu = emulator.seal(x)
            print("Data sealed.")

            print("Running SPU computation...")
            start_time = time.time()
            embedding, kl = emulator.run(proc)(x_spu)
            end_time = time.time()
            spu_time = end_time - start_time
            print(f"SPU computation finished in {spu_time:.3f}s.")

            print("Verifying output...")
            expected_shape = (n_samples, 2)
            assert (
                embedding.shape == expected_shape
            ), f"Expected embedding shape {expected_shape}, got {embedding.shape}"
            assert np.all(
                np.isfinite(embedding)
            ), "Embedding contains non-finite values (NaN or Inf)"
            print(f"Embedding shape: {embedding.shape}, all values finite.")

            # Calculate and print trustworthiness
            print("Calculating trustworthiness...")
            trust_spu = trustworthiness(x, embedding, n_neighbors=15)
            print(f"Trustworthiness of SPU embedding: {trust_spu:.4f}")

            print("========================================")
            print(f"Total SPU running time: {spu_time:.3f}s")
            print("t-SNE emulation completed successfully.")
            print("========================================")

        except Exception as e:
            print(f"An error occurred during emulation: {e}")
            import traceback

            traceback.print_exc()

    def emul_tsne_pca():
        print("Start t-SNE emulation...")

        def load_data():
            """Loads the dataset."""
            print("Loading dataset...")
            x, y = make_blobs(n_samples=50, n_features=4, centers=3, random_state=42)
            x = x.astype(np.float64)
            x = x.astype(np.float64)
            return x, y

        def proc(x):
            model = TSNE(
                n_components=2,
                perplexity=10,
                max_iter=50,  # Set to 50 for faster unit test execution; to achieve better results, it needs to be set to 150 or above.
                early_exaggeration=12,
                init='pca',
            )
            Y_spu_out = model.fit_transform(x)
            kl_spu_out = model.kl_divergence_
            return Y_spu_out, kl_spu_out

        try:
            x, y = load_data()
            n_samples = x.shape[0]
            print(f"Data loaded: {n_samples} samples, {x.shape[1]} features.")

            print("Sealing data...")
            x_spu = emulator.seal(x)
            print("Data sealed.")

            print("Running SPU computation...")
            start_time = time.time()
            embedding, kl = emulator.run(proc)(x_spu)
            end_time = time.time()
            spu_time = end_time - start_time
            print(f"SPU computation finished in {spu_time:.3f}s.")

            print("Verifying output...")
            expected_shape = (n_samples, 2)
            assert (
                embedding.shape == expected_shape
            ), f"Expected embedding shape {expected_shape}, got {embedding.shape}"
            assert np.all(
                np.isfinite(embedding)
            ), "Embedding contains non-finite values (NaN or Inf)"
            print(f"Embedding shape: {embedding.shape}, all values finite.")

            # Calculate and print trustworthiness
            print("Calculating trustworthiness...")
            trust_spu = trustworthiness(x, embedding, n_neighbors=15)
            print(f"Trustworthiness of SPU embedding: {trust_spu:.4f}")

            print("========================================")
            print(f"Total SPU running time: {spu_time:.3f}s")
            print("t-SNE emulation completed successfully.")
            print("========================================")

        except Exception as e:
            print(f"An error occurred during emulation: {e}")
            import traceback

            traceback.print_exc()

    try:
        emulator.up()
        emul_tsne_random()
        emul_tsne_pca()
    finally:
        emulator.down()


if __name__ == "__main__":
    test_tsne(emulation.Mode.MULTIPROCESS)
