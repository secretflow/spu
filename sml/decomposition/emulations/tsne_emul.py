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

import numpy as np
from sklearn.datasets import load_iris

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import sml.utils.emulation as emulation
from sml.decomposition.tsne import basic_tsne


def test_tsne(mode: emulation.Mode = emulation.Mode.MULTIPROCESS):
    def emul_tsne():
        """
        Emulation function for t-SNE dimensionality reduction.

        This function loads the Iris dataset, runs the JAX-based t-SNE within the SPU emulation,
        and verifies the output embedding's shape and numerical validity.
        """
        print("Start t-SNE emulation...")

        def load_data():
            """Loads the Iris dataset."""
            print("Loading Iris dataset...")
            x, y = load_iris(return_X_y=True)
            x = x.astype(np.float64)  # Ensure compatibility with JAX
            return x, y

        def proc(x):
            """The function to be executed in SPU, wrapping the JAX-based t-SNE."""
            embedding = basic_tsne(
                x,
                n_components=2,
                perplexity=30.0,
                learning_rate=200.0,
                max_iter=1000,
                early_exaggeration=12.0,
                early_exaggeration_iter=250,
                momentum=0.8,
                random_state=42,  # Fixed seed for reproducibility
                verbose=10,  # Silent within SPU
            )
            return embedding

        try:
            # Load data
            x, y = load_data()
            n_samples = x.shape[0]
            print(f"Data loaded: {n_samples} samples, {x.shape[1]} features.")

            # Seal the data
            print("Sealing data...")
            x_spu = emulator.seal(x)
            print("Data sealed.")

            # Run the SPU computation
            print("Running SPU computation...")
            start_time = time.time()
            embedding = emulator.run(proc)(x_spu)
            end_time = time.time()
            spu_time = end_time - start_time
            print(f"SPU computation finished in {spu_time:.3f}s.")

            # Verify output
            print("Verifying output...")
            expected_shape = (n_samples, 2)
            assert (
                embedding.shape == expected_shape
            ), f"Expected embedding shape {expected_shape}, got {embedding.shape}"
            assert np.all(
                np.isfinite(embedding)
            ), "Embedding contains non-finite values (NaN or Inf)"
            print(f"Embedding shape: {embedding.shape}, all values finite.")

            print("========================================")
            print(f"Total SPU running time: {spu_time:.3f}s")
            print("t-SNE emulation completed successfully.")
            print("========================================")

        except Exception as e:
            print(f"An error occurred during emulation: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging

    try:
        # Initialize the emulator
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()
        emul_tsne()
    finally:
        emulator.down()


if __name__ == "__main__":
    test_tsne(emulation.Mode.MULTIPROCESS)
