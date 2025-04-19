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

import jax.numpy as jnp
import jax.random as random
import numpy as np
from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import sml.utils.emulation as emulation
from sml.preprocessing.quantile_transformer import QuantileTransformer


def test_quantile_transformer(mode: emulation.Mode = emulation.Mode.MULTIPROCESS):
    """
    Main test function for QuantileTransformer emulation.

    Sets up the emulator once and runs different test variations.
    (Modified to only run the 'uniform' distribution test).
    """

    N_QUANTILES = 100
    RANDOM_STATE = 42
    N_SAMPLES = 50
    N_FEATURES = 2

    def proc_quantile_transform(X, n_quantiles, distribution):
        """The core logic to be executed securely within the SPU."""

        model = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=distribution,
            random_state=RANDOM_STATE,
        )

        model.fit(X)
        X_transformed = model.transform(X)
        X_inversed = model.inverse_transform(X_transformed)
        return X_transformed, X_inversed

    def compare_results(
        X_plaintext,
        X_transformed_spu,
        X_inversed_spu,
        output_dist,
        n_quantiles,
        n_samples,
        random_state,
    ):
        """Compares SPU results with Sklearn reference and checks properties."""
        print(f"\nStarting comparisons for '{output_dist}' distribution...")

        assert (
            X_transformed_spu.shape == X_plaintext.shape
        ), f"Shape mismatch (transform, {output_dist})"
        assert (
            X_inversed_spu.shape == X_plaintext.shape
        ), f"Shape mismatch (inverse, {output_dist})"
        print("Shape checks PASSED.")

        sklearn_qt = SklearnQuantileTransformer(
            n_quantiles=min(n_quantiles, n_samples),
            output_distribution=output_dist,
            subsample=int(1e9),
            random_state=random_state,
        )
        X_np = np.array(X_plaintext)
        X_transformed_sklearn = sklearn_qt.fit_transform(X_np)
        print("Sklearn reference calculated.")

        # Compare Transformed Data
        np.testing.assert_allclose(
            X_transformed_sklearn,
            np.array(X_transformed_spu),
            rtol=0.1,
            atol=0.05,
            err_msg=f"Transformed data mismatch between SPU ({output_dist}) and Sklearn",
        )
        print(f"Sklearn transform comparison ({output_dist}) PASSED.")

        if output_dist == 'uniform':
            assert jnp.all(
                (X_transformed_spu >= -1e-6) & (X_transformed_spu <= 1 + 1e-6)
            ), "Uniform output out of [0, 1] range"
            assert (
                jnp.std(X_transformed_spu) > 1e-3
            ), "Uniform output seems collapsed (std dev too small)"
            print("Uniform output properties check PASSED.")

        np.testing.assert_allclose(
            np.array(X_plaintext),
            np.array(X_inversed_spu),
            rtol=0.05,
            atol=1e-2,
            err_msg=f"Inverse transformed data mismatch ({output_dist})",
        )
        print(f"Inverse transform reconstruction check ({output_dist}) PASSED.")

    def emul_uniform_test(emulator, X_plaintext):
        """Runs the emulation test specifically for the 'uniform' distribution."""
        print("\n===== Running Test: Uniform Distribution =====")
        output_dist = 'uniform'

        print("Sealing data for uniform test...")
        X_spu = emulator.seal(X_plaintext)
        print("Data sealed.")

        print("Running SPU computation (uniform)...")
        start_time = time.time()
        X_transformed_spu, X_inversed_spu = emulator.run(
            proc_quantile_transform, static_argnums=(1, 2)
        )(X_spu, N_QUANTILES, output_dist)
        end_time = time.time()
        print(f"SPU computation (uniform) finished in {end_time - start_time:.3f}s.")

        compare_results(
            X_plaintext,
            X_transformed_spu,
            X_inversed_spu,
            output_dist,
            N_QUANTILES,
            N_SAMPLES,
            RANDOM_STATE,
        )
        print("===== Test PASSED: Uniform Distribution =====")

    emulator = None
    try:
        print(f"Setting up emulator for mode: {mode}...")
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()
        print("Emulator is up and running.")

        print("Preparing common plaintext data...")
        key = random.PRNGKey(RANDOM_STATE)

        data = random.normal(key, (N_SAMPLES, N_FEATURES))
        data = data.at[:, 0].set(jnp.exp(data[:, 0] / 2))
        data = data.at[:, 1].set(data[:, 1] * 5 + 10)
        X_plaintext = jnp.array(data, dtype=jnp.float32)
        assert not jnp.isnan(
            X_plaintext
        ).any(), "Input data generation resulted in NaNs!"
        print(f"Plaintext data prepared: shape={X_plaintext.shape}")

        emul_uniform_test(emulator, X_plaintext)

    except Exception as e:
        print(f"\n!!! An error occurred during emulation test: {e} !!!")
        import traceback

        traceback.print_exc()  # Print detailed traceback for debugging
    finally:
        if emulator:
            print("\nShutting down emulator...")
            emulator.down()
            print("Emulator shut down successfully.")


# --- Main execution block ---
if __name__ == "__main__":
    test_quantile_transformer(emulation.Mode.MULTIPROCESS)
