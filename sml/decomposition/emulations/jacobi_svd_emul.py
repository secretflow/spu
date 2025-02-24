import os
import sys

import jax.numpy as jnp
import jax.random as random
import jax.lax as lax
import numpy as np
from scipy.linalg import svd as scipy_svd

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

import sml.utils.emulation as emulation
from sml.decomposition.jacobi_svd import jacobi_svd


def emul_jacobi_svd(mode=emulation.Mode.MULTIPROCESS):
    print("Start Jacobi SVD emulation.")

    def proc_transform(A):
        singular_values = jacobi_svd(A)
        return singular_values

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Create a random dataset
        A = random.normal(random.PRNGKey(seed), (10, 10))
        A = (A + A.T) / 2
        A_spu = emulator.seal(A)
        singular_values = emulator.run(proc_transform)(A_spu)

        # Compare with scipy
        _, singular_values_scipy, _ = scipy_svd(np.array(A), full_matrices=False)

        # Compare the results
        np.testing.assert_allclose(
            np.sort(singular_values), np.sort(singular_values_scipy), atol=1e-3
        )

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_jacobi_svd(emulation.Mode.MULTIPROCESS)
