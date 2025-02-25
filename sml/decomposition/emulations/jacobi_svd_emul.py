import os
import sys

import jax.random as random
import numpy as np
from sklearn.decomposition import TruncatedSVD as SklearnSVD

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
        A_np = np.array(A)
        A_spu = emulator.seal(A)
        singular_values = emulator.run(proc_transform)(A_spu)

        # Compare with sklearn
        model = SklearnSVD(n_components=min(A_np.shape))
        model.fit(A_np)
        Sigma = model.singular_values_

        # Compare the results
        np.testing.assert_allclose(singular_values, Sigma, rtol=0.1, atol=0.1)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_jacobi_svd(emulation.Mode.MULTIPROCESS)
