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
        U, singular_values, V_T = jacobi_svd(A)
        return U, singular_values, V_T

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Create a random dataset
        A = random.normal(random.PRNGKey(0), (10, 10))
        A = (A + A.T) / 2
        A_np = np.array(A)
        A_spu = emulator.seal(A)
        results = emulator.run(proc_transform)(A_spu)

        # Compare with sklearn
        model = SklearnSVD(n_components=min(A_np.shape))
        model.fit(A_np)
        Sigma = model.singular_values_
        Matrix = model.components_
        
        # Compare the results
        np.testing.assert_allclose(Sigma, results[1], rtol=0.01, atol=0.01)
        np.testing.assert_allclose(np.abs(Matrix), np.abs(results[2]), rtol=0.01, atol=0.01)
        
    finally:
        emulator.down()


if __name__ == "__main__":
    emul_jacobi_svd(emulation.Mode.MULTIPROCESS)
