import os
import sys

import jax.numpy as jnp
import numpy as np

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import sml.utils.emulation as emulation
from sml.fyy_pca.jacobi_evd import generate_ring_sequence, serial_jacobi_evd


def emul_jacobievd(mode: emulation.Mode.MULTIPROCESS):
    print("start jacobi evd emulation.")
    np.random.seed(0)

    # ONLY test small matrix for usage purpose
    n = 10
    mat = jnp.array(np.random.rand(n, n))
    mat = (mat + mat.T) / 2

    def _check_jacobievd_single(mat, max_jacobi_iter=5):
        print("start jacobi evd emulation test, with shape=", mat.shape)

        mat_spu = emulator.seal(mat)
        rotate_mat_spu = emulator.seal(jnp.eye(mat.shape[0]))
        val, vec = emulator.run(serial_jacobi_evd, static_argnums=(2,))(
            mat_spu, rotate_mat_spu, max_jacobi_iter
        )
        sorted_indices = jnp.argsort(val)[::-1]
        eig_vec = vec.T[sorted_indices]
        eig_val = val[sorted_indices]

        val_np, vec_np = np.linalg.eig(mat)
        sorted_indices = jnp.argsort(val_np)[::-1]
        eig_vec_np = vec_np.T[sorted_indices]
        eig_val_np = val_np[sorted_indices]

        abs_diff = np.abs(np.abs(eig_vec_np) - np.abs(eig_vec))
        rel_error = abs_diff / (np.abs(eig_vec_np) + 1e-8)

        print("avg absolute error:\n", np.mean(abs_diff))
        print("avg relative error:\n", np.mean(rel_error))

        # check eigen values equal
        np.testing.assert_allclose(eig_val_np, eig_val, rtol=0.01, atol=0.01)

        # check eigen vectors (maybe with sign flip)
        np.testing.assert_allclose(
            np.abs(eig_vec_np), np.abs(eig_vec), rtol=0.01, atol=0.01
        )

    try:
        conf_path = "sml/fyy_pca/emulations/3pc_128.json"
        emulator = emulation.Emulator(conf_path, mode, bandwidth=300, latency=20)
        emulator.up()

        _check_jacobievd_single(mat)

        print("evd emulation pass.")

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_jacobievd(emulation.Mode.MULTIPROCESS)
