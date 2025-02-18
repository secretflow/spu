import os
import sys
import jax.numpy as jnp
import jax.random as random
import jax.lax as lax
import numpy as np
from scipy.linalg import svd as scipy_svd

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import sml.utils.emulation as emulation

def generate_symmetric_matrix(n, seed=0):
    A = random.normal(random.PRNGKey(seed), (n, n))
    S = (A + A.T) / 2
    return S

def jacobi_rotation(A, p, q):
    tau = (A[q, q] - A[p, p]) / (2 * A[p, q])
    t = jnp.sign(tau) / (jnp.abs(tau) + jnp.sqrt(1 + tau**2))
    c = 1 / jnp.sqrt(1 + t**2)
    s = t * c
    return c, s

def apply_jacobi_rotation_A(A, c, s, p, q):
    A_new = A.copy()
    A = A.at[p, :].set(c * A_new[p, :] - s * A_new[q, :])
    A = A.at[q, :].set(s * A_new[p, :] + c * A_new[q, :])
    A_new = A.copy()
    A = A.at[:, p].set(c * A_new[:, p] - s * A_new[:, q])
    A = A.at[:, q].set(s * A_new[:, p] + c * A_new[:, q])
    return A

def jacobi_svd(A, tol=1e-10, max_iter=5):
    n = A.shape[0]
    A = jnp.array(A)

    def body_fun(i, val):
        A, max_off_diag = val
        mask = jnp.abs(A - jnp.diagonal(A)) > tol  
        for p in range(n):
            for q in range(p + 1, n):
                A = lax.cond(
                    mask[p, q],
                    lambda A: apply_jacobi_rotation_A(A, *jacobi_rotation(A, p, q), p, q),
                    lambda A: A,
                    A
                )
                max_off_diag = lax.cond(
                    mask[p, q],
                    lambda x: jnp.maximum(x, jnp.abs(A[p, q])),
                    lambda x: x,
                    max_off_diag
                )       
        return A, max_off_diag

    max_off_diag = jnp.inf
    A, _, = lax.fori_loop(0, max_iter, body_fun, (A, max_off_diag))

    singular_values = jnp.abs(jnp.diag(A))
    idx = jnp.argsort(-singular_values)
    singular_values = singular_values[idx]
    
    return singular_values

def emul_jacobi_svd(mode=emulation.Mode.MULTIPROCESS):
    print("Start Jacobi SVD emulation.")

    def proc_transform(A):
        singular_values = jacobi_svd(A)
        return singular_values

    try:
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        A = generate_symmetric_matrix(10)
        A_spu = emulator.seal(A)
        singular_values = emulator.run(proc_transform)(A_spu)

        _, singular_values_scipy, _ = scipy_svd(np.array(A), full_matrices=False)

        np.testing.assert_allclose(np.sort(singular_values), np.sort(singular_values_scipy), atol=1e-3)

    finally:
        emulator.down()

if __name__ == "__main__":
    emul_jacobi_svd(emulation.Mode.MULTIPROCESS)
