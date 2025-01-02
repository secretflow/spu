import jax.numpy as jnp
import jax.random as random
import jax.lax as lax
from jax import jit, vmap
import numpy as np
import time

@jit
def jacobi_rotation(A, p, q):
    tau = (A[q, q] - A[p, p]) / (2 * A[p, q])
    t = jnp.sign(tau) / (jnp.abs(tau) + jnp.sqrt(1 + tau**2))
    c = 1 / jnp.sqrt(1 + t**2)
    s = t * c
    return c, s

@jit
def apply_jacobi_rotation_A(A, c, s, p, q):
    A_new = A.copy()
    A = A.at[p, :].set(c * A_new[p, :] - s * A_new[q, :])
    A = A.at[q, :].set(s * A_new[p, :] + c * A_new[q, :])
    A_new = A.copy()
    A = A.at[:, p].set(c * A_new[:, p] - s * A_new[:, q])
    A = A.at[:, q].set(s * A_new[:, p] + c * A_new[:, q])
    return A

@jit
def jacobi_svd(A, tol=1e-10, max_iter=5):
    n = A.shape[0]
    A = jnp.array(A)

    def body_fun(i, val):
        A, max_off_diag, iterations = val
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
        return A, max_off_diag, iterations

    max_off_diag = jnp.inf
    iterations = 0
    A, _, final_iterations = lax.fori_loop(0, max_iter, body_fun, (A, max_off_diag, iterations))

    singular_values = jnp.abs(jnp.diag(A))
    idx = jnp.argsort(-singular_values)
    singular_values = singular_values[idx]
    
    return singular_values

def generate_symmetric_matrix(n, seed=0):
    A = random.normal(random.PRNGKey(seed), (n, n))
    S = (A + A.T) / 2
    return S

n = 10

A_jax = generate_symmetric_matrix(n)

start_time = time.time()
singular_values = jacobi_svd(A_jax)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Run Time: {elapsed_time:.6f} s")

print("Singular Values Jacobi_svd:")
print(singular_values)

A_np = np.array(A_jax)
_, Sigma, _ = np.linalg.svd(A_np)
print("Sigma:")
print(Sigma)