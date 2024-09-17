# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from warnings import warn

import jax
import jax.numpy as jnp


def _sym_solve(Dinv, A, r1, r2, M):
    r = r2 + A.dot(Dinv * r1)
    v = jnp.linalg.solve(M, r)
    u = Dinv * (A.T.dot(v) - r1)
    return u, v


def _get_delta(
    A,
    b,
    c,
    x,
    y,
    z,
    tau,
    kappa,
    gamma,
    pc=True,
    ip=False,
):
    def eta(g):
        return jnp.where(ip, 1, 1 - g)

    n_x = len(x)

    # [4] Equation 8.8
    r_P = b * tau - A.dot(x)
    r_D = c * tau - A.T.dot(y) - z
    r_G = c.dot(x) - b.transpose().dot(y) + kappa
    mu = (x.dot(z) + tau * kappa) / (n_x + 1)

    #  Assemble M from [4] Equation 8.31
    Dinv = x / (z)

    M = A.dot(Dinv.reshape(-1, 1) * A.T)

    # pc: "predictor-corrector" [4] Section 4.1
    # In development this option could be turned off
    # but it always seems to improve performance substantially
    # pc的影响是n_corrections，导致后续循环中i==1是否运行

    alpha, d_x, d_z, d_tau, d_kappa = 0, 0, 0, 0, 0

    # loop1
    # Reference [4] Eq. 8.6
    eta_gamma = eta(gamma)
    rhatp = eta_gamma * r_P
    rhatd = eta_gamma * r_D
    rhatg = eta_gamma * r_G

    # Reference [4] Eq. 8.7
    rhatxs = gamma * mu - x * z
    rhattk = gamma * mu - tau * kappa

    # [4] Equation 8.28
    p, q = _sym_solve(Dinv, A, c, b, M)

    # [4] Equation 8.29
    u, v = _sym_solve(Dinv, A, rhatd - (1 / x) * rhatxs, rhatp, M)

    d_tau = (rhatg + 1 / tau * rhattk - (-c.dot(u) + b.dot(v))) / (
        1 / tau * kappa + (-c.dot(p) + b.dot(q))
    )
    d_x = u + p * d_tau
    d_y = v + q * d_tau

    # [4] Relations between  after 8.25 and 8.26

    dinv_value = safe_value(
        1 / (x),
        lower_bound_neg=-2,
        upper_bound_neg=None,
        lower_bound_pos=None,
        upper_bound_pos=2,
    )
    d_z = dinv_value * (rhatxs - z * d_x)

    d_kappa = 1 / tau * (rhattk - kappa * d_tau)

    # [4] 8.12 and "Let alpha be the maximal possible step..." before 8.23
    alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, 1)
    gamma = jnp.where(ip, 10, (1 - alpha) ** 2 * jnp.minimum(0.1, (1 - alpha)))

    # loop2
    # Reference [4] Eq. 8.6
    eta_gamma = eta(gamma)
    rhatp = eta_gamma * r_P
    rhatd = eta_gamma * r_D
    rhatg = eta_gamma * r_G

    # Reference [4] Eq. 8.7
    rhatxs = gamma * mu - x * z
    rhattk = gamma * mu - tau * kappa

    rhatxs_ip_True = (1 - alpha) * gamma * mu - x * z - alpha**2 * d_x * d_z
    rhattk_ip_True = (1 - alpha) * gamma * mu - tau * kappa - alpha**2 * d_tau * d_kappa

    rhatxs_ip_False = rhatxs - d_x * d_z
    rhattk_ip_False = rhattk - d_tau * d_kappa

    rhatxs = jnp.where(ip, rhatxs_ip_True, rhatxs_ip_False)
    rhattk = jnp.where(ip, rhattk_ip_True, rhattk_ip_False)

    # [4] Equation 8.28
    p, q = _sym_solve(Dinv, A, c, b, M)
    # [4] Equation 8.29
    u, v = _sym_solve(Dinv, A, rhatd - (1 / x) * rhatxs, rhatp, M)

    d_tau = (rhatg + 1 / tau * rhattk - (-c.dot(u) + b.dot(v))) / (
        1 / tau * kappa + (-c.dot(p) + b.dot(q))
    )
    d_x = u + p * d_tau
    d_y = v + q * d_tau

    # [4] Relations between  after 8.25 and 8.26

    dinv_value = safe_value(
        1 / (x),
        lower_bound_neg=-0.75,
        upper_bound_neg=None,
        lower_bound_pos=None,
        upper_bound_pos=0.75,
    )
    d_z = dinv_value * (rhatxs - z * d_x)

    d_kappa = 1 / tau * (rhattk - kappa * d_tau)

    return d_x, d_y, d_z, d_tau, d_kappa


def _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0):
    # [4] 4.3 Equation 8.21, ignoring 8.20 requirement
    # same step is taken in primal and dual spaces
    # alpha0 is basically beta3 from [4] Table 8.1, but instead of beta3
    # the value 1 is used in Mehrota corrector and initial point correction

    i_x = d_x < 0
    i_z = d_z < 0

    result_x = jnp.min(jnp.where(i_x, x / -d_x, 1))
    alpha_x = jnp.where(jnp.any(i_x), alpha0 * result_x, 1.0)

    result_z = jnp.min(jnp.where(i_z, z / -d_z, 1))
    alpha_z = jnp.where(jnp.any(i_z), alpha0 * result_z, 1.0)

    alpha_tau = jnp.where(d_tau < 0, alpha0 * tau / -d_tau, 1)
    alpha_kappa = jnp.where(d_kappa < 0, alpha0 * kappa / -d_kappa, 1)

    alpha = jnp.min(jnp.array([1, alpha_x, alpha_tau, alpha_z, alpha_kappa]))

    return alpha


def _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha):
    x = x + alpha * d_x
    tau = tau + alpha * d_tau
    z = z + alpha * d_z
    kappa = kappa + alpha * d_kappa
    y = y + alpha * d_y
    return x, y, z, tau, kappa


def _get_blind_start(shape):
    m, n = shape
    x0 = jnp.ones(n)
    y0 = jnp.zeros(m)
    z0 = jnp.ones(n)
    tau0 = 1
    kappa0 = 1
    return x0, y0, z0, tau0, kappa0


def _indicators(A, b, c, c0, x, y, z, tau, kappa):
    # residuals for termination are relative to initial values
    x0, y0, z0, tau0, kappa0 = _get_blind_start(A.shape)

    # See [4], Section 4 - The Homogeneous Algorithm, Equation 8.8
    def r_p(x, tau):
        return b * tau - A.dot(x)

    def r_d(y, z, tau):
        return c * tau - A.T.dot(y) - z

    def r_g(x, y, kappa):
        return kappa + c.dot(x) - b.dot(y)

    # np.dot unpacks if they are arrays of size one
    def mu(x, tau, z, kappa):
        return (x.dot(z) + tau * kappa) / (len(x) + 1)

    obj = c.dot(x / tau) + c0

    def norm(a):
        return jnp.linalg.norm(a)

    # See [4], Section 4.5 - The Stopping Criteria
    r_p0 = r_p(x0, tau0)
    r_d0 = r_d(y0, z0, tau0)
    r_g0 = r_g(x0, y0, kappa0)
    mu_0 = mu(x0, tau0, z0, kappa0)
    rho_A = norm(c.T.dot(x) - b.T.dot(y)) / (tau + norm(b.T.dot(y)))
    rho_p = norm(r_p(x, tau)) / jnp.maximum(1, norm(r_p0))
    rho_d = norm(r_d(y, z, tau)) / jnp.maximum(1, norm(r_d0))
    rho_g = norm(r_g(x, y, kappa)) / jnp.maximum(1, norm(r_g0))
    rho_mu = mu(x, tau, z, kappa) / mu_0
    return rho_p, rho_d, rho_A, rho_g, rho_mu, obj


def safe_value(
    x,
    lower_bound_neg=-1e5,
    upper_bound_neg=-1e-5,
    lower_bound_pos=1e-5,
    upper_bound_pos=1e5,
):
    x_neg = jnp.clip(x, lower_bound_neg, upper_bound_neg)
    x_pos = jnp.clip(x, lower_bound_pos, upper_bound_pos)
    return jnp.where(x < 0, x_neg, x_pos)


def perform_iteration(
    A,
    b,
    c,
    c0,
    x,
    y,
    z,
    tau,
    kappa,
    ip,
    pc,
    beta,
    alpha0,
    tol,
    iteration,
    status,
):
    iteration += 1
    # print("iteration", iteration)

    gamma = jnp.where(ip, 1, jnp.where(pc, 0, beta * jnp.mean(z * x)))

    # Solve [4] 8.6 and 8.7/8.13/8.23
    d_x, d_y, d_z, d_tau, d_kappa = _get_delta(
        A, b, c, x, y, z, tau, kappa, gamma, pc, ip
    )

    step = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0)
    alpha = jnp.where(ip, 1.0, step)

    x, y, z, tau, kappa = _do_step(
        x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha
    )

    x = jnp.where(ip & (x < 1), 1, x)
    z = jnp.where(ip & (z < 1), 1, z)
    tau = jnp.where(ip, jnp.maximum(1, tau), tau)
    kappa = jnp.where(ip, jnp.maximum(1, kappa), kappa)

    ip = False

    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
        A, b, c, c0, x, y, z, tau, kappa
    )

    inf1 = jnp.logical_and(
        jnp.logical_and(rho_p < tol, rho_d < tol),
        jnp.logical_and(rho_g < tol, tau < tol * jnp.maximum(1, kappa)),
    )
    inf2 = jnp.logical_and(rho_mu < tol, tau < tol * jnp.minimum(1, kappa))

    status = jnp.where(
        jnp.logical_or(inf1, inf2), jnp.where(b.transpose().dot(y) > tol, 1, 2), status
    )
    # 0 : Optimization terminated successfully
    # 1 : Problem appears to be infeasible
    # 2 : Problem appears to be unbounded

    return x, y, z, tau, kappa, ip, status, iteration


def _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, tol, pc, ip):

    iteration = 0

    # default initial point
    x, y, z, tau, kappa = _get_blind_start(A.shape)

    # first iteration is special improvement of initial point
    ip = jnp.where(pc, ip, False)

    # [4] 4.5
    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
        A, b, c, c0, x, y, z, tau, kappa
    )

    status = 0
    while iteration < maxiter:
        x, y, z, tau, kappa, ip, status, iteration = perform_iteration(
            A,
            b,
            c,
            c0,
            x,
            y,
            z,
            tau,
            kappa,
            ip,
            pc,
            beta,
            alpha0,
            tol,
            iteration,
            status,
        )
    x_hat = x / tau

    return x_hat, status, iteration


def _linprog_ip(
    c,
    c0,
    A,
    b,
    maxiter=10,
    tol=1e-8,
    alpha0=0.99995,
    beta=0.1,
    pc=True,
    ip=False,
):

    x, status, iteration = _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, tol, pc, ip)

    return x, status, iteration
