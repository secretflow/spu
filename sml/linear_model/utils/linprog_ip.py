import jax
import jax.numpy as jnp
from jax import jit

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
import jax.scipy.linalg as jsl
from jax import lax, jit

# @jit
def cho_solve(c_and_lower, b):
    c, lower = c_and_lower
    
    # Solve for y in Ly = b (if lower is True, otherwise L^T y = b)
    y_Ture = jnp.linalg.solve(c, b)
    x_True = jnp.linalg.solve(c.T, y_Ture)
    
    y_False = jnp.linalg.solve(c.T, b)
    x_False = jnp.linalg.solve(c, y_False)
    y = jnp.where(lower, y_Ture, y_False)
    x = jnp.where(lower, x_True, x_False)
    
    return x

def _sym_solve(Dinv, A, r1, r2, M, lstsq=False, sym_pos=True, cholesky=True):
    def solve(r, sym_pos=sym_pos):
        x = jnp.zeros_like(r)
        # x_lstsq = jnp.linalg.lstsq(M, r)[0]
        
        Q, R = jnp.linalg.qr(M)
        x_lstsq = jnp.linalg.solve(R, jnp.dot(Q.T, r))
        
        L = jnp.linalg.cholesky(M)
        x_cholesky = cho_solve((L, True), r)
        # x_has_nan代表M不是正定的
        x_has_nan = jnp.isnan(x_cholesky).any()
        x_not_cholesky = jnp.linalg.solve(M , r)
        x_cholesky = jnp.where(x_has_nan, x_not_cholesky, x_cholesky)
        
        x_sym_pos = jsl.solve(M, r, assume_a="pos")
        x_sym = jsl.solve(M, r)
        x_sym_pos = jnp.where(x_has_nan, x_sym, x_sym_pos)
        
        x = jnp.where(lstsq, x_lstsq, x)
        cholesky_T_condition = jnp.logical_and(cholesky, ~lstsq)
        cholesky_F_condition = jnp.logical_and(~cholesky, ~lstsq)
        x = jnp.where(cholesky_T_condition, x_cholesky, x)
        sym_pos_condition = jnp.logical_and(sym_pos, cholesky_F_condition)
        sym_condition = jnp.logical_and(~sym_pos, cholesky_F_condition)
        x = jnp.where(sym_pos_condition, x_sym_pos, x)
        x = jnp.where(sym_condition, x_sym, x)
        
        return x
    
    r = r2 + jnp.dot(A, Dinv * r1)
    v = solve(r)
    u = Dinv * (jnp.dot(A.T, v) - r1)

    return u, v

# @jit
def _get_delta(A, b, c, x, y, z, tau, kappa, gamma, 
                pc=True, ip=False, lstsq=False, sym_pos=True, cholesky=True):
    # @jit
    def eta(g):
        return jnp.where(ip, 1, 1 - g)
    n_x = len(x)
    
    # [4] Equation 8.8
    r_P = b * tau - jnp.dot(A, x)
    r_D = c * tau - jnp.dot(A.T, y) - z
    r_G = jnp.dot(c, x) - jnp.dot(b.T, y) + kappa
    mu = (jnp.dot(x, z) + tau * kappa) / (n_x + 1)

    #  Assemble M from [4] Equation 8.31
    Dinv = x / z
    
    # # 不使用稀疏矩阵
    M = jnp.dot(A, Dinv.reshape(-1, 1) * A.T)
    
    # pc: "predictor-corrector" [4] Section 4.1
    # In development this option could be turned off
    # but it always seems to improve performance substantially
    # pc的影响是n_corrections，导致后续循环中i==1是否运行
    # n_corrections = 1 if pc else 1
    
    # i = 0
    alpha, d_x, d_z, d_tau, d_kappa = 0, 0, 0, 0, 0
    
    # loop1
    # Reference [4] Eq. 8.6
    rhatp = eta(gamma) * r_P
    rhatd = eta(gamma) * r_D
    rhatg = eta(gamma) * r_G
    
    # Reference [4] Eq. 8.7
    rhatxs = gamma * mu - x * z
    rhattk = gamma * mu - tau * kappa
    
    # [4] Equation 8.28
    p, q = _sym_solve(Dinv, A, c, b, M, lstsq, sym_pos, cholesky)

    # [4] Equation 8.29
    u, v = _sym_solve(Dinv, A, rhatd -
                        (1 / x) * rhatxs, rhatp, M, lstsq, sym_pos, cholesky)

    p, q = _sym_solve(Dinv, A, c, b, M, lstsq, sym_pos, cholesky)
    # [4] Equation 8.29
    u, v = _sym_solve(Dinv, A, rhatd -
                    (1 / x) * rhatxs, rhatp, M, lstsq, sym_pos, cholesky)
    nan_0_condition = jnp.logical_or(jnp.any(p==0), jnp.any(q==0))
    nan_condition = jnp.logical_or(jnp.any(jnp.isnan(p)), jnp.any(jnp.isnan(q)))
    nan_condition = jnp.logical_or(nan_0_condition, nan_condition)
    cholesky = jnp.where(nan_condition, False, cholesky)

    p, q = _sym_solve(Dinv, A, c, b, M, lstsq, sym_pos, cholesky)
    # [4] Equation 8.29
    u, v = _sym_solve(Dinv, A, rhatd -
                    (1 / x) * rhatxs, rhatp, M, lstsq, sym_pos, cholesky)
    
    d_tau = (rhatg + 1 / tau * rhattk - (-jnp.dot(c, u) + jnp.dot(b, v))) / (
        1 / tau * kappa + (-jnp.dot(c, p) + jnp.dot(b, q))
    )

    d_x = u + p * d_tau
    d_y = v + q * d_tau

    # [4] Relations between  after 8.25 and 8.26
    # d_z = (1 / x) * (rhatxs - z * d_x)
    # d_z这个除法导致问题
    d_z = (1 / (x + 1e-5)) * (rhatxs - z * d_x)
    d_kappa = 1 / tau * (rhattk - kappa * d_tau)
    
    # [4] 8.12 and "Let alpha be the maximal possible step..." before 8.23
    alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, 1)
    gamma = jnp.where(ip, 10, (1 - alpha)**2 * jnp.minimum(0.1, (1 - alpha)))

    # loop2
    # Reference [4] Eq. 8.6
    rhatp = eta(gamma) * r_P
    rhatd = eta(gamma) * r_D
    rhatg = eta(gamma) * r_G
    
    # Reference [4] Eq. 8.7
    rhatxs = gamma * mu - x * z
    rhattk = gamma * mu - tau * kappa

    rhatxs_ip_True = ((1 - alpha) * gamma * mu - x * z - alpha**2 * d_x * d_z)
    rhattk_ip_True = ((1 - alpha) * gamma * mu - tau * kappa - alpha**2 * d_tau * d_kappa)
    
    rhatxs_ip_False = rhatxs - d_x * d_z
    rhattk_ip_False = rhattk - d_tau * d_kappa
    
    rhatxs = jnp.where(ip, rhatxs_ip_True, rhatxs_ip_False)
    rhattk = jnp.where(ip, rhattk_ip_True, rhattk_ip_False)
        
    # [4] Equation 8.28
    p, q = _sym_solve(Dinv, A, c, b, M, lstsq, sym_pos, cholesky)
    # [4] Equation 8.29
    u, v = _sym_solve(Dinv, A, rhatd -
                        (1 / x) * rhatxs, rhatp, M, lstsq, sym_pos, cholesky)

    p, q = _sym_solve(Dinv, A, c, b, M, lstsq, sym_pos, cholesky)
    # [4] Equation 8.29
    u, v = _sym_solve(Dinv, A, rhatd -
                    (1 / x) * rhatxs, rhatp, M, lstsq, sym_pos, cholesky)
    nan_0_condition = jnp.logical_or(jnp.any(p==0), jnp.any(q==0))
    nan_condition = jnp.logical_or(jnp.any(jnp.isnan(p)), jnp.any(jnp.isnan(q)))
    nan_condition = jnp.logical_or(nan_0_condition, nan_condition)
    cholesky = jnp.where(nan_condition, False, cholesky)

    p, q = _sym_solve(Dinv, A, c, b, M, lstsq, sym_pos, cholesky)
    # [4] Equation 8.29
    u, v = _sym_solve(Dinv, A, rhatd -
                    (1 / x) * rhatxs, rhatp, M, lstsq, sym_pos, cholesky)
    
    d_tau = (rhatg + 1 / tau * rhattk - (-jnp.dot(c, u) + jnp.dot(b, v))) / (
        1 / tau * kappa + (-jnp.dot(c, p) + jnp.dot(b, q))
    )
    d_x = u + p * d_tau
    d_y = v + q * d_tau

    # [4] Relations between  after 8.25 and 8.26
    d_z = (1 / x) * (rhatxs - z * d_x)
    d_kappa = 1 / tau * (rhattk - kappa * d_tau)
    
    # [4] 8.12 and "Let alpha be the maximal possible step..." before 8.23
    alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, 1)
    gamma = jnp.where(ip, 10, (1 - alpha)**2 * jnp.minimum(0.1, (1 - alpha)))

    return d_x, d_y, d_z, d_tau, d_kappa

# @jit
def _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0):
    # [4] 4.3 Equation 8.21, ignoring 8.20 requirement
    # same step is taken in primal and dual spaces
    # alpha0 is basically beta3 from [4] Table 8.1, but instead of beta3
    # the value 1 is used in Mehrota corrector and initial point correction

    i_x = d_x < 0
    i_z = d_z < 0

    # result_x = jnp.where(i_x, x / -d_x, jnp.inf)
    result_x = jnp.min(jnp.where(i_x, x / -d_x, 1.75921860e13))
    # result_x = jnp.where(i_x, x / -d_x, 100000)
    alpha_x = jnp.where(jnp.any(i_x), alpha0 * result_x, 1.0)
    
    result_z = jnp.min(jnp.where(i_z, z / -d_z, 1.75921860e13))
    # result_z = jnp.where(i_z, z / -d_z, 100000)
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
# @jit
def _get_blind_start(shape):
    """
    Return the starting point from [4] 4.4

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    m, n = shape
    x0 = jnp.ones(n)
    y0 = jnp.zeros(m)
    z0 = jnp.ones(n)
    tau0 = 1
    kappa0 = 1
    return x0, y0, z0, tau0, kappa0

def _indicators(A, b, c, c0, x, y, z, tau, kappa):
    """
    Implementation of several equations from [4] used as indicators of
    the status of optimization.

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """

    # residuals for termination are relative to initial values
    x0, y0, z0, tau0, kappa0 = _get_blind_start(A.shape)

    # See [4], Section 4 - The Homogeneous Algorithm, Equation 8.8
    def r_p(x, tau):
        return b * tau - jnp.dot(A, x)

    def r_d(y, z, tau):
        return c * tau - jnp.dot(A.T, y) - z

    def r_g(x, y, kappa):
        return kappa + jnp.dot(c, x) - jnp.dot(b, y)

    # np.dot unpacks if they are arrays of size one
    def mu(x, tau, z, kappa):
        return (jnp.dot(x, z) + jnp.dot(tau, kappa)) / (len(x) + 1)

    obj = jnp.dot(c, x / tau) + c0

    def norm(a):
        return jnp.linalg.norm(a)

    # See [4], Section 4.5 - The Stopping Criteria
    r_p0 = r_p(x0, tau0)
    r_d0 = r_d(y0, z0, tau0)
    r_g0 = r_g(x0, y0, kappa0)
    mu_0 = mu(x0, tau0, z0, kappa0)
    rho_A = norm(jnp.dot(c.T, x) - jnp.dot(b.T, y)) / (tau + norm(jnp.dot(b.T, y)))
    rho_p = norm(r_p(x, tau)) / jnp.maximum(1, norm(r_p0))
    rho_d = norm(r_d(y, z, tau)) / jnp.maximum(1, norm(r_d0))
    rho_g = norm(r_g(x, y, kappa)) / jnp.maximum(1, norm(r_g0))
    rho_mu = mu(x, tau, z, kappa) / mu_0
    return rho_p, rho_d, rho_A, rho_g, rho_mu, obj

def _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj, header=False):
    """
    Print indicators of optimization status to the console.

    Parameters
    ----------
    rho_p : float
        The (normalized) primal feasibility, see [4] 4.5
    rho_d : float
        The (normalized) dual feasibility, see [4] 4.5
    rho_g : float
        The (normalized) duality gap, see [4] 4.5
    alpha : float
        The step size, see [4] 4.3
    rho_mu : float
        The (normalized) path parameter, see [4] 4.5
    obj : float
        The objective function value of the current iterate
    header : bool
        True if a header is to be printed

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    # if header:
    #     print("Primal Feasibility ",
    #           "Dual Feasibility   ",
    #           "Duality Gap        ",
    #           "Step            ",
    #           "Path Parameter     ",
    #           "Objective          ")

    # no clue why this works
    alpha_converted = jnp.where(isinstance(alpha, str), alpha, float(alpha))
    fmt = '{0:<20.13}{1:<20.13}{2:<20.13}{3:<17.13}{4:<20.13}{5:<20.13}'
    print(fmt.format(
        float(rho_p),
        float(rho_d),
        float(rho_g),
        alpha_converted,
        # alpha if isinstance(alpha, str) else float(alpha),
        float(rho_mu),
        float(obj)))

def perform_iteration(A, b, c, c0, x, y, z, tau, kappa, ip, pc, beta, alpha0, lstsq, sym_pos, cholesky, tol, maxiter, iteration, status):
    iteration += 1
    print("iteration", iteration)
    
    gamma = jnp.where(ip, 1, jnp.where(pc, 0, beta * jnp.mean(z * x)))

    # Solve [4] 8.6 and 8.7/8.13/8.23
    d_x, d_y, d_z, d_tau, d_kappa = _get_delta(
        A, b, c, x, y, z, tau, kappa, gamma, pc, ip, lstsq, sym_pos, cholesky)

    step = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0)
    alpha = jnp.where(ip, 1.0, step)
    
    # 执行步骤
    x, y, z, tau, kappa = _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)
    
    # 对 x, z, tau, kappa 进行处理
    x = jnp.where(ip & (x < 1), 1, x)
    z = jnp.where(ip & (z < 1), 1, z)
    tau = jnp.where(ip, jnp.maximum(1, tau), tau)
    kappa = jnp.where(ip, jnp.maximum(1, kappa), kappa)

    ip = False
    
    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
        A, b, c, c0, x, y, z, tau, kappa)
    go = jnp.logical_or((rho_g > tol), jnp.logical_or((rho_d > tol), (rho_A > tol)))
    
    inf1 = jnp.logical_and(
        jnp.logical_and(rho_p < tol, rho_d < tol),
        jnp.logical_and(rho_g < tol, tau < tol * jnp.maximum(1, kappa))
    )
    inf2 = jnp.logical_and(rho_mu < tol, tau < tol * jnp.minimum(1, kappa))

    status = jnp.where(
        jnp.logical_or(inf1, inf2),
        jnp.where(b.transpose().dot(y) > tol, 2, 3),
        status
    )
    status = jnp.where(iteration >= maxiter, 1, status)
    
    return x, y, z, tau, kappa, ip, status, iteration, go

# @jit
def _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, tol,
            pc, ip, lstsq, sym_pos, cholesky):
    
    iteration = 0

    # default initial point
    x, y, z, tau, kappa = _get_blind_start(A.shape)
    
    # first iteration is special improvement of initial point
    ip = jnp.where(pc, ip, False)
    
    # [4] 4.5
    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
        A, b, c, c0, x, y, z, tau, kappa)

    # if disp:
    #     _display_iter(rho_p, rho_d, rho_g, "-", rho_mu, obj, header=True)
        
    status = 0
    go = jnp.logical_or((rho_g > tol),jnp.logical_or((rho_d > tol),(rho_A > tol)))
    # while go:
    while iteration < 3: 
        
        x, y, z, tau, kappa, ip, status, iteration, go = perform_iteration(
            A, b, c, c0, x, y, z, tau, kappa, ip, pc, beta, alpha0, lstsq, sym_pos, cholesky, tol, maxiter, iteration, status
        )
    
    x_hat = x / tau
    # [4] Statement after Theorem 8.2
    return x_hat, status, iteration

@jit
def _linprog_ip(c, c0, A, b, maxiter=1000, tol=1e-8,
                alpha0=.99995, beta=0.1, lstsq=False, sym_pos=True,
                cholesky=None, pc=True, ip=False):

    cholesky = jnp.logical_or(
        cholesky,  # 如果 cholesky 不是 None
        jnp.logical_and(
            cholesky is None,
            jnp.logical_and(sym_pos, jnp.logical_not(lstsq))  # 且 sym_pos 为 True, lstsq 为 False
        )
    )
    # cholesky = cholesky or (cholesky is None and sym_pos and not lstsq)
    x, status, iteration = _ip_hsd(A, b, c, c0, alpha0, beta,
                                            maxiter, tol, pc, ip,
                                            lstsq, sym_pos, cholesky)
    print(x)

    return x, status, iteration