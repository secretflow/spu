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

import jax
import jax.numpy as jnp


def _log_beta(a, b):
    """
    Computes the logarithm of the Beta function using the Log-Gamma approximation.

    Args:
        a (float or array): First shape parameter of the Beta distribution.
        b (float or array): Second shape parameter of the Beta distribution.

    Returns:
        float or array: The log of the Beta function, log(B(a, b)).
    """
    return jax.lax.lgamma(a) + jax.lax.lgamma(b) - jax.lax.lgamma(a + b)


def _regularized_beta(x, a, b, max_iter=100, epsilon=1e-8, fpmin=1e-30):
    """
    Computes the regularized incomplete Beta function I_x(a, b) using the continued fraction method.

    Args:
        x (float or array): The point at which to evaluate the regularized incomplete Beta function.
        a (float or array): First shape parameter of the Beta distribution.
        b (float or array): Second shape parameter of the Beta distribution.
        max_iter (int, optional): Maximum number of iterations for the continued fraction approximation. Default is 100.
        epsilon (float, optional): Small value for numerical stability. Default is 1e-8.
        fpmin (float, optional): Small value to prevent underflow in continued fraction. Default is 1e-30.

    Returns:
        float or array: The value of the regularized incomplete Beta function I_x(a, b).
    """
    x = jnp.clip(x, epsilon, 1 - epsilon)

    def betacf(a, b, x):
        """
        Computes the continued fraction for the incomplete Beta function I_x(a, b) using a stable iterative method.

        Args:
            a (float): First shape parameter of the Beta distribution.
            b (float): Second shape parameter of the Beta distribution.
            x (float): The point at which to evaluate the incomplete Beta function.

        Returns:
            float: The value of the continued fraction approximation for I_x(a, b).
        """
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0

        c = jnp.ones_like(x)
        d = 1.0 - qab * x / qap
        d = jnp.where(jnp.abs(d) < fpmin, fpmin, d)
        d = 1.0 / d
        h = d

        def body(m, val):
            """
            Computes the iterative step of the continued fraction using recurrence relations.

            Args:
                m (int): Current iteration step.
                val (tuple): Tuple containing the intermediate values (c, d, h) for the continued fraction.

            Returns:
                tuple: Updated values of (c, d, h) after one iteration.
            """
            c, d, h = val
            m2 = 2 * m

            num = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + num * d
            d = jnp.where(jnp.abs(d) < fpmin, fpmin, d)
            c = 1.0 + num / c
            c = jnp.where(jnp.abs(c) < fpmin, fpmin, c)
            d = 1.0 / d
            h *= d * c

            num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + num * d
            d = jnp.where(jnp.abs(d) < fpmin, fpmin, d)
            c = 1.0 + num / c
            c = jnp.where(jnp.abs(c) < fpmin, fpmin, c)
            d = 1.0 / d
            h *= d * c

            return (c, d, h)

        init_val = (c, d, h)
        _, _, h = jax.lax.fori_loop(1, max_iter, body, init_val)
        return h

    ln_front = a * jnp.log(x) + b * jnp.log(1.0 - x) - _log_beta(a, b)
    front = jnp.exp(ln_front)
    cf = betacf(a, b, x)
    return front * cf / a


def f_dist_sf(f, d1, d2, max_iter=100, epsilon=1e-8, fpmin=1e-30):
    """
    Computes the survival function (1 - CDF) of the F-distribution with parameters d1 and d2 using the regularized Beta function.

    Args:
        f (float): The F-statistic value.
        d1 (int): The degrees of freedom for the numerator (first group).
        d2 (int): The degrees of freedom for the denominator (second group).
        max_iter (int, optional): Maximum number of iterations for the Beta continued fraction. Default is 100.
        epsilon (float, optional): Small value for numerical stability. Default is 1e-8.
        fpmin (float, optional): Small value to prevent underflow in continued fraction. Default is 1e-30.

    Returns:
        float: The survival function (1 - CDF) of the F-distribution at the given F value.
    """
    f_safe = jnp.maximum(f, 0.0)
    x = d2 / (d2 + d1 * f_safe + epsilon)
    a = d2 / 2.0
    b = d1 / 2.0
    return _regularized_beta(x, a, b, max_iter=max_iter, epsilon=epsilon, fpmin=fpmin)


def f_classif(X, y, n_classes, p_value_iter=100, epsilon=1e-8, fpmin=1e-30):
    """
    Computes the ANOVA F-statistic and corresponding p-values for each feature in a multi-class classification dataset.

    Args:
        X (array): The input feature matrix of shape (N, n_features), where N is the number of samples.
        y (array): The target labels of shape (N,), with integer values from 0 to n_classes-1.
        n_classes (int): The number of unique classes in y.
        p_value_iter (int, optional): The maximum number of iterations for the continued fraction approximation in the p-value calculation. Default is 100.
        epsilon (float, optional): A small value used for numerical stability, e.g., to avoid division by zero. Default is 1e-8.
        fpmin (float, optional): A small value used in the continued fraction approximation to prevent underflow. Default is 1e-30.

    Returns:
        tuple:
            - F-statistics for each feature (array of shape (n_features,)).
            - p-values for each feature (array of shape (n_features,)).
    """
    N, n_features = X.shape
    k = n_classes
    dtype = X.dtype

    if k <= 1:
        return jnp.zeros(n_features, dtype=dtype), jnp.ones(n_features, dtype=dtype)

    y_oh = (y[:, None] == jnp.arange(k)).astype(dtype)
    n_g = jnp.sum(y_oh, axis=0)
    n_g_safe = n_g + epsilon

    sum_g = jnp.einsum('nk,nd->kd', y_oh, X)
    mean_g = sum_g / n_g_safe[:, None]
    overall_mean = jnp.mean(X, axis=0)

    SSB = jnp.sum(n_g[:, None] * (mean_g - overall_mean[None, :]) ** 2, axis=0)
    SST = jnp.sum((X - overall_mean[None, :]) ** 2, axis=0)
    SSB = jnp.maximum(SSB, 0.0)
    SSW = jnp.maximum(SST - SSB, 0.0)

    if N <= k:
        is_ssb_zero = SSB <= epsilon
        return jnp.where(is_ssb_zero, 0.0, jnp.inf), jnp.where(is_ssb_zero, 1.0, 0.0)

    dfB = k - 1
    dfW = N - k
    dfB_f = jnp.array(dfB, dtype=dtype)
    dfW_f = jnp.array(dfW, dtype=dtype)

    MSB = SSB / (dfB_f + epsilon)
    MSW = SSW / (dfW_f + epsilon)

    is_msb_zero = MSB <= epsilon
    is_msw_zero = MSW <= epsilon

    f_stat = jnp.where(is_msw_zero, jnp.inf, MSB / MSW)
    f_stat_res = jnp.where(is_msb_zero, 0.0, f_stat)

    f_stat_finite_safe = MSB / (MSW + epsilon)
    p_val_approx = f_dist_sf(
        f_stat_finite_safe,
        dfB_f,
        dfW_f,
        max_iter=p_value_iter,
        epsilon=epsilon,
        fpmin=fpmin,
    )

    p_val_res = jnp.where(is_msb_zero, 1.0, jnp.where(is_msw_zero, 0.0, p_val_approx))
    return f_stat_res, p_val_res
