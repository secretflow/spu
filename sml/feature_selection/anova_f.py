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

_EPSILON = 1e-8


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


def _regularized_beta(x, a, b, max_iter=100):
    """
    Computes the regularized incomplete Beta function I_x(a, b) using the continued fraction method.

    Args:
        x (float or array): The point at which to evaluate the regularized incomplete Beta function.
        a (float or array): First shape parameter of the Beta distribution.
        b (float or array): Second shape parameter of the Beta distribution.
        max_iter (int, optional): Maximum number of iterations for the continued fraction approximation. Default is 100.

    Returns:
        float or array: The value of the regularized incomplete Beta function I_x(a, b).
    """
    x = jnp.clip(x, _EPSILON, 1 - _EPSILON)

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
        fpmin = 1e-30  # Minimum allowable value for intermediate calculations
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

    # Calculate the Beta function log value
    ln_front = a * jnp.log(x) + b * jnp.log(1.0 - x) - _log_beta(a, b)
    front = jnp.exp(ln_front)
    cf = betacf(a, b, x)  # Compute the continued fraction approximation
    return front * cf / a  # Return the regularized Beta function


def f_dist_sf(f, d1, d2, max_iter=100):
    """
    Computes the survival function (1 - CDF) of the F-distribution with parameters d1 (degrees of freedom 1)
    and d2 (degrees of freedom 2) using the regularized Beta function.

    Args:
        f (float): The F-statistic value.
        d1 (int): The degrees of freedom for the numerator (first group).
        d2 (int): The degrees of freedom for the denominator (second group).
        max_iter (int, optional): Maximum number of iterations for the Beta continued fraction. Default is 100.

    Returns:
        float: The survival function (1 - CDF) of the F-distribution at the given F value.
    """
    f_safe = jnp.maximum(f, 0.0)  # Ensure F is non-negative
    x = d2 / (d2 + d1 * f_safe + _EPSILON)
    a = d2 / 2.0
    b = d1 / 2.0
    return _regularized_beta(x, a, b, max_iter=max_iter)


def f_classif_multi(X, y, n_classes, p_value_iter=100):
    """
    Performs multi-class classification using the F-statistic for feature selection and computes
    p-values for each feature using the F-distribution.

    Args:
        X (array): The input feature matrix (N samples x n_features).
        y (array): The labels (N samples) for classification.
        n_classes (int): The number of classes in the target labels.
        p_value_iter (int, optional): The number of iterations for calculating p-values. Default is 100.

    Returns:
        tuple:
            - F-statistics for each feature.
            - p-values for each feature.
    """
    N, n_features = X.shape
    k = n_classes
    dtype = X.dtype

    if k <= 1:
        return jnp.zeros(n_features, dtype=dtype), jnp.ones(n_features, dtype=dtype)

    # Create one-hot encoded labels for each class
    y_oh = (y[:, None] == jnp.arange(k)).astype(dtype)
    n_g = jnp.sum(y_oh, axis=0)
    n_g_safe = n_g + _EPSILON

    # Compute the sum of feature values per class and class means
    sum_g = jnp.einsum('nk,nd->kd', y_oh, X)
    mean_g = sum_g / n_g_safe[:, None]
    overall_mean = jnp.mean(X, axis=0)

    # Between-class sum of squares (SSB) and total sum of squares (SST)
    SSB = jnp.sum(n_g[:, None] * (mean_g - overall_mean[None, :]) ** 2, axis=0)
    SST = jnp.sum((X - overall_mean[None, :]) ** 2, axis=0)
    SSB = jnp.maximum(SSB, 0.0)
    SSW = jnp.maximum(SST - SSB, 0.0)

    if N <= k:
        is_ssb_zero = SSB <= _EPSILON
        return jnp.where(is_ssb_zero, 0.0, jnp.inf), jnp.where(is_ssb_zero, 1.0, 0.0)

    # Degrees of freedom for between-class (dfB) and within-class (dfW)
    dfB = k - 1
    dfW = N - k
    dfB_f = jnp.array(dfB, dtype=dtype)
    dfW_f = jnp.array(dfW, dtype=dtype)

    # Mean squares between (MSB) and within (MSW)
    MSB = SSB / (dfB_f + _EPSILON)
    MSW = SSW / (dfW_f + _EPSILON)

    # Handle zero cases for MSB and MSW
    is_msb_zero = MSB <= _EPSILON
    is_msw_zero = MSW <= _EPSILON

    # F-statistic: ratio of mean squares
    f_stat = jnp.where(is_msw_zero, jnp.inf, MSB / MSW)
    f_stat_res = jnp.where(is_msb_zero, 0.0, f_stat)

    # F-statistic approximation for finite values
    f_stat_finite_safe = MSB / (MSW + _EPSILON)
    p_val_approx = f_dist_sf(f_stat_finite_safe, dfB_f, dfW_f, max_iter=p_value_iter)

    # Final p-value results
    p_val_res = jnp.where(is_msb_zero, 1.0, jnp.where(is_msw_zero, 0.0, p_val_approx))
    return f_stat_res, p_val_res
