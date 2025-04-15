# Copyright 2024 Ant Group Co., Ltd.
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

"""
ANOVA F-value statistic and p-value calculation.

Uses JAX for computations. P-value is calculated using a Chi-squared approximation
valid for large denominator degrees of freedom.
"""

import jax
import jax.numpy as jnp

# Define a small epsilon for safe division and comparisons
_EPSILON = 1e-8

# --- Chi2 SF implementation copied from univariate_selection.py ---
# (Required for p-value approximation)


def _igam(a, x, max_iter=3):
    """
    Computes the regularized lower incomplete gamma function P(a, x).

    Uses a power series expansion (DLMF 8.9.2).

    Parameters
    ----------
    a : float or array-like
        The shape parameter of the gamma function (must be positive).
    x : array-like, shape (n_quantiles,)
        The quantiles (upper integration limit) at which to compute the function.
        Must be non-negative.
    max_iter : int, default=3
        Maximum number of iterations for the power series computation.

    Returns
    -------
    ans : array-like, shape (n_quantiles,)
        The regularized lower incomplete gamma function P(a, x) evaluated at x.

    Notes
    -----
    - The `max_iter` parameter is crucial for numerical stability and convergence.
      Higher values increase accuracy but risk overflow/underflow, especially
      in fixed-point arithmetic. Consider using higher precision or reducing
      `max_iter` if stability issues arise.
    - This implementation assumes `jax.lax.lgamma` is available and stable in the
      target environment. If `lgamma` itself causes issues (e.g., legalization errors),
      this function will fail.
    - Input `x` is clamped to a minimum value `_EPSILON` to avoid `log(0)`.

    References
    ----------
    [1] NIST Digital Library of Mathematical Functions, Chapter 8,
        https://dlmf.nist.gov/8.9#E2
    """
    x_safe = jnp.maximum(x, _EPSILON)
    try:
        # Calculate ax = x**a * exp(-x) / Gamma(a) using logs for stability
        term1 = a * jnp.log(x_safe) - x - jax.lax.lgamma(a)
        ax = jnp.exp(term1)
    except Exception as e:
        print(f"DEBUG: Possible lgamma issue in _igam: {e}")
        raise e  # Re-raise the error if lgamma fails

    # Power series initialization
    r = jnp.full_like(x, a)  # Denominator term, starts at a
    c = jnp.ones_like(x)  # Current term multiplier
    ans = jnp.ones_like(x)  # Sum of series terms, starts with first term (1)

    # Power series loop: ans = 1 + x/(a+1) + x^2/((a+1)(a+2)) + ...
    def loop_body(_, val):
        r_in, c_in, ans_in = val
        r_new = r_in + 1.0
        # Avoid division by zero in loop, though r_new should be > a >= epsilon
        c_new = c_in * x / (r_new + _EPSILON)
        ans_new = ans_in + c_new
        return (r_new, c_new, ans_new)

    init_val = (r, c, ans)
    _, _, ans = jax.lax.fori_loop(0, max_iter, loop_body, init_val)

    # Final result is (sum of series terms) * ax / a
    # Avoid division by zero if a is very small
    return ans * ax / (a + _EPSILON)


def _igamc(a, x, max_iter=3):
    """
    Computes the complementary regularized lower incomplete gamma function Q(a, x).

    Uses a continued fraction representation (DLMF 8.11.4). Q(a, x) = 1 - P(a, x).

    Parameters
    ----------
    a : float or array-like
        The shape parameter of the gamma function (must be positive).
    x : array-like, shape (n_quantiles,)
        The quantiles (lower integration limit) at which to compute the function.
        Must be non-negative.
    max_iter : int, default=3
        Maximum number of iterations for the continued fraction computation.

    Returns
    -------
    ans : array-like, shape (n_quantiles,)
        The complementary regularized lower incomplete gamma function Q(a, x)
        evaluated at x.

    Notes
    -----
    - The `max_iter` parameter is crucial for numerical stability and convergence.
      Higher values increase accuracy but risk overflow/underflow, especially
      in fixed-point arithmetic. Consider using higher precision or reducing
      `max_iter` if stability issues arise. Continued fractions can be numerically
      unstable. Iterations are internally limited further for safety.
    - This implementation assumes `jax.lax.lgamma` is available and stable in the
      target environment. If `lgamma` itself causes issues (e.g., legalization errors),
      this function will fail.
    - Input `x` is clamped to a minimum value `_EPSILON` to avoid `log(0)`.
    - Intermediate terms `pk`, `qk` can grow rapidly; high precision (e.g., FM128)
      might be necessary. Result is clipped to prevent large unstable values.

    References
    ----------
    [1] NIST Digital Library of Mathematical Functions, Chapter 8,
        https://dlmf.nist.gov/8.11#E4
    """
    x_safe = jnp.maximum(x, _EPSILON)
    try:
        # Calculate ax = x**a * exp(-x) / Gamma(a) using logs for stability
        term1 = a * jnp.log(x_safe) - x - jax.lax.lgamma(a)
        ax = jnp.exp(term1)
    except Exception as e:
        print(f"DEBUG: Possible lgamma issue in _igamc: {e}")
        raise e  # Re-raise the error if lgamma fails

    # Lentz's method for continued fraction initialization
    y = jnp.ones_like(x) - a
    z = x + y + 1.0
    c = jnp.zeros_like(x)
    pkm2 = jnp.ones_like(x)
    # Avoid division by zero if x is zero initially
    qkm2 = x + _EPSILON
    pkm1 = x + 1.0
    qkm1 = z * x
    # Avoid division by zero if qkm1 is zero
    ans = pkm1 / (qkm1 + _EPSILON)

    # Continued Fraction loop (Lentz's algorithm requires careful implementation)
    # The version here seems based on modified Lentz or similar, directly calculating
    # convergents pk/qk. This can be unstable.
    def loop_body(_, val):
        (c_in, y_in, z_in, pkm1_in, pkm2_in, qkm1_in, qkm2_in), _ = (
            val  # Ignore previous ans
        )
        c_new = c_in + 1.0
        y_new = y_in + 1.0
        z_new = z_in + 2.0
        yc = y_new * c_new
        # Calculate numerator and denominator of k-th convergent
        pk = pkm1_in * z_new - pkm2_in * yc
        qk = qkm1_in * z_new - qkm2_in * yc

        # Avoid division by zero; use safe division
        qk_safe = jnp.where(qk == 0, _EPSILON, qk)
        r = pk / qk_safe

        # Store result and update terms for next iteration
        ans_new = r
        pkm2_new = pkm1_in
        pkm1_new = pk
        qkm2_new = qkm1_in
        qkm1_new = qk
        # Note: Potential for pk/qk to overflow/underflow rapidly.
        # SPU fixed-point might saturate/wrap around.

        return (c_new, y_new, z_new, pkm1_new, pkm2_new, qkm1_new, qkm2_new), ans_new

    params = (c, y, z, pkm1, pkm2, qkm1, qkm2)
    init_val = (params, ans)
    # Limit iterations more strictly than max_iter for stability
    limited_max_iter = min(max_iter, 10)
    params, ans = jax.lax.fori_loop(0, limited_max_iter, loop_body, init_val)

    # Clip result to avoid potential large values due to instability
    ans_clipped = jnp.clip(ans, 0.0, 1e8)

    # Final result Q(a, x) = [x^a * e^(-x) / Gamma(a)] * continued_fraction_result
    return ans_clipped * ax


def _sf(x, df, max_iter=3):
    """
    Calculates the survival function (1 - CDF) of the chi-squared distribution.

    Uses `_igam` and `_igamc` to compute the result based on the relationship
    between the chi-squared CDF and the incomplete gamma function.

    Parameters
    ----------
    x : array-like, shape (n_quantiles,)
        The quantiles (chi-squared values) at which to compute the survival function.
    df : float or array-like
        The degrees of freedom of the chi-squared distribution (must be positive).
    max_iter : int, default=3
        Maximum number of iterations passed to internal `_igam` and `_igamc` calls.

    Returns
    -------
    result : array-like, shape (n_quantiles,)
        The survival function (p-value) evaluated at the given quantiles x.

    Notes
    -----
    - The `max_iter` parameter impacts the accuracy and stability of the internal
      incomplete gamma function approximations. See notes in `_igam` and `_igamc`.
    - Requires `df` to be positive. Input is clamped to a minimum `_EPSILON`.
    - Relies on the numerical stability of the underlying `_igam` and `_igamc`.
    - Result is clipped to [0, 1] to handle potential approximation errors near the bounds.

    References
    ----------
    [1] NIST Digital Library of Mathematical Functions, Chapter 8 & 26,
        https://dlmf.nist.gov/8.2#E4
        https://dlmf.nist.gov/26.4#E13
    """
    # Ensure df > 0 for calculations
    df_safe = jnp.maximum(df, _EPSILON)
    # Ensure x is non-negative, as chi-squared values cannot be negative.
    x_safe = jnp.maximum(x, 0.0)

    # Chi2 CDF F(x; df) = P(df/2, x/2), where P is regularized lower incomplete gamma.
    # Chi2 SF = 1 - CDF = 1 - P(df/2, x/2) = Q(df/2, x/2), where Q is regularized upper/complementary.

    # Use appropriate function based on x relative to df for better stability:
    # - If x < df, compute 1 - P(df/2, x/2) using _igam.
    # - Otherwise (x >= df), compute Q(df/2, x/2) using _igamc.
    # Handle edge cases x=0 separately.

    condlist = [
        x_safe == 0,  # If x is 0, SF is 1
        x_safe < df_safe,  # If x < df, use 1 - _igam
    ]
    choicelist = [
        jnp.ones_like(x_safe),  # SF = 1 for x = 0
        1.0 - _igam(0.5 * df_safe, 0.5 * x_safe, max_iter),  # Compute 1 - P(a, x)
    ]
    # Clamp the result of 1.0 - _igam to [0, 1]
    choice_1_safe = jnp.clip(choicelist[1], 0.0, 1.0)

    result = jnp.select(
        condlist,
        [choicelist[0], choice_1_safe],
        default=_igamc(
            df_safe * 0.5, x_safe * 0.5, max_iter
        ),  # Compute Q(a, x) otherwise
    )

    # Final clamp for safety
    return jnp.clip(result, 0.0, 1.0)


# --- End Copied Chi2 SF implementation ---


def _ohe_manual(y, k, dtype):
    """
    Performs One-Hot Encoding on a JAX array assuming labels 0 to k-1.

    Parameters
    ----------
    y : jax.Array, shape (n_samples,)
        Input array of integer labels from 0 to k-1.
    k : int
        The number of distinct classes (must be positive).
    dtype : jax.Dtype
        Target JAX data type for the output OHE array.

    Returns
    -------
    ohe : jax.Array, shape (n_samples, k)
        The One-Hot Encoded array.
    """
    # Ensure y is 1D
    y_reshaped = y[:, jnp.newaxis]
    # Create array [0, 1, ..., k-1] for comparison
    classes_arr = jnp.arange(k, dtype=y.dtype)[jnp.newaxis, :]
    # Perform comparison using broadcasting
    ohe = y_reshaped == classes_arr
    return ohe.astype(dtype)


def f_classif_logic(X, y, n_classes, p_value_chi2_approx_iter=3):
    """
    Computes the ANOVA F-value statistic and its p-value for a single feature.

    Calculates the F-statistic based on the ratio of variance between groups
    (defined by y) to the variance within groups. The p-value is calculated
    using a Chi-squared approximation: Chi2.sf(dfn * F, dfn), which is generally
    valid for large denominator degrees of freedom (dfW = N - k).

    Parameters
    ----------
    X : jax.Array, shape (n_samples, 1)
        The input feature data (single feature column). Expected numerical dtype.
    y : jax.Array, shape (n_samples,)
        Target labels. Expected integer labels from 0 to n_classes-1.
    n_classes : int
        The number of distinct classes (k). Must be >= 0.
    p_value_chi2_approx_iter : int, default=3
        Maximum number of iterations for the internal Chi-squared SF approximation
        used for p-value calculation. Passed to `_sf`.

    Returns
    -------
    F_statistic : jax.Array, shape (1,)
        The computed F-statistic. Contains jnp.inf if within-group variance is
        zero and between-group variance is non-zero. Contains 0.0 if
        between-group variance is zero.
    p_value : jax.Array, shape (1,)
        The computed p-value based on the Chi-squared approximation. Is 1.0 if
        F-statistic is 0, and 0.0 if F-statistic is infinite.

    Raises
    ------
    ValueError
        If input shapes or n_classes are invalid.

    Notes
    -----
    - The p-value calculation relies on an approximation and may differ from
      exact methods, especially when the denominator degrees of freedom (N - k)
      is small.
    - The accuracy and stability of the p-value depend on the `_sf` implementation
      and the `p_value_chi2_approx_iter` parameter. See notes for `_sf`.
    - Requires N > k for a non-trivial result where dfW > 0. Edge cases
      k <= 1 and N <= k are handled.
    """
    # --- Input Validation ---
    if not isinstance(n_classes, int) or n_classes < 0:
        raise ValueError("n_classes must be a non-negative integer")
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError(f"X must have shape (n_samples, 1), got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must have shape (n_samples,), got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same samples: {X.shape[0]} != {y.shape[0]}"
        )

    N = X.shape[0]
    k = n_classes
    dtype = X.dtype

    # --- Handle Edge Cases (k<=1) ---
    if k <= 1:
        # F is undefined or 0. Return 0.0, 1.0 as a default.
        f_stat_res = jnp.array([0.0], dtype=dtype)
        p_val_res = jnp.array([1.0], dtype=dtype)
        return f_stat_res, p_val_res

    # --- Calculate Sums and Means ---
    # Need OHE to calculate group means and counts
    Y_ohe = _ohe_manual(y, k, dtype)  # Shape (N, k)
    n_g = jnp.sum(Y_ohe, axis=0)  # Shape (k,) - counts per group
    n_g_safe = n_g + _EPSILON  # Avoid division by zero if a group is empty

    X_col = X.reshape((N, 1))  # Ensure shape (N, 1)
    # Sum per group: einsum('nk,nd->kd', Y_ohe, X_col) -> shape (k, 1)
    # More simply for single feature: sum(X_col * Y_ohe, axis=0) -> shape (k,)
    sum_g = jnp.sum(X_col * Y_ohe, axis=0)  # Shape (k,)
    mean_g = sum_g / n_g_safe  # Shape (k,)

    # Avoid division by zero if N=0 (should be caught by k>=1 check)
    N_safe = N + _EPSILON
    overall_mean = jnp.sum(X_col) / N_safe  # Scalar

    # --- Calculate Sum of Squares ---
    SST = jnp.sum((X_col - overall_mean) ** 2)  # Total SS
    SSB = jnp.sum(n_g * (mean_g - overall_mean) ** 2)  # Between SS
    SSB = jnp.maximum(SSB, 0.0)  # Ensure non-negative
    SSW = jnp.maximum(SST - SSB, 0.0)  # Within SS (ensure non-negative)

    # --- Handle N <= k case (dfW <= 0) ---
    if N <= k:
        # Degrees of freedom within is zero or negative. F is degenerate.
        is_ssb_zero = SSB <= _EPSILON
        # If SSB=0, F=0, p=1. If SSB>0, F=inf, p=0.
        f_stat_res = jnp.where(is_ssb_zero, 0.0, jnp.inf)
        p_val_res = jnp.where(is_ssb_zero, 1.0, 0.0)
        return f_stat_res.reshape(1), p_val_res.reshape(1)

    # --- Main Calculation (N > k > 1) ---
    dfB = k - 1
    dfW = N - k
    # Use computation dtype, ensure dfB > 0
    dfB_f = jnp.array(dfB, dtype=dtype)

    # Mean Squares
    MSB = SSB / (dfB_f + _EPSILON)
    MSW = SSW / (jnp.array(dfW, dtype=dtype) + _EPSILON)  # dfW > 0 here

    # Flags for edge cases based on Mean Squares
    is_msb_zero = MSB <= _EPSILON
    is_msw_zero = MSW <= _EPSILON

    # Calculate F-statistic
    # If MSW is effectively zero, F approaches infinity (if MSB > 0)
    f_stat = jnp.where(is_msw_zero, jnp.inf, MSB / MSW)
    # If MSB is effectively zero, F is zero
    f_stat_res = jnp.where(is_msb_zero, 0.0, f_stat)

    # Calculate p-value using Chi-squared approximation: Chi2.sf(dfn * F, dfn)
    # Need a finite F value for the approximation input
    f_stat_finite = MSB / (MSW + _EPSILON)
    f_stat_finite_safe = jnp.maximum(f_stat_finite, 0.0)  # Ensure non-negative

    # Chi2 approx arguments: x = dfn * F, df = dfn
    chi2_approx_x = dfB_f * f_stat_finite_safe
    chi2_approx_df = dfB_f  # Must be > 0 since k > 1

    # Calculate approximate p-value using the chi2 _sf function
    p_val_approx = _sf(chi2_approx_x, chi2_approx_df, max_iter=p_value_chi2_approx_iter)

    # Determine final p-value based on edge cases for F derived from MSB/MSW
    p_val_res = jnp.where(
        is_msb_zero,
        1.0,  # If F = 0 (MSB=0), p = 1
        jnp.where(
            is_msw_zero, 0.0, p_val_approx  # If F = inf (MSW=0, MSB>0), p = 0
        ),  # Otherwise, use approximation
    )

    # Return results shaped as (1,)
    return f_stat_res.reshape(1), p_val_res.reshape(1)


def f_classif_multi(X, y, n_classes, p_value_chi2_approx_iter=3):
    """
    Computes ANOVA F-values and p-values for multiple features simultaneously.

    Calculates the F-statistic and p-value for each feature column in X
    independently against the target labels y. P-values are computed using the
    Chi-squared approximation: Chi2.sf(dfn * F, dfn).

    Parameters
    ----------
    X : jax.Array, shape (n_samples, n_features)
        The input feature data matrix. Expected numerical dtype.
    y : jax.Array, shape (n_samples,)
        Target labels. Expected integer labels from 0 to n_classes-1.
    n_classes : int
        The number of distinct classes (k). Must be >= 0.
    p_value_chi2_approx_iter : int, default=3
        Maximum number of iterations for the internal Chi-squared SF approximation
        used for p-value calculation. Passed to `_sf`.

    Returns
    -------
    F_statistics : jax.Array, shape (n_features,)
        The computed F-statistic for each feature.
    p_values : jax.Array, shape (n_features,)
        The computed p-value for each feature based on the Chi-squared approximation.

    Raises
    ------
    ValueError
        If input shapes or n_classes are invalid.

    Notes
    -----
    - This function vectorizes the calculation across features for efficiency.
    - P-value calculation relies on the Chi-squared approximation (valid for large N-k).
      See notes in `f_classif_logic` and `_sf`.
    - Handles edge cases k <= 1 and N <= k.
    """
    # --- Input Validation ---
    if not isinstance(n_classes, int) or n_classes < 0:
        raise ValueError("n_classes must be a non-negative integer")
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n_samples, n_features), got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must have shape (n_samples,), got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same samples: {X.shape[0]} != {y.shape[0]}"
        )

    N, n_features = X.shape
    k = n_classes
    dtype = X.dtype

    # --- Handle Edge Cases (k<=1) ---
    if k <= 1:
        # Return arrays of 0s (F) and 1s (p) for all features
        f_stat_res = jnp.zeros(n_features, dtype=dtype)
        p_val_res = jnp.ones(n_features, dtype=dtype)
        return f_stat_res, p_val_res

    # --- Calculate OHE and Group Counts (Common for all features) ---
    Y_ohe = _ohe_manual(y, k, dtype)  # Shape (N, k)
    n_g = jnp.sum(Y_ohe, axis=0)  # Shape (k,)
    n_g_safe = n_g + _EPSILON

    # --- Calculate Sums and Means (Vectorized across features) ---
    # Sum per group for each feature using einsum or matmul
    # sum_g[i, j] = sum(X[p, j] for p where y[p] == i)
    sum_g = jnp.einsum("nk,nd->kd", Y_ohe, X)  # Shape (k, n_features)
    # mean_g[i, j] = mean(X[p, j] for p where y[p] == i)
    # Need broadcasting for division: (k, n_features) / (k, 1)
    mean_g = sum_g / n_g_safe[:, None]  # Shape (k, n_features)

    # Overall mean per feature
    overall_mean = jnp.mean(X, axis=0)  # Shape (n_features,)

    # --- Calculate Sum of Squares (Vectorized across features) ---
    # SSB: Sum over groups [ n_g * (mean_g - overall_mean)^2 ]
    # Broadcasting: (k, 1) * ( (k, n_features) - (1, n_features) )^2
    SSB = jnp.sum(
        n_g[:, None] * (mean_g - overall_mean[None, :]) ** 2, axis=0
    )  # Sum over k -> shape (n_features,)
    SSB = jnp.maximum(SSB, 0.0)

    # SST: Sum over samples [ (X - overall_mean)^2 ]
    # Broadcasting: (N, n_features) - (1, n_features)
    SST = jnp.sum(
        (X - overall_mean[None, :]) ** 2, axis=0
    )  # Sum over N -> shape (n_features,)
    SSW = jnp.maximum(SST - SSB, 0.0)  # Within SS, shape (n_features,)

    # --- Handle N <= k case (dfW <= 0) ---
    if N <= k:
        is_ssb_zero = SSB <= _EPSILON  # Shape (n_features,)
        f_stat_res = jnp.where(is_ssb_zero, 0.0, jnp.inf)
        p_val_res = jnp.where(is_ssb_zero, 1.0, 0.0)
        return f_stat_res, p_val_res

    # --- Main Calculation (N > k > 1, vectorized) ---
    dfB = k - 1
    dfW = N - k
    # Ensure floats for division
    dfB_f = jnp.array(dfB, dtype=dtype)
    dfW_f = jnp.array(dfW, dtype=dtype)

    # Mean Squares (vectorized)
    MSB = SSB / (dfB_f + _EPSILON)  # Shape (n_features,)
    MSW = SSW / (dfW_f + _EPSILON)  # Shape (n_features,)

    # Flags for edge cases (vectorized)
    is_msb_zero = MSB <= _EPSILON  # Shape (n_features,)
    is_msw_zero = MSW <= _EPSILON  # Shape (n_features,)

    # Calculate F-statistic (vectorized)
    f_stat = jnp.where(is_msw_zero, jnp.inf, MSB / MSW)
    f_stat_res = jnp.where(is_msb_zero, 0.0, f_stat)  # Shape (n_features,)

    # Calculate p-value using Chi-squared approximation (vectorized)
    f_stat_finite = MSB / (MSW + _EPSILON)
    f_stat_finite_safe = jnp.maximum(f_stat_finite, 0.0)

    # Chi2 approx arguments (vectorized)
    # Need dfB_f broadcastable to shape (n_features,) if needed, but it's scalar here
    chi2_approx_x = dfB_f * f_stat_finite_safe  # Shape (n_features,)
    chi2_approx_df = dfB_f  # Scalar df

    # Calculate approximate p-value using the chi2 _sf function (vectorized)
    # _sf should handle array inputs for x if df is scalar
    p_val_approx = _sf(
        chi2_approx_x, chi2_approx_df, max_iter=p_value_chi2_approx_iter
    )  # Shape (n_features,)

    # Determine final p-value based on edge cases (vectorized)
    p_val_res = jnp.where(
        is_msb_zero,
        1.0,  # If F = 0, p = 1
        jnp.where(
            is_msw_zero, 0.0, p_val_approx  # If F = inf, p = 0
        ),  # Otherwise, use approximation
    )  # Shape (n_features,)

    return f_stat_res, p_val_res
