import jax
import jax.numpy as jnp


def _sf(x, df, max_iter=3):
    """
    Calculates the survival function (1 - CDF) of the chi-squared distribution.

    Parameters
    ----------
    x : array-like, shape (n_quantiles,)
        The quantiles at which to compute the survival function.

    df : int
        The degrees of freedom of the chi-squared distribution. Must be positive.

    max_iter : int, default=3
        The maximum number of iterations for the numerical computation.

    Returns
    -------
    result : array-like, shape (n_quantiles,)
        The survival function evaluated at the given quantiles.

    Notes
    -----
    The `max_iter` parameter is critical for the numerical stability of the computation. A high value can lead to overflow or underflow errors. Consider using a higher precision data type or reducing `max_iter` to ensure stability.
    """
    if df <= 0:
        raise ValueError("Domain error, df must be positive")
    condlist = [x < 0, x == 0, jnp.logical_or(x < 1, x < df)]
    choicelist = [
        jnp.ones_like(x),
        jnp.zeros_like(x),
        1 - _igam(0.5 * df, 0.5 * x, max_iter),
    ]
    result = jnp.select(
        condlist, choicelist, default=_igamc(df * 0.5, x * 0.5, max_iter)
    )
    return result


def _igam(a, x, max_iter=3):
    """
    Computes the regularized lower incomplete gamma function using a power series expansion for a given shape parameter.

    Parameters
    ----------
    a : int
        The shape parameter of the gamma function.

    x : array-like, shape (n_quantiles,)
        The quantiles at which to compute the incomplete gamma function.

    max_iter : int, default=3
        The maximum number of iterations for the numerical computation.

    Returns
    -------
    ans : array-like, shape (n_quantiles,)
        The regularized lower incomplete gamma function evaluated at the given quantiles.

    Notes
    -----
    The `max_iter` parameter is crucial for the numerical stability of the computation. A high value can lead to overflow or underflow errors. It is recommended to use a higher precision data type or reduce the `max_iter` value to avoid such issues.
    """
    ax = jnp.power(x, a) * jnp.exp(-x) * jnp.exp(-jax.lax.lgamma(a))
    # Power series
    r = jnp.full_like(x, a)
    c = jnp.ones_like(x)
    ans = jnp.ones_like(x)

    def loop_body(_, val):
        r, c, ans = val
        r += 1.0
        c *= x / r
        ans += c
        return (r, c, ans)

    init_val = (r, c, ans)
    _, _, ans = jax.lax.fori_loop(0, max_iter, loop_body, init_val)
    return ans * ax / a


def _igamc(a, x, max_iter=3):
    """
    Computes the complementary regularized lower incomplete gamma function using a continued fraction representation for a given shape parameter.

    Parameters
    ----------
    a : int
        The shape parameter of the gamma function.

    x : array-like, shape (n_quantiles,)
        The quantiles at which to compute the complementary of incomplete gamma function.

    max_iter : int, default=3
        The maximum number of iterations for the numerical computation.

    Returns
    -------
    ans : array-like, shape (n_quantiles,)
        The complementary regularized lower incomplete gamma function evaluated at the given quantiles.

    Notes
    -----
    The `max_iter` parameter is crucial for the numerical stability of the computation. A high value can lead to overflow or underflow errors. It is recommended to use a higher precision data type or reduce the `max_iter` value to avoid such issues.
    """
    # Compute ax = x**a * exp(-x) / Gamma(a)
    ax = jnp.power(x, a) * jnp.exp(-x) * jnp.exp(-jax.lax.lgamma(a))
    y = jnp.ones_like(x) - a
    z = x + y + 1.0
    c = jnp.zeros_like(x)
    pkm2 = jnp.ones_like(x)
    qkm2 = jnp.full_like(x, a)
    pkm1 = x + 1.0
    qkm1 = z * x
    ans = pkm1 / qkm1

    # Using Continued Fraction
    def loop_body(_, val):
        (c, y, z, pkm1, pkm2, qkm1, qkm2), ans = val
        c += 1.0
        y += 1.0
        z += 2.0
        yc = y * c
        pk = pkm1 * z - pkm2 * yc
        qk = qkm1 * z - qkm2 * yc
        r = pk / qk
        ans = r
        pkm2 = pkm1
        pkm1 = pk
        qkm2 = qkm1
        qkm1 = qk
        return (c, y, z, pkm1, pkm2, qkm1, qkm2), ans

    params = (c, y, z, pkm1, pkm2, qkm1, qkm2)
    init_val = (params, ans)
    params, ans = jax.lax.fori_loop(0, max_iter, loop_body, init_val)
    return ans * ax


def chi2(X, y, n_classes, max_iter=3, compute_p_value=False):
    """
    Performs a chi-squared test for independence between features and class labels.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input feature matrix where each row represents a sample and each column a feature.

    y : array-like, shape (n_samples,)
        An array of class labels for the samples in `X`. Labels should be integers in the range [0, n_classes-1].

    n_classes : int
        The number of unique classes in the classification problem.

    max_iter : int, default=3
        The maximum number of iterations for the numerical computation of the p-value. This parameter is crucial for numerical stability and convergence.

    compute_p_value : bool, default=False
        Whether to compute the p-value alongside the chi-squared statistic.

    Returns
    -------
    chi2_stats : array-like, shape (n_features,)
        The chi-squared statistics computed for each feature.

    p_value : array-like, shape (n_features,), optional
        The p-values corresponding to each feature's chi-squared statistic. Returned only if `compute_p_value` is set to True.

    Notes
    -----
    - The `max_iter` parameter should be set with consideration of the potential for numerical overflow or underflow. In cases where numerical stability is a concern, consider using a higher precision data type (e.g., Float128 if available) or reducing the `max_iter` value.
    - The p value computation cost maybe expensive. For applications such as feature selection where only the chi-squared statistic is needed, it may be more efficient to set `compute_p_value` to False.
    - The input `y` should be provided as a 1-dimensional array with class labels encoded as integers in the specified range.
    """

    # one hot encoding
    y = jnp.eye(n_classes)[y]
    X = jnp.array(X)
    y = jnp.array(y)
    feature_count = jnp.sum(X, axis=0)
    class_prob = jnp.mean(y, axis=0)
    # Calculate the observed frequency count
    observed = jnp.dot(X.T, y)
    expected = jnp.outer(feature_count, class_prob)
    # Calculate the chi-squared statistic
    chi2_stats = (observed - expected) ** 2 / expected
    # Sum over class dimensions
    chi2_stats = jnp.nansum(chi2_stats, axis=1)
    # Degrees of freedom
    df = n_classes - 1
    p_value = None
    if compute_p_value:
        # Calculate the p-value for each feature
        p_value = _sf(chi2_stats, df=df, max_iter=max_iter)
        p_value = jnp.where(p_value < 0, 0, p_value)
    return chi2_stats, p_value
