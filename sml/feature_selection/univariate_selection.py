import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats.chi2 import sf
from sklearn.datasets import load_iris


def chi2(X, y, label_lst):
    """
    Calculate the chi-squared statistic and p-value for feature independence testing.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The feature matrix from which to calculate the chi-squared statistic. Each row
        represents a sample and each column a feature.

    y : array-like, shape (n_samples, n_classes)
        The one-hot encoded target vector indicating the class membership of
        each sample. Each row should correspond to the class labels of the sample in
        X.

    label_lst : list of str, length = n_classes
        A list of class labels corresponding to the columns of y.

    Returns
    -------
    chi2_stats : array, shape (n_features,)
        The chi-squared statistic for each feature, indicating the degree of
        association between the feature and the class labels.

    p_value : array, shape (n_features,)
        The p-value associated with each chi-squared statistic, which can be used
        to test the null hypothesis that the features are independent of the
        class labels.
    """
    total_samples = X.shape[0]
    num_feature = X.shape[1]
    num_class = len(label_lst)
    X = jnp.array(X)
    y = jnp.array(y)  # Ensure y is a one-hot encoded vector
    feature_count = jnp.sum(X, axis=0)
    class_prob = jnp.mean(y, axis=0)
    # observed = jnp.zeros((num_feature, num_class))
    # expected = jnp.zeros((num_feature, num_class))
    # Calculate the observed frequency count
    observed = jnp.dot(X.T, y)
    # Calculate the expected frequency count
    expected = jnp.outer(feature_count, class_prob)
    # Calculate the chi-squared statistic
    chi2_stats = (observed - expected) ** 2 / expected
    # Sum over class dimensions
    chi2_stats = jnp.nansum(chi2_stats, axis=1)
    # Degrees of freedom
    df = num_class - 1
    # Calculate the p-value for each feature
    p_value = sf(chi2_stats, df=df)
    return chi2_stats, p_value


if __name__ == '__main__':
    x, y = load_iris(return_X_y=True)
    label_lst = np.unique(y)
    num_class = len(label_lst)
    y = np.eye(num_class)[y]
    chi2_stats, p_value = chi2(x, y, label_lst)
    print(chi2_stats, p_value)
    from sklearn.feature_selection import chi2 as chi2_sklearn

    sklearn_chi2_stats, sklearn_p_value = chi2_sklearn(x, y)
    print(sklearn_chi2_stats, sklearn_p_value)
