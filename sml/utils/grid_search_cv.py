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

import copy
import itertools
import time
import warnings
from collections import defaultdict

import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from sml.metrics.classification.classification import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sml.metrics.regression.regression import mean_squared_error

# --- Helper Functions ---


def generate_param_combinations(param_grid):
    """
    Generates all parameter combinations from a grid dictionary.

    Args:
        param_grid (dict): Dictionary with parameter names (str) as keys and lists of parameter settings as values.

    Yields:
        dict: A dictionary representing one combination of parameters.
    """
    if not param_grid:
        yield {}
        return
    param_names = list(param_grid.keys())
    value_lists = list(param_grid.values())
    for value_combination in itertools.product(*value_lists):
        params = dict(zip(param_names, value_combination))
        yield params


# --- Scoring Functions ---


def jax_r2_score(y_true, y_pred):
    """
    Calculates the R^2 (coefficient of determination) regression score using JAX.

    Args:
        y_true (jnp.ndarray): True target values.
        y_pred (jnp.ndarray): Predicted target values.

    Returns:
        float: R^2 score.
    """
    y_true, y_pred = jnp.asarray(y_true).ravel(), jnp.asarray(y_pred).ravel()
    ss_res = jnp.sum(jnp.square(y_true - y_pred))
    ss_tot = jnp.sum(jnp.square(y_true - jnp.mean(y_true)))

    special_case = ss_tot == 0
    r2_special = jnp.where(ss_res == 0, 1.0, 0.0)

    r2_normal = 1 - (ss_res / ss_tot)

    return jnp.where(special_case, r2_special, r2_normal)


# --- Cross-Validation Functions ---


def jax_kfold_split(n_samples, n_splits, shuffle=False, random_state=None):
    """
    Generates K-Fold train/test indices with even fold sizes using JAX.

    Args:
        n_samples (int): Total number of samples.
        n_splits (int): Number of folds (K). Must be >= 2.
        shuffle (bool): Whether to shuffle data before splitting.
        random_state (int): Seed for shuffling.

    Yields:
        tuple: (train_indices, test_indices) for each fold.
    """
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2.")
    if n_splits > n_samples:
        raise ValueError(f"n_splits ({n_splits}) > n_samples ({n_samples}).")

    indices = jnp.arange(n_samples)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop

    for k in range(n_splits):
        test_indices = folds[k]
        train_indices = jnp.concatenate([folds[i] for i in range(n_splits) if i != k])
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError("Generated empty train or test set.")
        yield train_indices, test_indices


# --- Main GridSearchCV Class ---


class GridSearchCV:
    """
    Exhaustive search over specified parameter values for an estimator.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Args:
        estimator: Estimator object. This is assumed to implement the scikit-learn estimator interface.
        param_grid (dict): Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values.
        scoring (str or callable): Strategy to evaluate the performance of the cross-validated model on the test set.
            If scoring represents a single score, use a single string (see the SCORERS dictionary).
            If None, the estimator's score method is used.
        cv (int, cross-validation generator or an iterable): Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - integer, to specify the number of folds in a `(Stratified)KFold`,
            - An iterable yielding (train, test) splits as arrays of indices.
        refit (bool): Refit an estimator using the best found parameters on the whole dataset.
        error_score ('raise' or numeric): Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised. If a numeric value is given, FitFailedWarning is raised.
        task_type (str): 'classification' or 'regression'. Determines the default CV strategy.
    """

    _SCORERS = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'neg_mean_squared_error': lambda yt, yp: -mean_squared_error(yt, yp),
        'r2': jax_r2_score,
    }

    def __init__(
        self,
        estimator,
        param_grid,
        scoring='accuracy',
        cv=5,
        refit=False,
        error_score='raise',
        task_type='classification',
    ):
        """
        Initialize the GridSearchCV object.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.error_score = error_score
        self.task_type = task_type
        self._scorer_func = self._get_scorer(scoring)
        self.param_combinations = list(generate_param_combinations(self.param_grid))
        self.param_keys = list(self.param_combinations[0].keys())
        self.param_values = jnp.array(
            [list(p.values()) for p in self.param_combinations]
        )

    def _get_scorer(self, scoring):
        """
        Get the scoring function based on the scoring parameter.
        """
        if callable(scoring):
            return scoring
        elif isinstance(scoring, str) and scoring in self._SCORERS:
            return self._SCORERS[scoring]
        else:
            raise ValueError(
                f"Unsupported scoring: {scoring}. Expected one of {list(self._SCORERS.keys())} or callable."
            )

    def _get_cv_splitter(self, X, y):
        """
        Get the cross-validation split generator.
        """
        n_samples = X.shape[0]
        if callable(self.cv):
            return lambda: self.cv(X, y)
        elif isinstance(self.cv, int):
            return lambda: jax_kfold_split(n_samples, n_splits=self.cv)
        else:
            return lambda: self.cv

    def fit(self, X, y):
        """
        Run fit with all sets of parameters.

        Args:
            X (jnp.ndarray): Training vector, where n_samples is the number of samples and
                             n_features is the number of features.
            y (jnp.ndarray): Target relative to X for classification or regression;
                             None for unsupervised learning.

        Returns:
            self: object. Returns the instance itself.
        """
        X, y = jnp.asarray(X), jnp.asarray(y)
        n_candidates = len(self.param_combinations)
        results = defaultdict(list)

        cv_splitter = self._get_cv_splitter(X, y)()

        for i, params in enumerate(self.param_combinations):
            scores = []
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter):
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                estimator = copy.deepcopy(self.estimator)

                try:
                    if hasattr(estimator, 'set_params'):
                        estimator.set_params(**params)
                    else:
                        for k, v in params.items():
                            setattr(estimator, k, v)

                    estimator.fit(X_train, y_train)
                    y_pred = estimator.predict(X_test)
                    score = self._scorer_func(y_test, y_pred)
                except Exception as e:
                    if self.error_score == 'raise':
                        raise e
                    score = self.error_score
                    warnings.warn(f"Fit or score failed: {str(e)}", FitFailedWarning)
                scores.append(score)

            mean_score = jnp.nanmean(jnp.array(scores))
            results['mean_test_score'].append(mean_score)

        mean_test_scores = jnp.array(results['mean_test_score'])
        self.best_index_ = jnp.nanargmax(mean_test_scores)
        self.best_score_ = mean_test_scores[self.best_index_]

        best_params_values = jnp.take(self.param_values, self.best_index_, axis=0)
        self.best_params_ = dict(zip(self.param_keys, best_params_values))

        if self.refit:
            self.best_estimator_ = copy.deepcopy(self.estimator)
            try:
                if hasattr(self.best_estimator_, 'set_params'):
                    self.best_estimator_.set_params(**self.best_params_)
                else:
                    for k, v in self.best_params_.items():
                        setattr(self.best_estimator_, k, v)
                self.best_estimator_.fit(X, y)
            except Exception as e:
                warnings.warn(f"Refitting failed: {str(e)}", FitFailedWarning)
                self.best_estimator_ = None

        return self

    def predict(self, X):
        """
        Call predict on the estimator with the best found parameters.

        Only available if `refit=True` and the underlying estimator supports
        `predict`.

        Args:
            X (jnp.ndarray): Data to predict on. Must fulfill input requirements
                             of the underlying estimator.

        Returns:
            jnp.ndarray: Predicted values.
        """
        if not self.refit:
            raise NotFittedError("predict is not available when refit=False")
        if self.best_estimator_ is None:
            raise NotFittedError("predict is not available because refit failed.")
        X = jnp.asarray(X)
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        """
        Return the score on the given data, if the estimator has been refit.

        Uses the score defined by `scoring` where available, otherwise
        uses the `score` method of the underlying estimator.

        Args:
            X (jnp.ndarray): Input data, where n_samples is the number of samples and
                             n_features is the number of features.
            y (jnp.ndarray): True labels for X.

        Returns:
            float: Score of the estimator on the test data.
        """
        if not self.refit:
            raise NotFittedError("score is not available when refit=False")
        if self.best_estimator_ is None:
            raise NotFittedError("score is not available because refit failed.")
        X, y = jnp.asarray(X), jnp.asarray(y)
        y_pred = self.best_estimator_.predict(X)
        return self._scorer_func(y, y_pred)


# --- Custom Exceptions and Warnings ---


class FitFailedWarning(UserWarning):
    """Warning raised when an estimator fails to fit."""

    pass


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting."""

    pass
