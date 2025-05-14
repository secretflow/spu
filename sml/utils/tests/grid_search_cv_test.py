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
import os
import sys
import unittest

import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import spu.libspu as libspu
import spu.utils.simulation as spsim
from sml.ensemble.adaboost import AdaBoostClassifier
from sml.ensemble.forest import RandomForestClassifier
from sml.gaussian_process._gpc import GaussianProcessClassifier
from sml.linear_model.glm import _GeneralizedLinearRegressor
from sml.linear_model.logistic import LogisticRegression
from sml.linear_model.pla import Perceptron
from sml.linear_model.quantile import QuantileRegressor
from sml.linear_model.ridge import Ridge
from sml.linear_model.sgd_classifier import SGDClassifier
from sml.naive_bayes.gnb import GaussianNB
from sml.neighbors.knn import KNNClassifer
from sml.svm.svm import SVM
from sml.tree.tree import DecisionTreeClassifier
from sml.utils.grid_search_cv import GridSearchCV


class ComprehensiveGridSearchTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sim = spsim.Simulator.simple(
            3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64
        )
        cls.random_seed = 42
        cls.n_samples = 60
        cls.n_features = 8
        cls.n_classes_binary = 2
        cls.n_classes_multi = 3
        cls.cv_folds = 2

        cls.X_clf_bin, cls.y_clf_bin = make_classification(
            n_samples=cls.n_samples,
            n_features=cls.n_features,
            n_informative=4,
            n_redundant=1,
            n_classes=cls.n_classes_binary,
            random_state=cls.random_seed,
        )
        cls.y_clf_bin = jnp.array(cls.y_clf_bin)
        cls.y_clf_bin_reshaped = cls.y_clf_bin.reshape(-1, 1)

        cls.y_clf_bin_negpos = jnp.where(cls.y_clf_bin == 0, -1, 1)
        cls.y_clf_bin_negpos_reshaped = cls.y_clf_bin_negpos.reshape(-1, 1)

        cls.X_clf_multi, cls.y_clf_multi = make_classification(
            n_samples=cls.n_samples,
            n_features=cls.n_features,
            n_informative=5,
            n_redundant=1,
            n_classes=cls.n_classes_multi,
            n_clusters_per_class=1,
            random_state=cls.random_seed,
        )
        cls.y_clf_multi = jnp.array(cls.y_clf_multi)
        cls.y_clf_multi_reshaped = cls.y_clf_multi.reshape(-1, 1)

        from sml.preprocessing.preprocessing import KBinsDiscretizer

        binner = KBinsDiscretizer(n_bins=2, strategy='uniform')
        cls.X_clf_bin_binary_features = binner.fit_transform(cls.X_clf_bin)

        cls.X_reg, cls.y_reg = make_regression(
            n_samples=cls.n_samples,
            n_features=cls.n_features,
            n_informative=5,
            noise=0.5,
            random_state=cls.random_seed,
        )
        cls.y_reg_reshaped = cls.y_reg.reshape(-1, 1)

    def _run_test(
        self,
        model_name,
        estimator,
        param_grid,
        X,
        y,
        scoring,
        task_type,
        refit=False,
        cv_type='iterable',
    ):
        print(f"\n--- Testing GridSearchCV with {model_name} ---")

        if cv_type == 'iterable':
            if task_type == 'classification':
                skf = StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=42
                )
                cv_splits = [
                    (jnp.array(train_idx), jnp.array(test_idx))
                    for train_idx, test_idx in skf.split(X, y)
                ]
            else:
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                cv_splits = [
                    (jnp.array(train_idx), jnp.array(test_idx))
                    for train_idx, test_idx in kf.split(X)
                ]
            cv = cv_splits
        elif cv_type == 'int':
            cv = self.cv_folds
        else:
            raise ValueError("cv_type must be 'iterable' or 'int'")

        def run_grid_search_spu(X_spu, y_spu):
            grid_search = GridSearchCV(
                estimator=copy.deepcopy(estimator),
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                refit=refit,
                task_type=task_type,
            )
            grid_search.fit(X_spu, y_spu)
            return grid_search.best_score_, grid_search.best_params_

        spu_best_score, spu_best_param = spsim.sim_jax(self.sim, run_grid_search_spu)(
            X, y
        )
        print(f"SPU Best CV Score ({scoring}): {spu_best_score}")
        print(f"SPU Best Params: {spu_best_param}")

        grid_search_plain = GridSearchCV(
            estimator=copy.deepcopy(estimator),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            refit=refit,
            task_type=task_type,
        )
        grid_search_plain.fit(X, y)
        plain_best_score = grid_search_plain.best_score_
        print(f"Plaintext Best CV Score ({scoring}): {plain_best_score}")
        print(f"Plaintext Best Params: {grid_search_plain.best_params_}")

        np.testing.assert_allclose(
            spu_best_score, plain_best_score, rtol=1e-2, atol=1e-2
        )
        print(f"--- {model_name} Test Passed ---")

        if refit:
            X_test = X[:10]  # Use a subset for testing
            y_test = y[:10]
            spu_pred = grid_search_plain.predict(X_test)
            spu_score = grid_search_plain.score(X_test, y_test)
            print(f"SPU Prediction: {spu_pred}")
            print(f"SPU Score: {spu_score}")

    @unittest.skip(
        "For logistic, this algorithm neither supports the set_params method nor allows internal parameters to be updated through assignment."
    )
    def test_gridsearch_logistic(self):
        estimator = LogisticRegression(epochs=3, batch_size=16, class_labels=[0, 1])
        param_grid = {'learning_rate': [0.01, 0.1, 0.05], 'C': [1.0, 2.0, 5.0]}
        self._run_test(
            "LogisticRegression with cv",
            estimator,
            param_grid,
            self.X_clf_bin,
            self.y_clf_bin_reshaped,
            'accuracy',
            'classification',
            refit=True,
        )

    def test_gridsearch_knn(self):
        estimator = KNNClassifer(n_classes=self.n_classes_binary)
        param_grid = {'n_neighbors': [2, 3, 4, 5]}
        self._run_test(
            "KNNClassifier with cv as iterable",
            estimator,
            param_grid,
            self.X_clf_bin,
            self.y_clf_bin,
            'accuracy',
            'classification',
            refit=True,
        )

    def test_gridsearch_gnb(self):
        classes = jnp.unique(self.y_clf_bin)
        estimator = GaussianNB(classes_=classes, var_smoothing=1e-7)
        param_grid = {'var_smoothing': [1e-6, 2e-6, 1e-5]}
        self._run_test(
            "GaussianNB with cv as int",
            estimator,
            param_grid,
            self.X_clf_bin,
            self.y_clf_bin,
            'accuracy',
            'classification',
            refit=True,
            cv_type='int',
        )

    def test_gridsearch_perceptron(self):
        estimator = Perceptron(max_iter=10)
        param_grid = {'alpha': [0.0001, 0.001], 'eta0': [0.01, 0.1, 1.0]}
        self._run_test(
            "Perceptron with cv as iterable",
            estimator,
            param_grid,
            self.X_clf_bin,
            self.y_clf_bin_negpos_reshaped,
            'accuracy',
            'classification',
            refit=True,
        )

    def test_gridsearch_svm(self):
        estimator = SVM(max_iter=10, C=1.0)
        param_grid = {'C': [0.5, 1.0, 5.0]}
        self._run_test(
            "SVM with cv as int",
            estimator,
            param_grid,
            self.X_clf_bin,
            self.y_clf_bin_negpos,
            'accuracy',
            'classification',
            refit=True,
            cv_type='int',
        )

    @unittest.skip("GPC is often slow to settings")
    def test_gridsearch_gpc(self):
        estimator = GaussianProcessClassifier(
            max_iter_predict=5, n_classes_=self.n_classes_binary
        )
        param_grid = {'max_iter_predict': [1, 3, 5]}
        self._run_test(
            "GaussianProcessClassifier with cv as iterable",
            estimator,
            param_grid,
            self.X_clf_bin,
            self.y_clf_bin,
            'accuracy',
            'classification',
            refit=True,
        )

    @unittest.skip("SGDClassifier needs predict() added first")
    def test_gridsearch_sgdclassifier(self):
        estimator = SGDClassifier(
            epochs=3,
            learning_rate=0.1,
            batch_size=16,
            reg_type='logistic',
            penalty='l2',
        )
        param_grid = {'learning_rate': [0.1, 0.05], 'l2_norm': [0.01, 0.1]}
        self._run_test(
            "SGDClassifier with cv as int",
            estimator,
            param_grid,
            self.X_clf_bin,
            self.y_clf_bin_reshaped,
            'accuracy',
            'classification',
            refit=True,
            cv_type='int',
        )

    @unittest.skip("FIXME: Test it when we support revealing the best model from SPU during the program.")
    def test_gridsearch_decisiontree(self):
        estimator = DecisionTreeClassifier(
            max_depth=3,
            n_labels=self.n_classes_binary,
            criterion='gini',
            splitter='best',
        )
        param_grid = {'max_depth': [2, 3, 4]}
        self._run_test(
            "DecisionTreeClassifier with cv as iterable",
            estimator,
            param_grid,
            self.X_clf_bin_binary_features,
            self.y_clf_bin,
            'accuracy',
            'classification',
            refit=True,
        )

    @unittest.skip("FIXME: Test it when we support revealing the best model from SPU during the program.")
    def test_gridsearch_randomforest(self):
        estimator = RandomForestClassifier(
            n_estimators=3,
            max_depth=3,
            n_labels=self.n_classes_binary,
            criterion='gini',
            splitter='best',
            max_features=0.5,
            bootstrap=False,
            max_samples=None,
        )
        param_grid = {'max_depth': [2, 3], 'n_estimators': [2, 4]}
        self._run_test(
            "RandomForestClassifier with cv",
            estimator,
            param_grid,
            self.X_clf_bin_binary_features,
            self.y_clf_bin,
            'accuracy',
            'classification',
            refit=True,
        )

    @unittest.skip("FIXME: Test it when we support revealing the best model from SPU during the program.")
    def test_gridsearch_adaboost(self):
        base_estimator = DecisionTreeClassifier(
            max_depth=1,
            n_labels=self.n_classes_binary,
            criterion='gini',
            splitter='best',
        )
        estimator = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=3,
            learning_rate=1.0,
            algorithm='discrete',
        )
        param_grid = {'n_estimators': [2, 4]}
        self._run_test(
            "AdaBoostClassifier with cv as iterable",
            estimator,
            param_grid,
            self.X_clf_bin_binary_features,
            self.y_clf_bin,
            'accuracy',
            'classification',
            refit=True,
        )

    def test_gridsearch_ridge(self):
        estimator = Ridge(solver='cholesky')
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        self._run_test(
            "Ridge with cv as int",
            estimator,
            param_grid,
            self.X_reg,
            self.y_reg_reshaped,
            'r2',
            'regression',
            refit=True,
            cv_type='int',
        )

    def test_gridsearch_glm(self):
        estimator = _GeneralizedLinearRegressor(max_iter=10)
        param_grid = {'alpha': [0.0, 0.1, 0.2]}
        self._run_test(
            "GeneralizedLinearRegressor with cv as iterable",
            estimator,
            param_grid,
            self.X_reg,
            self.y_reg,
            'neg_mean_squared_error',
            'regression',
            refit=True,
        )

    @unittest.skip(
        "QuantileRegressor requires simplex solver, may be slow/complex for basic test"
    )
    def test_gridsearch_quantile(self):
        estimator = QuantileRegressor(max_iter=20, lr=0.05)
        param_grid = {'quantile': [0.25, 0.5, 0.75], 'alpha': [0.1, 0.5]}
        self._run_test(
            "QuantileRegressor with cv as int",
            estimator,
            param_grid,
            self.X_reg,
            self.y_reg,
            'r2',
            'regression',
            refit=True,
            cv_type='int',
        )


if __name__ == "__main__":
    unittest.main()
