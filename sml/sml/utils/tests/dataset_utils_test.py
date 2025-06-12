# Copyright 2025 Ant Group Co., Ltd.
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

import unittest
import numpy as np

from sml.utils.dataset_utils import (
    load_open_source_datasets,
    load_mock_datasets,
    BI_CLASSIFICATION_OPEN_DATASETS,
    MULTI_CLASSIFICATION_OPEN_DATASETS,
    REGRESSION_OPEN_DATASETS,
    _supported_glm_dist,
)


class TestDatasetUtils(unittest.TestCase):
    """Test dataset utility functions."""

    def test_load_open_source_datasets_with_split(self):
        """Test loading open source datasets with train/test split."""
        # Test binary classification datasets
        for dataset_name in BI_CLASSIFICATION_OPEN_DATASETS:
            with self.subTest(dataset=dataset_name):
                print(f"Loading open source dataset with split: {dataset_name}")
                result = load_open_source_datasets(
                    name=dataset_name, test_size=0.2, need_split_train_test=True
                )
                # Verify return length
                self.assertEqual(len(result), 4)
                x_train, x_test, y_train, y_test = result
                # Verify shape consistency
                self.assertEqual(x_train.shape[0], y_train.shape[0])
                self.assertEqual(x_test.shape[0], y_test.shape[0])
                self.assertEqual(x_train.shape[1], x_test.shape[1])
                # Verify data types
                self.assertIsInstance(x_train, np.ndarray)
                self.assertIsInstance(y_train, np.ndarray)

        # Test multi-class classification datasets
        for dataset_name in MULTI_CLASSIFICATION_OPEN_DATASETS:
            with self.subTest(dataset=dataset_name):
                print(f"Loading open source dataset with split: {dataset_name}")
                result = load_open_source_datasets(
                    name=dataset_name, test_size=0.2, need_split_train_test=True
                )
                # Verify return length
                self.assertEqual(len(result), 4)
                x_train, x_test, y_train, y_test = result
                # Verify shape consistency
                self.assertEqual(x_train.shape[0], y_train.shape[0])
                self.assertEqual(x_test.shape[0], y_test.shape[0])
                self.assertEqual(x_train.shape[1], x_test.shape[1])
                # Verify data types
                self.assertIsInstance(x_train, np.ndarray)
                self.assertIsInstance(y_train, np.ndarray)

        # Test regression datasets
        for dataset_name in REGRESSION_OPEN_DATASETS:
            with self.subTest(dataset=dataset_name):
                print(f"Loading open source dataset with split: {dataset_name}")
                result = load_open_source_datasets(
                    name=dataset_name, test_size=0.2, need_split_train_test=True
                )
                # Verify return length
                self.assertEqual(len(result), 4)
                x_train, x_test, y_train, y_test = result
                # Verify shape consistency
                self.assertEqual(x_train.shape[0], y_train.shape[0])
                self.assertEqual(x_test.shape[0], y_test.shape[0])
                self.assertEqual(x_train.shape[1], x_test.shape[1])
                # Verify data types
                self.assertIsInstance(x_train, np.ndarray)
                self.assertIsInstance(y_train, np.ndarray)

        # Test GLM datasets
        # for mtpl2, support 3 distributions: Poisson, Gamma, Tweedie, but load the data is time-consuming, so we skip it.
        # for dist in ["Tweedie"]:
        #     dataset_name = f"mtpl2_{dist}"
        #     with self.subTest(dataset=dataset_name):
        #         print(f"Loading open source dataset with split: {dataset_name}")
        #         result = load_open_source_datasets(
        #             name=dataset_name, test_size=0.2, need_split_train_test=True
        #         )
        #         # Verify return length (GLM has 6 returns: x_train, x_test, y_train, y_test, w_train, w_test)
        #         self.assertEqual(len(result), 6)
        #         x_train, x_test, y_train, y_test, w_train, w_test = result
        #         # Verify shape consistency
        #         self.assertEqual(x_train.shape[0], y_train.shape[0])
        #         self.assertEqual(x_train.shape[0], w_train.shape[0])
        #         self.assertEqual(x_test.shape[0], y_test.shape[0])
        #         self.assertEqual(x_test.shape[0], w_test.shape[0])
        #         self.assertEqual(x_train.shape[1], x_test.shape[1])
        #         # Verify data types
        #         self.assertIsInstance(x_train, np.ndarray)
        #         self.assertIsInstance(y_train, np.ndarray)
        #         self.assertIsInstance(w_train, np.ndarray)

    def test_load_mock_datasets_classification(self):
        """Test loading mock datasets for classification tasks."""
        # Test binary classification with split
        print("Loading mock dataset: binary classification with split")
        result = load_mock_datasets(
            n_samples=100,
            task_type="bi_classification",
            n_features=5,
            need_split_train_test=True,
        )
        # Verify return length
        self.assertEqual(len(result), 4)
        x_train, x_test, y_train, y_test = result
        # Verify shape consistency
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])
        self.assertEqual(x_train.shape[1], 5)
        self.assertEqual(x_test.shape[1], 5)
        # Verify data types
        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)

        # Test binary classification without split
        print("Loading mock dataset: binary classification without split")
        result = load_mock_datasets(
            n_samples=100,
            task_type="bi_classification",
            n_features=5,
            need_split_train_test=False,
        )
        # Verify return length
        self.assertEqual(len(result), 2)
        X, y = result
        # Verify shape consistency
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(X.shape[1], 5)
        # Verify data types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

        # Test multi-class classification with split
        print("Loading mock dataset: multi-class classification with split")
        result = load_mock_datasets(
            n_samples=100,
            task_type="multi_classification",
            n_features=10,
            n_classes=3,
            n_informative=4,
            need_split_train_test=True,
        )
        # Verify return length
        self.assertEqual(len(result), 4)
        x_train, x_test, y_train, y_test = result
        # Verify shape consistency
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])
        self.assertEqual(x_train.shape[1], 10)
        self.assertEqual(x_test.shape[1], 10)
        # Verify data types
        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        # Verify class values
        self.assertTrue(np.all(np.unique(y_train) < 3))
        self.assertTrue(np.all(np.unique(y_test) < 3))

        # Test multi-class classification without split
        print("Loading mock dataset: multi-class classification without split")
        result = load_mock_datasets(
            n_samples=100,
            task_type="multi_classification",
            n_features=10,
            n_classes=3,
            n_informative=4,
            need_split_train_test=False,
        )
        # Verify return length
        self.assertEqual(len(result), 2)
        X, y = result
        # Verify shape consistency
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(X.shape[1], 10)
        # Verify data types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        # Verify class values
        self.assertTrue(np.all(np.unique(y) < 3))

    def test_load_mock_datasets_regression(self):
        """Test loading mock datasets for regression tasks."""
        # Test regression with split
        print("Loading mock dataset: regression with split")
        result = load_mock_datasets(
            n_samples=100,
            task_type="regression",
            n_features=5,
            need_split_train_test=True,
        )
        # Verify return length
        self.assertEqual(len(result), 4)
        x_train, x_test, y_train, y_test = result
        # Verify shape consistency
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])
        self.assertEqual(x_train.shape[1], 5)
        self.assertEqual(x_test.shape[1], 5)
        # Verify data types
        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)

        # Test regression without split
        print("Loading mock dataset: regression without split")
        result = load_mock_datasets(
            n_samples=100,
            task_type="regression",
            n_features=5,
            need_split_train_test=False,
        )
        # Verify return length
        self.assertEqual(len(result), 2)
        X, y = result
        # Verify shape consistency
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(X.shape[1], 5)
        # Verify data types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_load_mock_datasets_clustering(self):
        """Test loading mock datasets for clustering tasks."""
        print("Loading mock dataset: clustering")
        result = load_mock_datasets(
            n_samples=100,
            task_type="clustering",
            n_features=5,
            centers=3,
        )
        # Verify return length
        self.assertEqual(len(result), 2)
        X, y = result
        # Verify shape consistency
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(X.shape[1], 5)
        # Verify data types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        # Verify cluster values
        self.assertTrue(np.all(np.unique(y) < 3))

    def test_load_mock_datasets_metric(self):
        """Test loading mock datasets for metric tasks."""
        metric_types = [
            "bi_classification_rank",
            "bi_classification",
            "multi_classification",
            "regression",
        ]

        for metric_type in metric_types:
            with self.subTest(metric_type=metric_type):
                print(f"Loading mock dataset: metric {metric_type}")
                result = load_mock_datasets(
                    n_samples=100,
                    task_type="metric",
                    metric_type=metric_type,
                )
                # Verify return length
                self.assertEqual(len(result), 2)
                y_true, y_pred = result
                # Verify shape consistency
                self.assertEqual(y_true.shape[0], y_pred.shape[0])
                self.assertEqual(y_true.shape[0], 100)
                # Verify data types
                self.assertIsInstance(y_true, np.ndarray)
                self.assertIsInstance(y_pred, np.ndarray)

        # Test GLM metric
        for dist in _supported_glm_dist:
            with self.subTest(distribution=dist):
                print(f"Loading mock dataset: metric glm {dist}")
                result = load_mock_datasets(
                    n_samples=100,
                    task_type="metric",
                    metric_type="glm",
                    distribution=dist,
                )
                # Verify return length
                self.assertEqual(len(result), 2)
                y_true, y_pred = result
                # Verify shape consistency
                self.assertEqual(y_true.shape[0], y_pred.shape[0])
                self.assertEqual(y_true.shape[0], 100)
                # Verify data types
                self.assertIsInstance(y_true, np.ndarray)
                self.assertIsInstance(y_pred, np.ndarray)
                # For GLM, values should be positive
                self.assertTrue(np.all(y_true > 0))
                self.assertTrue(np.all(y_pred > 0))

    def test_load_mock_datasets_decomposition(self):
        """Test loading mock datasets for decomposition tasks."""
        print("Loading mock dataset: decomposition")
        result = load_mock_datasets(
            n_samples=100,
            task_type="decomposition",
            n_factors=2,
        )
        # Verify return is a single matrix
        self.assertIsInstance(result, np.ndarray)
        # Verify shape
        self.assertEqual(result.shape[0], 100)
        self.assertEqual(result.shape[1], 20)  # 10 * n_factors


if __name__ == "__main__":
    unittest.main()
