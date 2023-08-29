# Copyright 2023 Ant Group Co., Ltd.
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

import os
import sys
import unittest

import jax.numpy as jnp
import numpy as np
from jax import random
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# Add the sml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from sml.naive_bayes.gnb import GaussianNB


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(" ========= start test of gnb package ========= \n")

        # 1. init sim
        cls.sim64 = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )
        config128 = spu_pb2.RuntimeConfig(
            protocol=spu_pb2.ProtocolKind.ABY3,
            field=spu_pb2.FieldType.FM128,
            fxp_fraction_bits=30,
        )
        cls.sim128 = spsim.Simulator(3, config128)

    def test_gnb(self):
        print("start test gnb method.")

        # Test fit and partial fit
        def proc(X1, y1, X2, y2, classes):
            model = GaussianNB(
                classes_=classes,
                var_smoothing=1e-6,
            )

            model.fit(X1, y1)
            y1_pred = model.predict(X)

            model.fit(X2, y2)
            y2_pred = model.predict(X)

            return y1_pred, y2_pred

        def test_precision(X1, y1, X2, y2, classes):
            model = GaussianNB(
                classes_=classes,
                var_smoothing=1e-6,
            )

            model.fit(X1, y1)
            theta1, var1 = model.theta_, model.var_

            model.fit(X2, y2)
            theta2, var2 = model.theta_, model.var_

            return theta1, var1, theta2, var2

        # Create a simple dataset
        partial = 0.5
        n_samples = 1000
        n_features = 100
        centers = 3
        X, y = datasets.make_blobs(
            n_samples=n_samples, n_features=n_features, centers=centers
        )
        classes = jnp.unique(y)
        assert len(classes) == centers, f'Retry or increase partial.'
        total_samples = len(y)
        split_idx = int(partial * len(y))
        X1, y1 = X[:split_idx], y[:split_idx]
        X2, y2 = X[split_idx:], y[split_idx:]

        # Run the simulation
        y1_pred, y2_pred = spsim.sim_jax(self.sim64, proc)(X1, y1, X2, y2, classes)
        result1 = (y == y1_pred).sum() / total_samples
        result2 = (y == y2_pred).sum() / total_samples

        # Run fit and partial_fit using sklearn
        sklearn_model = SklearnGaussianNB()
        sklearn_model.fit(X1, y1)
        y1_pred = sklearn_model.predict(X)
        sklearn_model.partial_fit(X2, y2)
        y2_pred = sklearn_model.predict(X)
        sk_result1 = (y == y1_pred).sum() / total_samples
        sk_result2 = (y == y2_pred).sum() / total_samples

        print("gaussian naive bayes result:")
        print("Prediction accuracy with once fit: ", result1)
        print("Prediction accuracy with twice fits: ", result2)
        print()
        print("sklearn gaussian naive bayes result:")
        print("Sklearn prediction accuracy with once fit: ", sk_result1)
        print("Sklearn prediction accuracy with twice fits: ", sk_result2)

        assert np.isclose(result1, sk_result1, atol=1e-4)
        assert np.isclose(result2, sk_result2, atol=1e-4)

        # Test precision of theta_ and var_
        theta1, var1, theta2, var2 = spsim.sim_jax(self.sim64, test_precision)(
            X1, y1, X2, y2, classes
        )
        sklearn_model = SklearnGaussianNB()
        sklearn_model.fit(X1, y1)
        sk_theta1, sk_var1 = sklearn_model.theta_, sklearn_model.var_
        sklearn_model.partial_fit(X2, y2)
        sk_theta2, sk_var2 = sklearn_model.theta_, sklearn_model.var_

        assert np.allclose(theta1, sk_theta1, rtol=1.0e-5, atol=1)
        assert np.allclose(var1, sk_var1, rtol=1.0e-5, atol=1)


if __name__ == "__main__":
    unittest.main()
