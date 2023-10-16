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

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as spsim

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from sml.linear_model.logistic import LogisticRegression


class UnitTests(unittest.TestCase):
    def test_logistic(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        penalty_list = ["l1", "l2", "elasticnet"]
        print(f"penalty_list={penalty_list}")

        # Test SGDClassifier
        def proc(x, y, penalty):
            model = LogisticRegression(
                epochs=1,
                learning_rate=0.1,
                batch_size=8,
                solver="sgd",
                penalty=penalty,
                sig_type="sr",
                C=1.0,
                l1_ratio=0.5,
                class_weight=None,
                multi_class="binary",
            )

            model = model.fit(x, y)

            prob = model.predict_proba(x)
            pred = model.predict(x)
            return prob, pred

        # Test Multi classification
        def proc2(x, y):
            model = LogisticRegression(
                epochs=1,
                learning_rate=0.1,
                batch_size=8,
                solver="sgd",
                penalty="l2",
                sig_type="sr",
                C=1.0,
                l1_ratio=0.5,
                class_weight=None,
                multi_class="ovr",
                class_labels=[0, 1, 2],
            )

            model = model.fit(x, y)

            prob = model.predict_proba(x)
            pred = model.predict(x)
            return prob, pred

        # Create dataset
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        scalar = MinMaxScaler(feature_range=(-2, 2))
        cols = X.columns
        X = scalar.fit_transform(X)
        X = pd.DataFrame(X, columns=cols)

        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            # Run
            result = spsim.sim_jax(sim, proc, static_argnums=(2,))(
                X.values, y.values.reshape(-1, 1), penalty
            )  # X, y should be two-dimension array
            # print("Predict result prob: ", result[0])
            # print("Predict result label: ", result[1])
            print(f"{penalty} ROC Score: {roc_auc_score(y.values, result[0])}")

        # Multi classification
        # dataset: wine
        X, y = load_wine(return_X_y=True, as_frame=True)
        scalar = MinMaxScaler(feature_range=(-2, 2))
        cols = X.columns
        X = scalar.fit_transform(X)
        X = pd.DataFrame(X, columns=cols)
        # Run
        result2 = spsim.sim_jax(sim, proc2)(
            X.values, y.values.reshape(-1, 1)
        )  # X, y should be two-dimension array
        print(result2[0].sum(axis=1))
        import jax.numpy as jnp

        print(jnp.allclose(1, result2[0].sum(axis=1)))
        print(
            f"Multi classification OVR ROC Score: {roc_auc_score(y.values, result2[0], multi_class='ovr')}"
        )


if __name__ == "__main__":
    unittest.main()
