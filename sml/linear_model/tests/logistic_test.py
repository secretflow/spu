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
    @staticmethod
    def load_data(multi_class="binary"):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )
        # Create dataset
        if multi_class == "binary":
            X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        else:
            X, y = load_wine(return_X_y=True, as_frame=True)
        scalar = MinMaxScaler(feature_range=(-2, 2))
        cols = X.columns
        X = scalar.fit_transform(X)
        X = pd.DataFrame(X, columns=cols)
        return sim, X, y

    @staticmethod
    def proc(x, y, penalty, multi_class="binary"):
        class_labels = [0, 1] if multi_class == "binary" else [0, 1, 2]
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
            multi_class=multi_class,
            class_labels=class_labels,
        )

        model = model.fit(x, y)
        prob = model.predict_proba(x)
        pred = model.predict(x)
        return prob, pred

    # Binary classification
    def test_logistic(self):
        penalty_list = ["l1", "l2", "elasticnet"]
        print(f"penalty_list={penalty_list}")
        sim, X, y = self.load_data(multi_class="binary")

        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            # Run
            result = spsim.sim_jax(sim, self.proc, static_argnums=(2, 3))(
                X.values, y.values.reshape(-1, 1), penalty, "binary"
            )  # X, y should be two-dimension array
            # print("Predict result prob: ", result[0])
            # print("Predict result label: ", result[1])
            print(f"{penalty} ROC Score: {roc_auc_score(y.values, result[0])}")

    # Multi classification
    def test_logistic_multi_classification(self):
        sim, X, y = self.load_data(multi_class="ovr")

        # Run
        result = spsim.sim_jax(sim, self.proc, static_argnums=(2, 3))(
            X.values, y.values.reshape(-1, 1), "l2", "ovr"
        )  # X, y should be two-dimension array
        print(
            f"Multi classification OVR ROC Score: {roc_auc_score(y.values, result[0], multi_class='ovr')}"
        )


if __name__ == "__main__":
    unittest.main()
