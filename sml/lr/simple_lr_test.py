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

import unittest
import jax.numpy as jnp
import pandas as pd
import sys
import os
import spu.utils.simulation as spsim
import spu.spu_pb2 as spu_pb2
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from sml.lr.simple_lr import LogisticRegression


class UnitTests(unittest.TestCase):
    def test_simple(self):
        sim = spsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64
        )

        # Test SGDClassifier
        def proc(x, y):
            model = LogisticRegression(
                epochs=3,
                learning_rate=0.1,
                batch_size=8,
                solver='sgd',
                penalty='l2',
                sig_type='sr',
                l2_norm=1.0,
                class_weight=None,
                multi_class='binary',
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

        # Run
        result = spsim.sim_jax(sim, proc)(
            X.values, y.values.reshape(-1, 1)
        )  # X, y should be two-dimension array
        print("Predict result prob: ", result[0])
        print("Predict result label: ", result[1])

        print("ROC Score: ", roc_auc_score(y.values, result[0]))


if __name__ == "__main__":
    unittest.main()
