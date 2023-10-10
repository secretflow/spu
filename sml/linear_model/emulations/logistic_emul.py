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

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import sml.utils.emulation as emulation
from sml.linear_model.logistic import LogisticRegression


def emul_LogisticRegression(mode: emulation.Mode.MULTIPROCESS):
    penalty_list = ["l1", "l2", "elasticnet"]
    print(f"penalty_list={penalty_list}")

    # Test SGDClassifier
    def proc(x, y, penalty):
        model = LogisticRegression(
            epochs=3,
            learning_rate=0.1,
            batch_size=8,
            solver="sgd",
            penalty=penalty,
            sig_type="sr",
            l2_norm=1.0,
            class_weight=None,
            multi_class="binary",
        )

        model = model.fit(x, y)

        prob = model.predict_proba(x)
        pred = model.predict(x)
        return prob, pred

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Create dataset
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        scalar = MinMaxScaler(feature_range=(-2, 2))
        cols = X.columns
        X = scalar.fit_transform(X)
        X = pd.DataFrame(X, columns=cols)

        # mark these data to be protected in SPU
        X_spu, y_spu = emulator.seal(
            X.values, y.values.reshape(-1, 1)
        )  # X, y should be two-dimension array

        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            # Run
            result = emulator.run(proc, static_argnums=(2,))(X_spu, y_spu, penalty)
            # print("Predict result prob: ", result[0])
            # print("Predict result label: ", result[1])
            print(f"{penalty} ROC Score: {roc_auc_score(y.values, result[0])}")

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_LogisticRegression(emulation.Mode.MULTIPROCESS)
