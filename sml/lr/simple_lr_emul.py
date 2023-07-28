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

import jax.numpy as jnp
import pandas as pd
import sys
import os
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from sml.lr.simple_lr import SGDClassifier
import sml.utils.emulation as emulation

# TODO: design the enumation framework, just like py.unittest
# all emulation action should begin with `emul_` (for reflection)
def emul_SGDClassifier(mode: emulation.Mode.MULTIPROCESS):
    def proc(x, y):
        model = SGDClassifier(
            epochs=3,
            learning_rate=0.1,
            batch_size=8,
            penalty='l2',
            sig_type='sr',
            l2_norm=1.0,
            class_weight=None,
            multi_class='ovr'
        )
        return model.fit(x, y).predict_proba(x)

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
        X,y = emulator.seal(X, y)
        
        # Run
        result = emulator.run(proc)(X.values, y.values.reshape(-1, 1))  # X, y should be two-dimension array
        print(result)
        print("Predict result: ", result)
        print("ROC Score: ", roc_auc_score(y.values, result))

    finally:
        emulator.down()

if __name__ == "__main__":
    emul_SGDClassifier(emulation.Mode.MULTIPROCESS)
