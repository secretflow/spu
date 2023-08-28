# Copyright 2023 Ant Group Co., Ltd.
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

import os
import sys

import jax.numpy as jnp
import jax.random as random
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import sml.utils.emulation as emulation
from sml.naive_bayes.gnb import GaussianNB


def emul_SimpleGNB(mode: emulation.Mode.MULTIPROCESS):
    print("start gaussian naive bayes emulation.")

    def proc_fit(X1, y1, X2, y2, classes):
        model = GaussianNB(
            classes_=classes,
            var_smoothing=1e-9,
        )

        model.fit(X1, y1)
        y1_pred = model.predict(X)

        model.partial_fit(X2, y2)
        y2_pred = model.predict(X)

        return y1_pred, y2_pred

    try:
        # bandwidth and latency only work for docker mode
        print(os.getcwd())
        emulator = emulation.Emulator('3pc.json', mode, bandwidth=300, latency=20)
        emulator.up()
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

        X1_spu, y1_spu = emulator.seal(X1), emulator.seal(y1)
        X2_spu, y2_spu = emulator.seal(X2), emulator.seal(y2)

        y1_pred, y2_pred = emulator.run(proc_fit)(X1_spu, y1_spu, X2_spu, y2_spu, classes)
        result1 = (y == y1_pred).sum() / total_samples
        result2 = (y == y2_pred).sum() / total_samples

        print("gaussian naive bayes result:")
        print("Prediction accuracy with once fit: ", result1)
        print("Prediction accuracy with twice fits: ", result2)
        print()

        # Compare with sklearn
        model = SklearnGaussianNB()

        model.fit(X1, y1)
        y1_pred = model.predict(X)

        model.partial_fit(X2, y2)
        y2_pred = model.predict(X)

        sk_result1 = (y == y1_pred).sum() / total_samples
        sk_result2 = (y == y2_pred).sum() / total_samples

        print("sklearn gaussian naive bayes result:")
        print("Sklearn prediction accuracy with once fit: ", sk_result1)
        print("Sklearn prediction accuracy with twice fits: ", sk_result2)

        assert np.isclose(result1, sk_result1, atol=1e-4)
        assert np.isclose(result2, sk_result2, atol=1e-4)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_SimpleGNB(emulation.Mode.MULTIPROCESS)
