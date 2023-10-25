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
from sklearn.datasets import load_iris

# Add the library directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import sml.utils.emulation as emulation
from sml.gaussian_process._gpc import GaussianProcessClassifier


def emul_gpc(mode: emulation.Mode.MULTIPROCESS):
    def proc(x, y, x_pred):
        model = GaussianProcessClassifier(max_iter_predict=10, n_classes_=3)
        model.fit(x, y)

        pred = model.predict(x_pred)
        return pred

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            "sml/gaussian_process/emulations/3pc.json", mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Create dataset
        x, y = load_iris(return_X_y=True)

        idx = list(range(45, 55)) + list(range(100, 105))
        prd_idx = list(range(0, 5)) + list(range(55, 60)) + list(range(110, 115))
        x_pred = x[prd_idx, :]
        y_pred = y[prd_idx]
        x = x[idx, :]
        y = y[idx]

        # mark these data to be protected in SPU
        x, y, x_pred = emulator.seal(x, y, x_pred)
        result = emulator.run(proc)(x, y, x_pred)

        print(result)
        print(y_pred)
        print("Accuracy: ", jnp.sum(result == y_pred) / len(y_pred))

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_gpc(emulation.Mode.MULTIPROCESS)
