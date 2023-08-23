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
    def proc(x, y):
        model = GaussianProcessClassifier(
            max_iter_predict=10
        )
        model.fit(x, y)

        pred = model.predict(x)
        return pred

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # load data
        x, y = load_iris(return_X_y=True)
        x = x[45:55, :]
        y = y[45:55]

        # mark these data to be protected in SPU
        x, y = emulator.seal(x, y)
        result = emulator.run(proc)(x, y)
        print("Accuracy: ", jnp.sum(result == y)/len(y))

    finally:
        emulator.down()

if __name__ == "__main__":
    emul_gpc(emulation.Mode.MULTIPROCESS)
