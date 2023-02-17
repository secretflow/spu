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

# from sklearn.metrics import roc_auc_score, explained_variance_score
import sml.utils.emulation as emulation

from sml.linear_model.simple_sgd import SGDClassifier


# TODO: design the enumation framework, just like py.unittest
# all emulation action should begin with `emul_` (for reflection)
def emul_SGDClassifier(mode: emulation.Mode.MULTIPROCESS):
    def proc(x1, x2, y):
        model = SGDClassifier(
            epochs=1,
            learning_rate=0.1,
            batch_size=1024,
            reg_type='logistic',
            penalty='None',
            l2_norm=0.0,
        )

        x = jnp.concatenate((x1, x2), axis=1)
        y = y.reshape((y.shape[0], 1))

        return model.fit(x, y).predict_proba(x)

    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()
        (x1, x2), y = emulator.prepare_dataset(emulation.DATASET_MOCK_REGRESSION_BASIC)
        result = emulator.run(proc)(x1, x2, y)
        print(result)
    finally:
        emulator.down()


if __name__ == "__main__":
    emul_SGDClassifier(emulation.Mode.MULTIPROCESS)
