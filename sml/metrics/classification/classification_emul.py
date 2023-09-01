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
# See the License for the specifi

import os
import sys

import numpy as np

# add ops dir to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from sml.metrics.classification.classification import roc_auc_score


import sml.utils.emulation as emulation


# TODO: design the enumation framework, just like py.unittest
# all emulation action should begin with `emul_` (for reflection)
def emul_SGDClassifier(mode: emulation.Mode.MULTIPROCESS):
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        # Create dataset
        row = 10000
        y_true = np.random.randint(0, 2, (row,))
        y_pred = np.random.random((row,))

        # Run
        result = emulator.run(roc_auc_score)(
            y_true, y_pred
        )  # X, y should be two-dimension array
        print(result)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_SGDClassifier(emulation.Mode.MULTIPROCESS)
