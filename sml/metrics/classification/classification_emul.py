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
import jax.numpy as jnp
from sklearn import metrics

# add ops dir to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import sml.utils.emulation as emulation
from sml.metrics.classification.classification import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


# TODO: design the enumation framework, just like py.unittest
# all emulation action should begin with `emul_` (for reflection)
def emul_auc(mode: emulation.Mode.MULTIPROCESS):
    # Create dataset
    row = 10000
    y_true = np.random.randint(0, 2, (row,))
    y_pred = np.random.random((row,))

    # Run
    result = emulator.run(roc_auc_score)(
        y_true, y_pred
    )  # X, y should be two-dimension array
    print(result)


def emul_Classification(mode: emulation.Mode.MULTIPROCESS):
    def proc(y_true, y_pred, average='binary', labels=None, pos_label=1, transform=1):
        f1 = f1_score(
            y_true,
            y_pred,
            average=average,
            labels=labels,
            pos_label=pos_label,
            transform=transform,
        )
        precision = precision_score(
            y_true,
            y_pred,
            average=average,
            labels=labels,
            pos_label=pos_label,
            transform=transform,
        )
        recall = recall_score(
            y_true,
            y_pred,
            average=average,
            labels=labels,
            pos_label=pos_label,
            transform=transform,
        )
        accuracy = accuracy_score(y_true, y_pred)
        return f1, precision, recall, accuracy

    def sklearn_proc(y_true, y_pred, average='binary', labels=None, pos_label=1):
        f1 = metrics.f1_score(
            y_true, y_pred, average=average, labels=labels, pos_label=pos_label
        )
        precision = metrics.precision_score(
            y_true, y_pred, average=average, labels=labels, pos_label=pos_label
        )
        recall = metrics.recall_score(
            y_true, y_pred, average=average, labels=labels, pos_label=pos_label
        )
        accuracy = metrics.accuracy_score(y_true, y_pred)
        return f1, precision, recall, accuracy

    def check(spu_result, sk_result):
        for pair in zip(spu_result, sk_result):
            np.testing.assert_allclose(pair[0], pair[1], rtol=1, atol=1e-5)

    # Test binary
    y_true = jnp.array([0, 1, 1, 0, 1, 1])
    y_pred = jnp.array([0, 0, 1, 0, 1, 1])
    spu_result = emulator.run(proc)(y_true, y_pred, average=None, labels=[0, 1, 2])
    sk_result = sklearn_proc(y_true, y_pred, average=None, labels=[0, 1, 2])
    check(spu_result, sk_result)

    # Test multiclass
    y_true = jnp.array([0, 1, 1, 0, 2, 1])
    y_pred = jnp.array([0, 0, 1, 0, 2, 1])
    spu_result = emulator.run(proc)(y_true, y_pred, average=None, labels=[0, 1, 2])
    sk_result = sklearn_proc(y_true, y_pred, average=None, labels=[0, 1, 2])
    check(spu_result, sk_result)


if __name__ == "__main__":
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC,
            emulation.Mode.MULTIPROCESS,
            bandwidth=300,
            latency=20,
        )
        emulator.up()
        emul_auc(emulation.Mode.MULTIPROCESS)
        emul_Classification(emulation.Mode.MULTIPROCESS)
    finally:
        emulator.down()
