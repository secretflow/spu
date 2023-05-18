# Copyright 2021 Ant Group Co., Ltd.
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


from jax.example_libraries import stax
from jax.example_libraries.stax import (
    Conv,
    MaxPool,
    AvgPool,
    Flatten,
    Dense,
    Relu,
    Sigmoid,
    LogSoftmax,
    Softmax,
    BatchNorm,
)


# Network A
# Ref: https://eprint.iacr.org/2017/396.pdf
def secureml():
    nn_init, nn_apply = stax.serial(
        Flatten,
        Dense(128),
        Relu,
        Dense(128),
        Relu,
        Dense(10),
    )
    return nn_init, nn_apply


# Network B
# Ref: https://eprint.iacr.org/2017/452.pdf
def minionn():
    nn_init, nn_apply = stax.serial(
        Conv(out_chan=16, filter_shape=(5, 5), strides=(1, 1), padding='valid'),
        MaxPool(window_shape=(2, 2), strides=(2, 2)),
        Relu,
        Conv(out_chan=16, filter_shape=(5, 5), strides=(1, 1), padding='valid'),
        MaxPool(window_shape=(2, 2), strides=(2, 2)),
        Relu,
        Flatten,
        Dense(100),
        Relu,
        Dense(10),
    )
    return nn_init, nn_apply


# Network C
# Ref: https://eprint.iacr.org/2018/442.pdf
def lenet():
    nn_init, nn_apply = stax.serial(
        Conv(out_chan=20, filter_shape=(5, 5), strides=(1, 1), padding='valid'),
        MaxPool(window_shape=(2, 2), strides=(2, 2)),
        Relu,
        Conv(out_chan=50, filter_shape=(5, 5), strides=(1, 1), padding='valid'),
        MaxPool(window_shape=(2, 2), strides=(2, 2)),
        Relu,
        Flatten,
        Dense(500),
        Relu,
        Dense(10),
    )
    return nn_init, nn_apply


# Network D
# Ref: https://eprint.iacr.org/2017/1164.pdf
def chameleon():
    nn_init, nn_apply = stax.serial(
        Conv(out_chan=5, filter_shape=(5, 5), strides=(2, 2), padding='same'),
        Relu,
        Flatten,
        Dense(100),
        Relu,
        Dense(10),
    )
    return nn_init, nn_apply


def alexnet(num_class=10):
    nn_init, nn_apply = stax.serial(
        Conv(
            out_chan=96,
            filter_shape=(11, 11),
            strides=(4, 4),
            padding=((9, 9), (9, 9)),
        ),
        Relu,
        AvgPool(window_shape=(3, 3), strides=(2, 2)),
        BatchNorm(),
        Conv(
            out_chan=256,
            filter_shape=(5, 5),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        ),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(1, 1)),
        BatchNorm(),
        Conv(
            out_chan=384,
            filter_shape=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        ),
        Relu,
        Conv(
            out_chan=384,
            filter_shape=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        ),
        Relu,
        Conv(
            out_chan=256,
            filter_shape=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
        ),
        Relu,
        # Classifier for CIFAR10
        Flatten,
        Dense(256),
        Relu,
        Dense(256),
        Relu,
        Dense(10),
    )
    return nn_init, nn_apply


def vgg16(num_class=10):
    nn_init, nn_apply = stax.serial(
        Conv(
            out_chan=64, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        Conv(
            out_chan=64, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(2, 2)),
        Conv(
            out_chan=128, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        Conv(
            out_chan=128, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(2, 2)),
        Conv(
            out_chan=256, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        Conv(
            out_chan=256, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        Conv(
            out_chan=256, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(2, 2)),
        Conv(
            out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        Conv(
            out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        Conv(
            out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(2, 2)),
        Conv(
            out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        Conv(
            out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        Conv(
            out_chan=512, filter_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1))
        ),
        Relu,
        AvgPool(window_shape=(2, 2), strides=(2, 2)),
        # Classifier for CIFAR10
        Flatten,
        Dense(256),
        Relu,
        Dense(256),
        Relu,
        Dense(10),
    )
    return nn_init, nn_apply


def custom_model():
    nn_init, nn_apply = stax.serial(
        Conv(2, (1, 1)),
        AvgPool((2, 2), (1, 1)),
        Flatten,
        Dense(10),
    )
    return nn_init, nn_apply


def LR():
    nn_init, nn_apply = stax.serial(
        Flatten,
        Dense(10),
        Sigmoid,
    )
    return nn_init, nn_apply
