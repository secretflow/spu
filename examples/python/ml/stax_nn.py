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

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run //examples/python/ml:stax_nn

import json
import time
import math

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from jax.example_libraries import optimizers, stax
from keras.datasets import cifar10
from datetime import datetime


import spu.binding.util.distributed as ppd
from examples.python.utils.stax_models import (
    LR,
    alexnet,
    chamelon,
    custom_model,
    lenet,
    minionn,
    secureml,
    vgg16,
)

from sklearn.metrics import accuracy_score

# Follows https://arxiv.org/pdf/2107.00501.pdf Appendix C.
DEFAULT_LEARNING_RATE = 0.01
# Suggested to be 150.
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 128


def train(
    train_x,
    train_y,
    init_fun,
    predict_fun,
    learning_rate,
    epochs,
    batch_size,
    run_on_spu,
):
    # Model Initialization
    key = jax.random.PRNGKey(42)
    input_shape = tuple(
        [-1 if idx == 0 else i for idx, i in enumerate(list(train_x.shape))]
    )
    _, params_init = init_fun(key, input_shape)
    opt_init, opt_update, get_params = optimizers.momentum(learning_rate, 0.9)
    opt_state = opt_init(params_init)

    def update_model(state, imgs, labels, i):
        def ce_loss(y, label):
            return -jnp.mean(jnp.sum(label * stax.logsoftmax(y), axis=1))

        def loss_func(params):
            y = predict_fun(params, imgs)
            return ce_loss(y, labels), y

        grad_fn = jax.value_and_grad(loss_func, has_aux=True)
        (loss, y), grads = grad_fn(get_params(state))
        return opt_update(i, grads, state)

    def identity(x):
        return x

    update_model_jit = jax.jit(update_model)

    print('Start trainning...')
    for i in range(1, epochs + 1):
        imgs_batchs = jnp.array_split(train_x, len(train_x) / batch_size, axis=0)
        labels_batchs = jnp.array_split(train_y, len(train_y) / batch_size, axis=0)

        for batch_idx, (batch_images, batch_labels) in enumerate(
            zip(imgs_batchs, labels_batchs)
        ):
            print(
                f'{datetime.now().time()} Epoch: {i}/{epochs}  Batch: {batch_idx}/{math.floor(len(train_x) / batch_size)}'
            )
            if run_on_spu:
                batch_images = ppd.device("P1")(identity)(batch_images)
                batch_labels = ppd.device("P2")(identity)(batch_labels)
                opt_state = ppd.device("SPU")(update_model)(
                    opt_state, batch_images, batch_labels, i
                )
            else:
                opt_state = update_model_jit(opt_state, batch_images, batch_labels, i)
    if run_on_spu:
        opt_state = ppd.get(opt_state)
    return get_params(opt_state)


def get_datasets(name='mnist'):
    """Load MNIST train and test datasets into memory."""
    if name == 'cifar10':
        train_ds, test_ds = cifar10.load_data()
        (train_x, train_y), (test_imgs, test_labels) = train_ds, test_ds
        train_x = np.float32(train_x) / 255.0
        train_y = np.squeeze(train_y)
        test_imgs = np.float32(test_imgs) / 255.0
        test_labels = np.squeeze(test_labels)

        return (train_x, train_y), (test_imgs, test_labels)
    ds_builder = tfds.builder(name)
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = np.float32(train_ds['image']) / 255.0
    test_ds['image'] = np.float32(test_ds['image']) / 255.0
    return train_ds, test_ds


def train_secureml(run_on_spu: bool = False):
    train_ds, test_ds = get_datasets('mnist')
    train_x, train_y = train_ds['image'], train_ds['label']
    train_y = jax.nn.one_hot(train_y, 10)

    # Hyper-parameters
    learning_rate = DEFAULT_LEARNING_RATE
    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE

    init_fun, predict_fun = secureml()

    start = time.perf_counter()
    params = train(
        train_x,
        train_y,
        init_fun,
        predict_fun,
        learning_rate,
        epochs,
        batch_size,
        run_on_spu,
    )
    end = time.perf_counter()

    print(f"train(cpu) elapsed time: {end - start:0.4f} seconds")
    test_x, test_y = test_ds['image'], test_ds['label']
    predict_y = predict_fun(params, test_x)

    print(f'accuracy(cpu): {accuracy_score(np.argmax(predict_y, axis=1),test_y)}')


def train_minionn(run_on_spu: bool = False):
    train_ds, test_ds = get_datasets('mnist')
    train_x, train_y = train_ds['image'], train_ds['label']
    train_y = jax.nn.one_hot(train_y, 10)

    # Hyper-parameters
    learning_rate = DEFAULT_LEARNING_RATE
    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE

    init_fun, predict_fun = minionn()

    start = time.perf_counter()
    params = train(
        train_x,
        train_y,
        init_fun,
        predict_fun,
        learning_rate,
        epochs,
        batch_size,
        run_on_spu,
    )
    end = time.perf_counter()

    print(f"train(cpu) elapsed time: {end - start:0.4f} seconds")
    test_x, test_y = test_ds['image'], test_ds['label']
    predict_y = predict_fun(params, test_x)

    print(f'accuracy(cpu): {accuracy_score(np.argmax(predict_y, axis=1),test_y)}')


def train_lenet(run_on_spu: bool = False):
    train_ds, test_ds = get_datasets('mnist')
    train_x, train_y = train_ds['image'], train_ds['label']
    train_y = jax.nn.one_hot(train_y, 10)

    # Hyper-parameters
    learning_rate = DEFAULT_LEARNING_RATE
    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE

    init_fun, predict_fun = lenet()

    start = time.perf_counter()
    params = train(
        train_x,
        train_y,
        init_fun,
        predict_fun,
        learning_rate,
        epochs,
        batch_size,
        run_on_spu,
    )
    end = time.perf_counter()

    print(f"train(cpu) elapsed time: {end - start:0.4f} seconds")
    test_x, test_y = test_ds['image'], test_ds['label']
    predict_y = predict_fun(params, test_x)

    print(f'accuracy(cpu): {accuracy_score(np.argmax(predict_y, axis=1),test_y)}')


def train_chamelon(run_on_spu: bool = False):
    train_ds, test_ds = get_datasets('mnist')
    train_x, train_y = train_ds['image'], train_ds['label']
    train_y = jax.nn.one_hot(train_y, 10)

    # Hyper-parameters
    learning_rate = DEFAULT_LEARNING_RATE
    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE

    init_fun, predict_fun = chamelon()

    start = time.perf_counter()
    params = train(
        train_x,
        train_y,
        init_fun,
        predict_fun,
        learning_rate,
        epochs,
        batch_size,
        run_on_spu,
    )
    end = time.perf_counter()

    print(f"train(cpu) elapsed time: {end - start:0.4f} seconds")
    test_x, test_y = test_ds['image'], test_ds['label']
    predict_y = predict_fun(params, test_x)

    print(f'accuracy(cpu): {accuracy_score(np.argmax(predict_y, axis=1),test_y)}')


"""
NN functionality test
"""
import argparse

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("--model", default='network_a')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

print(f'The selected NN model is {args.model}.')

if args.model == 'network_a':
    fn = train_secureml
elif args.model == 'network_b':
    fn = train_minionn
elif args.model == 'network_c':
    fn = train_lenet
elif args.model == 'network_d':
    fn = train_chamelon
else:
    raise RuntimeError("unsupported model.")
print('Run on CPU\n------\n')
fn(run_on_spu=False)

print('Run on SPU\n------\n')
with open(args.config, 'r') as file:
    conf = json.load(file)
ppd.init(conf["nodes"], conf["devices"])

fn(run_on_spu=True)
