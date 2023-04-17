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

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/stax_nn:stax_nn

import argparse
import itertools
import json
import math
import time
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from jax.example_libraries import stax
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score

import examples.python.ml.stax_nn.models as models
import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("--model", default='network_a', type=str)
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json", type=str)
parser.add_argument("-e", "--epoch", default=5, type=int)
parser.add_argument("-b", "--batch_size", default=128, type=int)
parser.add_argument("-o", "--optimizer", default="SGD", type=str)
parser.add_argument('--run_cpu', default=False, action='store_true')
args = parser.parse_args()

# Follows https://arxiv.org/pdf/2107.00501.pdf Appendix C.
DEFAULT_EPOCHS = args.epoch
DEFAULT_BATCH_SIZE = args.batch_size


def train(
    train_x,
    train_y,
    init_fun,
    predict_fun,
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
    opt_kind = args.optimizer.lower()
    if opt_kind == "sgd":
        from jax.example_libraries import optimizers

        opt_init, opt_update, get_params = optimizers.momentum(0.01, 0.9)
    elif opt_kind == "adam":
        from jax.example_libraries import optimizers

        opt_init, opt_update, get_params = optimizers.adam(0.001)
    elif opt_kind == "amsgrad":
        from examples.python.utils import optimizers

        opt_init, opt_update, get_params = optimizers.amsgrad(0.001)
    else:
        raise RuntimeError(f"Unsupported optimizer type {args.optimizer}.")
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
    itercount = itertools.count()

    print('Start training...')
    for i in range(1, epochs + 1):
        for batch_idx in range(math.ceil(len(train_x) / batch_size)):
            batch_images = train_x[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            batch_labels = train_y[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            it = next(itercount)
            print(
                f'{datetime.now().time()} Epoch: {i}/{epochs}  Batch: {batch_idx}/{math.floor(len(train_x) / batch_size)}'
            )
            if run_on_spu:
                batch_images = ppd.device("P1")(identity)(batch_images)
                batch_labels = ppd.device("P2")(identity)(batch_labels)
                opt_state = ppd.device("SPU")(update_model)(
                    opt_state, batch_images, batch_labels, it
                )
            else:
                opt_state = update_model_jit(opt_state, batch_images, batch_labels, it)
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


def train_mnist(model, run_on_spu: bool = False):
    train_ds, test_ds = get_datasets('mnist')
    train_x, train_y = train_ds['image'], train_ds['label']
    train_y = jax.nn.one_hot(train_y, 10)

    # Hyper-parameters
    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE

    init_fun, predict_fun = model()

    start = time.perf_counter()
    params = train(
        train_x,
        train_y,
        init_fun,
        predict_fun,
        epochs,
        batch_size,
        run_on_spu,
    )
    end = time.perf_counter()

    env = 'spu' if run_on_spu else 'cpu'
    print(f"train({env}) elapsed time: {end - start:0.4f} seconds")
    test_x, test_y = test_ds['image'], test_ds['label']
    predict_y = predict_fun(params, test_x)
    score = accuracy_score(np.argmax(predict_y, axis=1), test_y)
    print(f'accuracy({env}): {score}')
    return score


def run_model(model_name, run_cpu=True):
    print(f'The selected NN model is {args.model}.')

    MODEL_MAPS = {
        'network_a': models.secureml,
        'network_b': models.minionn,
        'network_c': models.lenet,
        'network_d': models.chameleon,
    }

    fn = partial(train_mnist, MODEL_MAPS.get(model_name))
    if run_cpu:
        print('Run on CPU\n------\n')
        return fn(run_on_spu=False)

    print('Run on SPU\n------\n')
    return fn(run_on_spu=True)


def main():
    if args.run_cpu:
        run_model(args.model, run_cpu=True)

    with open(args.config, 'r') as file:
        conf = json.load(file)
    ppd.init(conf["nodes"], conf["devices"])

    return run_model(args.model, run_cpu=False)


if __name__ == '__main__':
    main()
