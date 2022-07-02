# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.

The mini-library jax.example_libraries.stax is for neural network building, and
the mini-library jax.example_libraries.optimizers is for first-order stochastic
optimization.
"""

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run //examples/python/ml:stax_mnist_classifier

import time
import itertools

import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax, Flatten, Conv, MaxPool
import examples.python.utils.dataset_utils as datasets


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


##############################################################################
# MLP
##############################################################################
init_random_params, predict = stax.serial(
    Flatten,
    Dense(1024),
    Relu,
    Dense(1024),
    Relu,
    Dense(10),
    LogSoftmax,
)


rng = random.PRNGKey(0)

num_epochs = 1
batch_size = 128

train_images, train_labels, test_images, test_labels = datasets.mnist()
num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

step_size = 0.15
opt_init, opt_update, get_params = optimizers.sgd(step_size)

# step_size = 0.05
# momentum_mass = 0.9
# opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)


def data_stream():
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]


def shape_as_image(images, labels, dummy_dim=False):
    target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
    return jnp.reshape(images, target_shape), labels


def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)


def run_cpu():
    print("\nStarting training...")
    _, init_params = init_random_params(rng, (-1, 28, 28, 1))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    batches = data_stream()
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            print(f"Training batch {itercount}/{num_batches}")
            opt_state = jit(update)(
                next(itercount), opt_state, shape_as_image(*next(batches))
            )
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        train_acc = accuracy(params, shape_as_image(train_images, train_labels))
        test_acc = accuracy(params, shape_as_image(test_images, test_labels))
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))


def run_spu():
    import argparse
    import json
    import spu.binding.util.distributed as ppd

    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"])

    print("\nStarting training...")
    _, init_params = init_random_params(rng, (-1, 28, 28, 1))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    batches = ppd.device("P1")(data_stream)()

    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            print(f"Training batch {itercount}/{num_batches}")
            batch = ppd.device("P1")(lambda batches: shape_as_image(*next(batches)))(
                batches
            )
            opt_state = ppd.device("SPU")(update)(next(itercount), opt_state, batch)
        epoch_time = time.time() - start_time

        params = get_params(ppd.get(opt_state))
        train_acc = accuracy(params, shape_as_image(train_images, train_labels))
        test_acc = accuracy(params, shape_as_image(test_images, test_labels))
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))


print("run_cpu:")
run_cpu()
print("run_spu:")
run_spu()
