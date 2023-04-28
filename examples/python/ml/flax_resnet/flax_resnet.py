# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax implementation of ResNet V1."""

# See issue #620.
# pytype: disable=wrong-arg-count

from typing import Any
import argparse
import time

import tensorflow as tf
import tensorflow_datasets as tfds

import optax
from flax.training import train_state
import jax.numpy as jnp
import jax
from jax import random

from models import ResNet18

NUM_CLASSES = 10
IMAGE_SIZE = 32
resnet_model = ResNet18(num_classes=NUM_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="distributed driver.")
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--on_spu", action="store_true", default=False)
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument(
        "--config", default="examples/python/ml/flax_resnet/3pc.json", type=str
    )
    return parser.parse_args()


args = parse_args()


def prepare_image(batch):
    x = tf.cast(batch["image"], tf.float32)
    y = batch["label"]
    return x, y


@jax.vmap
def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=NUM_CLASSES)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


class TrainState(train_state.TrainState):
    batch_stats: Any


def create_train_state(rng, model, image_size):
    """Create initial training state."""

    params, batch_stats = initialized(rng, image_size, model)
    tx = optax.adam(args.learning_rate)
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats
    )
    return state


@jax.jit
def train_step(state, batch_x, batch_y):
    """Perform a single training step."""

    def loss_fn(params):
        """loss function used for training."""
        logits, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch_x,
            mutable=["batch_stats"],
        )

        loss = cross_entropy_loss(logits, batch_y)
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_decay = 0.0001
        weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        return loss.mean(), new_model_state

    aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_model_state = aux[1]
    return state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )


def eval_step(state, batch_x, batch_y):
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch_x,
        train=False,
        mutable=False,
    )
    return compute_metrics(logits, batch_y)


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({"params": key}, jnp.ones(input_shape, model.dtype))
    return variables["params"], variables["batch_stats"]


def train(run_on_spu: bool = True, args=None):
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], "GPU")
    tf.random.set_seed(args.seed)

    rng = random.PRNGKey(0)

    ds_builder = tfds.builder("cifar10")
    ds_builder.download_and_prepare()
    train_ds = ds_builder.as_dataset("train")
    train_ds = train_ds.map(prepare_image)
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = iter(tfds.as_numpy(train_ds))

    test_ds = ds_builder.as_dataset("test")
    test_ds = test_ds.map(prepare_image)
    test_ds = test_ds.batch(1024)
    test_ds = iter(tfds.as_numpy(test_ds))

    state = create_train_state(rng, resnet_model, IMAGE_SIZE)
    steps_per_epoch = args.num_steps if args.num_steps else 50000 // args.batch_size

    print(f"Start training ...")
    start_ts = time.time()
    for epoch in range(args.num_epochs):
        test_image, test_label = next(test_ds)
        for _ in range(steps_per_epoch):
            image, label = next(train_ds)
            if run_on_spu:
                spu_image = ppd.device("P1")(lambda x: x)(image)
                spu_label = ppd.device("P1")(lambda x: x)(label)
                state = ppd.device("SPU")(train_step)(state, spu_image, spu_label)

                metrics = eval_step(
                    ppd.get(state),
                    test_image,
                    test_label,
                )
                print(
                    "eval epoch: {}, loss: {:.4f}, accuracy: {:.4f}".format(
                        epoch + 1, metrics["loss"], metrics["accuracy"]
                    )
                )
            else:
                state = train_step(state, image, label)
                metrics = eval_step(
                    state,
                    test_image,
                    test_label,
                )
                print(
                    "eval epoch: {}, loss: {:.4f}, accuracy: {:.4f}".format(
                        epoch + 1, metrics["loss"], metrics["accuracy"]
                    )
                )

        suffix = "spu" if run_on_spu else "cpu"

        print(
            "== {} == eval epoch: {}, loss: {:.4f}, accuracy: {:.4f}".format(
                suffix, epoch + 1, metrics["loss"], metrics["accuracy"]
            )
        )
    print(f"Elapsed time:{time.time() - start_ts}")
    return metrics


import json

import spu.utils.distributed as ppd

with open(args.config, "r") as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def main():
    print("Run on CPU\n------\n")
    train(run_on_spu=False, args=args)
    print("Run on SPU\n------\n")
    train(run_on_spu=True, args=args)


if __name__ == "__main__":
    main()
