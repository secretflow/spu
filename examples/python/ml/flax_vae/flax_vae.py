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

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import random

import examples.python.ml.flax_vae.utils as vae_utils
from flax import linen as nn
from flax.training import train_state

# Replace absl.flags used by original authors with argparse for unittest
parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_epochs", default=5, type=int)
parser.add_argument("--num_steps", type=int)
parser.add_argument("--latents", default=20, type=int)
parser.add_argument("--output_dir", default=os.getcwd(), type=str)
args = parser.parse_args()


class Encoder(nn.Module):
    latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(500, name='fc1')(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(500, name='fc1')(z)
        z = nn.relu(z)
        z = nn.Dense(784, name='fc2')(z)
        return z


class CPU_VAE(nn.Module):
    latents: int = 20

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = cpu_reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))


def cpu_reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


class SPU_VAE(nn.Module):
    latents: int = 20

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, eps):
        mean, logvar = self.encoder(x)
        z = spu_reparameterize(mean, logvar, eps)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))


def spu_reparameterize(mean, logvar, eps):
    std = jnp.exp(0.5 * logvar)
    # random.normal is not supported on SPU currently
    return mean + eps * std


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))


def compute_metrics(recon_x, x, mean, logvar):
    bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    return {'bce': bce_loss, 'kld': kld_loss, 'loss': bce_loss + kld_loss}


def cpu_model():
    return CPU_VAE(latents=args.latents)


def spu_model():
    return SPU_VAE(latents=args.latents)


@jax.jit
def train_step_cpu(state, batch, z_rng):
    def loss_fn(params):
        recon_x, mean, logvar = cpu_model().apply({'params': params}, batch, z_rng)

        bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = bce_loss + kld_loss
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


@jax.jit
def eval_cpu(params, images, z, z_rng):
    def eval_model(vae):
        recon_images, mean, logvar = vae(images, z_rng)
        comparison = jnp.concatenate(
            [images[:8].reshape(-1, 28, 28, 1), recon_images[:8].reshape(-1, 28, 28, 1)]
        )

        generate_images = vae.generate(z)
        generate_images = generate_images.reshape(-1, 28, 28, 1)
        metrics = compute_metrics(recon_images, images, mean, logvar)
        return metrics, comparison, generate_images

    return nn.apply(eval_model, cpu_model())({'params': params})


@jax.jit
def train_step_spu(state, batch, eps):
    def loss_fn(params, eps):
        recon_x, mean, logvar = spu_model().apply({'params': params}, batch, eps)

        bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = bce_loss + kld_loss
        return loss

    grads = jax.grad(loss_fn)(state.params, eps)
    return state.apply_gradients(grads=grads)


@jax.jit
def eval_spu(params, images, z, eps):
    def eval_model(vae, eps):
        recon_images, mean, logvar = vae(images, eps)
        comparison = jnp.concatenate(
            [images[:8].reshape(-1, 28, 28, 1), recon_images[:8].reshape(-1, 28, 28, 1)]
        )

        generate_images = vae.generate(z)
        generate_images = generate_images.reshape(-1, 28, 28, 1)
        metrics = compute_metrics(recon_images, images, mean, logvar)
        return metrics, comparison, generate_images

    return nn.apply(eval_model, spu_model())({'params': params}, eps)


def prepare_image(x):
    x = tf.cast(x['image'], tf.float32)
    x = tf.reshape(x, (-1,))
    return x


import json
import time

import spu.utils.distributed as ppd

with open("examples/python/conf/3pc.json", 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def train(run_on_spu: bool = True):
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

    rng = random.PRNGKey(0)
    rng, key = random.split(rng)

    ds_builder = tfds.builder('binarized_mnist')
    ds_builder.download_and_prepare()
    train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
    train_ds = train_ds.map(prepare_image)
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(50000, seed=10)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = iter(tfds.as_numpy(train_ds))

    test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
    test_ds = test_ds.map(prepare_image).batch(10000)
    test_ds = np.array(list(test_ds)[0])
    test_ds = jax.device_put(test_ds)

    init_data = jnp.ones((args.batch_size, 784), jnp.float32)

    if run_on_spu:
        state = train_state.TrainState.create(
            apply_fn=spu_model().apply,
            params=spu_model().init(
                key,
                init_data,
                np.random.normal(size=(len(init_data), args.latents)),
            )['params'],
            tx=optax.adam(args.learning_rate),
        )
    else:
        state = train_state.TrainState.create(
            apply_fn=cpu_model().apply,
            params=cpu_model().init(key, init_data, rng)['params'],
            tx=optax.adam(args.learning_rate),
        )

    rng, z_key, eval_rng = random.split(rng, 3)
    z = random.normal(z_key, (64, args.latents))

    steps_per_epoch = args.num_steps if args.num_steps else 50000 // args.batch_size

    start_ts = time.time()
    for epoch in range(args.num_epochs):
        for _ in range(steps_per_epoch):
            batch = next(train_ds)
            if run_on_spu:
                spu_batch = ppd.device("P1")(lambda x: x)(batch)
                state = ppd.device("SPU")(train_step_spu)(
                    state,
                    spu_batch,
                    np.random.normal(size=(len(batch), args.latents)),
                )
                metrics, comparison, sample = eval_spu(
                    ppd.get(state).params,
                    test_ds,
                    z,
                    np.random.normal(size=(len(test_ds), args.latents)),
                )
            else:
                rng, key = random.split(rng)
                state = train_step_cpu(state, batch, key)
                metrics, comparison, sample = eval_cpu(
                    state.params, test_ds, z, eval_rng
                )
            print(
                'eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}'.format(
                    epoch + 1, metrics['loss'], metrics['bce'], metrics['kld']
                )
            )

        if run_on_spu:
            metrics, comparison, sample = eval_spu(
                ppd.get(state).params,
                test_ds,
                z,
                np.random.normal(size=(len(test_ds), args.latents)),
            )
        else:
            metrics, comparison, sample = eval_cpu(state.params, test_ds, z, eval_rng)
        suffix = "spu" if run_on_spu else "cpu"
        vae_utils.save_image(
            comparison,
            f'{args.output_dir}/reconstruction_{epoch}_{suffix}.png',
            nrow=8,
        )
        vae_utils.save_image(
            sample, f'{args.output_dir}/sample_{epoch}_{suffix}.png', nrow=8
        )

        print(
            'eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}'.format(
                epoch + 1, metrics['loss'], metrics['bce'], metrics['kld']
            )
        )
    print(f'Elapsed time:{time.time() - start_ts}')
    return metrics


def main():
    print('Run on CPU\n------\n')
    train(run_on_spu=False)
    print('Run on SPU\n------\n')
    train(run_on_spu=True)


if __name__ == '__main__':
    main()
