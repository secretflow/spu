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


import argparse
import json
import math
import os
import warnings
from typing import Tuple, TypeVar

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotnine as gg

import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='SPU LSTM example.')
parser.add_argument("--output_dir", default=os.getcwd())
parser.add_argument("--num_steps", default=2001, type=int)
args = parser.parse_args()

with open("examples/python/conf/3pc.json", 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


T = TypeVar('T')
Pair = Tuple[T, T]

gg.theme_set(gg.theme_bw())
warnings.filterwarnings('ignore')


def sine_seq(
    phase: float,
    seq_len: int,
    samples_per_cycle: int,
) -> Pair[np.ndarray]:
    """Returns x, y in [T, B] tensor."""
    t = np.arange(seq_len + 1) * (2 * math.pi / samples_per_cycle)
    t = t.reshape([-1, 1]) + phase
    sine_t = np.sin(t)
    return sine_t[:-1, :], sine_t[1:, :]


def generate_data(
    seq_len: int,
    train_size: int,
    valid_size: int,
) -> Pair[Pair[np.ndarray]]:
    np.random.seed(0)
    phases = np.random.uniform(0.0, 2 * math.pi, [train_size + valid_size])
    all_x, all_y = sine_seq(phases, seq_len, 3 * seq_len / 4)

    all_x = np.expand_dims(all_x, -1)
    all_y = np.expand_dims(all_y, -1)
    train_x = all_x[:, :train_size]
    train_y = all_y[:, :train_size]

    valid_x = all_x[:, train_size:]
    valid_y = all_y[:, train_size:]
    return (train_x, train_y), (valid_x, valid_y)


class Dataset:
    """An iterator over a numpy array, revealing batch_size elements at a time."""

    def __init__(self, xy: Pair[np.ndarray], batch_size: int):
        self._x, self._y = xy
        self._batch_size = batch_size
        self._length = self._x.shape[1]
        self._idx = 0
        if self._length % batch_size != 0:
            msg = 'dataset size {} must be divisible by batch_size {}.'
            raise ValueError(msg.format(self._length, batch_size))

    def __next__(self) -> Pair[np.ndarray]:
        start = self._idx
        end = start + self._batch_size
        x, y = self._x[:, start:end], self._y[:, start:end]
        if end >= self._length:
            end = end % self._length
            assert end == 0  # Guaranteed by ctor assertion.
        self._idx = end
        return x, y


TRAIN_SIZE = 2**14
VALID_SIZE = 128
BATCH_SIZE = 8
SEQ_LEN = 64

train, valid = generate_data(SEQ_LEN, TRAIN_SIZE, VALID_SIZE)

# Plot an observation/target pair.
df = pd.DataFrame({'x': train[0][:, 0, 0], 'y': train[1][:, 0, 0]}).reset_index()
df = pd.melt(df, id_vars=['index'], value_vars=['x', 'y'])
plot = gg.ggplot(df) + gg.aes(x='index', y='value', color='variable') + gg.geom_line()
plot.save(filename=f"{args.output_dir}/observation_target_pair.png")

train_ds = Dataset(train, BATCH_SIZE)
valid_ds = Dataset(valid, BATCH_SIZE)
del train, valid  # Don't leak temporaries.


def unroll_net(seqs: jnp.ndarray):
    """Unrolls an LSTM over seqs, mapping each output to a scalar."""
    # seqs is [T, B, F].
    core = hk.LSTM(32)
    batch_size = seqs.shape[1]
    outs, state = hk.dynamic_unroll(core, seqs, core.initial_state(batch_size))
    # We could include this Linear as part of the recurrent core!
    # However, it's more efficient on modern accelerators to run the linear once
    # over the entire sequence than once per sequence element.
    return hk.BatchApply(hk.Linear(1))(outs), state


model = hk.transform(unroll_net)


def train_model(
    train_ds: Dataset, valid_ds: Dataset, run_on_spu: bool = True
) -> hk.Params:
    """Initializes and trains a model on train_ds, returning the final params."""
    rng = jax.random.PRNGKey(428)
    opt = optax.adam(1e-3)

    @jax.jit
    def loss(params, x, y):
        pred, _ = model.apply(params, None, x)
        return jnp.mean(jnp.square(pred - y))

    @jax.jit
    def update(step, params, opt_state, x, y):
        l, grads = jax.value_and_grad(loss)(params, x, y)
        grads, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return l, params, opt_state

    # Initialize state.
    sample_x, _ = next(train_ds)
    params = model.init(rng, sample_x)
    opt_state = opt.init(params)

    import time

    start_ts = time.time()
    for step in range(args.num_steps):
        if step % 100 == 0:
            x, y = next(valid_ds)
            if run_on_spu:
                print(
                    "Step {}: valid loss {}".format(step, loss(ppd.get(params), x, y))
                )
            else:
                print("Step {}: valid loss {}".format(step, loss(params, x, y)))

        x, y = next(train_ds)
        if run_on_spu:
            x = ppd.device("P1")(lambda x: x)(x)
            y = ppd.device("P2")(lambda y: y)(y)
            train_loss, params, opt_state = ppd.device("SPU")(update)(
                step, params, opt_state, x, y
            )
        else:
            train_loss, params, opt_state = update(step, params, opt_state, x, y)
        if step % 100 == 0:
            if run_on_spu:
                print("Step {}: train loss {}".format(step, ppd.get(train_loss)))
            else:
                print("Step {}: train loss {}".format(step, train_loss))

    print(f'Elapsed time:{time.time() - start_ts}')

    return params, train_loss


def plot_samples(truth: np.ndarray, prediction: np.ndarray) -> gg.ggplot:
    assert truth.shape == prediction.shape
    df = pd.DataFrame(
        {'truth': truth.squeeze(), 'predicted': prediction.squeeze()}
    ).reset_index()
    df = pd.melt(df, id_vars=['index'], value_vars=['truth', 'predicted'])
    plot = (
        gg.ggplot(df) + gg.aes(x='index', y='value', color='variable') + gg.geom_line()
    )
    return plot


# Generate a prediction, feeding in ground truth at each point as input.
def draw_predict(run_on_spu: bool, trained_params, sample_x):
    predicted, _ = model.apply(trained_params, None, sample_x)
    suffix = "spu" if run_on_spu else "cpu"
    plot = plot_samples(sample_x[1:], predicted[:-1])
    plot.save(filename=f"{args.output_dir}/predication_{suffix}.png")
    del predicted


# Typically: the beginning of the predictions are a bit wonky, but the curve
# quickly smooths out.
def autoregressive_predict(
    trained_params: hk.Params,
    context: jnp.ndarray,
    seq_len: int,
):
    """Given a context, auto regressively generate the rest of a sine wave."""
    ar_outs = []
    context = jax.device_put(context)
    for _ in range(seq_len - context.shape[0]):
        full_context = jnp.concatenate([context] + ar_outs)
        outs, _ = jax.jit(model.apply)(trained_params, None, full_context)
        # print(f"{len(ar_outs)}, {len(outs)}, {outs[0]}, {context[0]}")
        # Append the newest prediction to ar_outs.
        ar_outs.append(outs[-1:])
    # Return the final full prediction.
    return outs


# We can reuse params we got from training for inference - as long as the
# declaration order is the same.
def draw_ar_prediction(run_on_spu: bool, trained_params, context, sample_x):
    predicted = autoregressive_predict(trained_params, context, SEQ_LEN)
    plot = plot_samples(sample_x[1:, :1], predicted)
    plot += gg.geom_vline(xintercept=len(context), linetype='dashed')
    suffix = "spu" if run_on_spu else "cpu"
    plot.save(filename=f"{args.output_dir}/ar_prediction_{suffix}.png")
    del predicted


def fast_autoregressive_predict_fn(context, seq_len):
    """Given a context, auto regressively generate the rest of a sine wave."""
    core = hk.LSTM(32)
    dense = hk.Linear(1)
    state = core.initial_state(context.shape[1])
    # Unroll over the context using `hk.dynamic_unroll`.
    # As before, we `hk.BatchApply` the Linear for efficiency.
    context_outs, state = hk.dynamic_unroll(core, context, state)
    context_outs = hk.BatchApply(dense)(context_outs)

    # Now, unroll one step at a time using the running recurrent state.
    ar_outs = []
    x = context_outs[-1]
    for _ in range(seq_len - context.shape[0]):
        x, state = core(x, state)
        x = dense(x)
        ar_outs.append(x)
    return jnp.concatenate([context_outs, jnp.stack(ar_outs)])


def draw_from_reused_params(run_on_spu: bool, trained_params, context, sample_x):
    fast_ar_predict = hk.transform(fast_autoregressive_predict_fn)
    fast_ar_predict = jax.jit(fast_ar_predict.apply, static_argnums=3)
    # Reuse the same context from the previous cell.
    predicted = fast_ar_predict(trained_params, None, context, SEQ_LEN)
    # The plots should be equivalent!
    plot = plot_samples(sample_x[1:, :1], predicted[:-1])
    plot += gg.geom_vline(xintercept=len(context), linetype='dashed')
    suffix = "spu" if run_on_spu else "cpu"
    plot.save(
        filename=f"{args.output_dir}/ar_prediction_with_reused_params_{suffix}.png"
    )


def main():
    # Train models
    print('Run on CPU\n------\n')
    cpu_trained_params, _ = train_model(train_ds, valid_ds, run_on_spu=False)
    print('Run on SPU\n------\n')
    spu_trained_params, _ = ppd.get(train_model(train_ds, valid_ds, run_on_spu=True))

    # Grab a sample from the validation set.
    sample_x, _ = next(valid_ds)
    sample_x = sample_x[:, :1]  # Shrink to batch-size 1.

    # Predict on CPU model
    draw_predict(False, cpu_trained_params, sample_x)
    # Predict on SPU model
    draw_predict(True, spu_trained_params, sample_x)
    del sample_x

    sample_x, _ = next(valid_ds)
    context_length = SEQ_LEN // 8
    # Cut the batch-size 1 context from the start of the sequence.
    context = sample_x[:context_length, :1]

    # Autoregressive predict on CPU model
    draw_ar_prediction(False, cpu_trained_params, context, sample_x)
    # Autoregressive predict on SPU model
    draw_ar_prediction(True, spu_trained_params, context, sample_x)

    # Reuse CPU model
    draw_from_reused_params(False, cpu_trained_params, context, sample_x)
    # Reuse SPU model
    draw_from_reused_params(True, spu_trained_params, context, sample_x)


if __name__ == '__main__':
    main()
