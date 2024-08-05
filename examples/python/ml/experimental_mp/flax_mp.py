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
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/experimental_mp/3pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/experimental_mp:flax_mp

import argparse
import json
from typing import Sequence
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import metrics
from contextlib import contextmanager
from dataclasses import dataclass
from flax.linen.linear import Array
from functools import partial

import examples.python.utils.dataset_utils as dsutil
import spu.utils.distributed as ppd

DTYPE = jnp.float32

@dataclass
class ModelConfig:
    msg: str
    use_one_hot: bool = True


class CustomDense(nn.Dense):
    """Hijack the Flax Dense layer"""

    custom_dense_config = None

    def setup(self):
        print(f"Custom Dense !!!!!!!!! {self.custom_dense_config.msg}")
        self.dtype = jnp.float32  # not working since dtype has been set when init
        return super().setup()

    def __call__(self, inputs: Array) -> Array:
        print(f"Custom Dense !!!!!!!!! {self.custom_dense_config.msg}")
        return super().__call__(inputs)

    def attend(self, query: Array) -> Array:
        return super().attend(query)


@contextmanager
def custom(msg: str, use_onehot: bool = True, enabled: bool = True):
    if not enabled:
        yield
        return
    CustomDense.custom_dense_config = ModelConfig(msg, use_one_hot=use_onehot)
    raw_dense = nn.Dense
    nn.Dense = CustomDense
    yield
    nn.Dense = raw_dense
    CustomDense.custom_dense_config = None


FEATURES = [30, 10, 5, 1]


class MLP(nn.Module):
    features: Sequence[int]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.relu = nn.relu

        self.layers = [nn.Dense(feat, dtype=self.dtype) for feat in self.features[:-1]]
        self.pred = nn.Dense(self.features[-1], dtype=self.dtype)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        x = self.pred(x)
        return x

    # @nn.compact
    # def __call__(self, x):
    #     for feat in self.features[:-1]:
    #         x = nn.relu(nn.Dense(feat)(x))
    #     x = nn.Dense(self.features[-1])(x)
    #     return x


class ManualMLP(nn.Module):
    features: Sequence[int]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.relu = nn.relu

        self.layers = [nn.Dense(feat, dtype=self.dtype) for feat in self.features[:-1]]
        self.pred = nn.Dense(self.features[-1], dtype=jnp.float32)

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.pred(x)
        return x


def predict(params, x):
    return ManualMLP(FEATURES, dtype=DTYPE).apply(params, x)


def loss_func(params, x, y):
    pred = predict(params, x)

    def mse(y, pred):
        def squared_error(y, y_pred):
            # TODO: check this
            return jnp.multiply(y - y_pred, y - y_pred) / 2.0
            # return jnp.inner(y - y_pred, y - y_pred) / 2.0 # fail, (10, 1) inner (10, 1) -> (10, 10), have to be (10,) inner (10,) -> scalar

        return jnp.mean(squared_error(y, pred))

    return mse(y, pred)


def train_auto_grad(x, y, params, n_batch=10, n_epochs=10, step_size=0.01):
    xs = jnp.array_split(x, len(x) / n_batch, axis=0)
    ys = jnp.array_split(y, len(y) / n_batch, axis=0)

    def body_fun(_, loop_carry):
        params = loop_carry
        for x, y in zip(xs, ys):
            _, grads = jax.value_and_grad(loss_func)(params, x, y)
            params = jax.tree_util.tree_map(
                lambda p, g: p - step_size * g, params, grads
            )
        return params

    params = jax.lax.fori_loop(0, n_epochs, body_fun, params)
    return params


# Model init is purely public and run on SPU leads to significant accuracy loss, thus hoist out and run in python
def model_init(n_batch=10):
    model = ManualMLP(FEATURES, dtype=DTYPE)
    return model.init(jax.random.PRNGKey(1), jnp.ones((n_batch, FEATURES[0])))


def run_on_cpu():
    x_train, y_train = dsutil.breast_cancer(slice(None, None, None), True)
    params = model_init()
    # params = jax.jit(train_auto_grad)(x_train, y_train, params)

    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)

    # print(jax.make_jaxpr(predict)(params, x_test))

    # 1. not working
    # with custom('ss'):
    #     model = MLP(FEATURES, dtype=jnp.float16)
    #     print(model.tabulate(jax.random.PRNGKey(0), x_test))

    # 2. not working
    # model = MLP(FEATURES, dtype=jnp.float16)
    # model_bound = model.bind(params)
    # print(model_bound.pred)
    # pred, _ = model_bound.pred.unbind()
    # pred.dtype = jnp.float32
    # print(pred)
    # # model_bound.pred = pred # not working, Frozen
    # print(model_bound.pred)
    # model, _ = model_bound.unbind()
    # print(model.tabulate(jax.random.PRNGKey(0), x_test))

    # 3. working. Manually set the dtype
    model = ManualMLP(FEATURES, dtype=DTYPE)
    # print(model.tabulate(jax.random.PRNGKey(0), x_test))

    y_predict = predict(params, x_test)
    print("AUC(cpu)={}".format(metrics.roc_auc_score(y_test, y_predict)))
    return params, y_predict


parser = argparse.ArgumentParser(description="distributed driver.")
parser.add_argument(
    "-c", "--config", default="examples/python/ml/experimental_mp/3pc.json"
)
args = parser.parse_args()

with open(args.config, "r") as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def compute_score(param, type):
    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)
    y_predict = predict(param, x_test)
    score = metrics.roc_auc_score(y_test, y_predict)
    print(f"AUC({type})={score}")
    return score


def run_on_spu():
    x_train, y_train = dsutil.breast_cancer(slice(None, None, None), True)
    params = model_init()
    # params = jax.jit(train_auto_grad)(x_train, y_train, params)

    x_test, y_test = dsutil.breast_cancer(slice(None, None, None), False)

    print(jax.make_jaxpr(predict)(params, x_test))

    model = ManualMLP(FEATURES, dtype=DTYPE)
    # print(model.tabulate(jax.random.PRNGKey(0), x_test))
    x_train_spu = ppd.device("P1")(lambda x: x)(x_train)
    params_spu = ppd.device("P1")(lambda x: x)(params)

    start = time.time()
    y_predict_spu = ppd.device("SPU")(predict)(params_spu, x_train_spu)
    end = time.time()
    print(f'Total time: {end - start} s')

    y_pred = ppd.get(y_predict_spu)
    print("AUC(spu)={}".format(metrics.roc_auc_score(y_train, y_pred)))
    return params, y_pred


def test_bert():
    from transformers import (
        BertTokenizerFast,
        FlaxBertForSequenceClassification,
    )
    from datasets import load_dataset

    tokenizer = BertTokenizerFast
    model = FlaxBertForSequenceClassification
    checkpoint = "bert-base-uncased"

    dataset = load_dataset("glue", "cola", split="train")
    dummy_input = next(iter(dataset))
    features, labels = dummy_input["sentence"], dummy_input["label"]

    tokenizer = tokenizer.from_pretrained(checkpoint)
    input_ids, attention_masks = (
        tokenizer(features, return_tensors="jax")["input_ids"],
        tokenizer(features, return_tensors="jax")["attention_mask"],
    )

    # Converting model params to FP16
    from flax import traverse_util

    model = model.from_pretrained(checkpoint)

    flat_params = traverse_util.flatten_dict(model.params)
    for path in flat_params:
        print(path)
    mask = {
        path: (
            path[-2] != ("LayerNorm", "bias")
            and path[-2:] != ("LayerNorm", "scale")
            and path[-2:] != ("classifier", "bias")
            and path[-2:] != ("classifier", "kernel")  # should match to var name
        )
        for path in flat_params
    }
    mask = traverse_util.unflatten_dict(mask)
    model.params = model.to_fp16(model.params, mask)

    # print(model.tabulate(jax.random.PRNGKey(0), input_ids, attention_masks))
    print(jax.make_jaxpr(model.__call__)(input_ids, attention_masks))


def some_tests():
    @jax.jit
    def func(ins):
        outputs, grads = jax.value_and_grad(lambda a: jnp.mean(nn.relu(a)))(ins)
        return outputs, grads

    ll = jnp.array([-1.1, 0, 1])
    print(jax.make_jaxpr(func)(ll))


if __name__ == "__main__":
    print("\n------\nRun on CPU")
    p, y = run_on_cpu()

    print("\n------\nRun on SPU")
    p_s, y_s = run_on_spu()

    # print(f"p: {p}\np_s: {p_s}")
    print(f"y: {y[:10]}\ny_s: {y_s[:10]}")

    # test_bert()
    # some_tests()
