# Copyright 2025 Ant Group Co., Ltd.
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

import flax
import jax
import jax.numpy as jnp
import numpy as np
from rich import print
from transformers import AutoTokenizer

import spu.utils.distributed as ppd
from examples.python.ml.flax_ds_qwen.model_flax import Qwen2Config, Qwen2ForCausalLM
from examples.python.ml.flax_ds_qwen.torch_to_flax import torch_to_flax

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument(
    "-c", "--config", default="examples/python/ml/flax_ds_qwen/3pc.json"
)
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def text_generation(input_ids, params, max_new_tokens=1):
    config = Qwen2Config()
    model = Qwen2ForCausalLM(config=config)

    # Generate tokens one by one
    output = model.generate(
        params,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=False,  # for reproducibility
    )
    return output


def init_model_params(flax_params_path):
    # Load model params globally
    config = Qwen2Config()
    model = Qwen2ForCausalLM(config=config)
    rng = jax.random.PRNGKey(0)
    input_shape = (1, 32)

    # Force initialization on CPU to avoid duplicate GPU allocations
    with jax.default_device(jax.devices("cpu")[0]):
        try:
            params = model.init(rng, jnp.ones(input_shape, dtype=jnp.int4))
        except Exception as e:
            params = model.init(rng, jnp.ones(input_shape, dtype=jnp.int32))

    # Load the parameters from the file
    try:
        with open(flax_params_path, "rb") as f:
            params = {
                "params": flax.serialization.from_bytes(params["params"], f.read())
            }
    except FileNotFoundError:
        print("File not found. Running conversion...")
        torch_to_flax()
        with open(flax_params_path, "rb") as f:
            params = {
                "params": flax.serialization.from_bytes(params["params"], f.read())
            }

    params = jax.device_put(params)

    return params


def run_on_cpu(input_ids, params):
    outputs = text_generation(input_ids, params)
    return outputs


def run_on_spu(input_ids, params):
    # Split computation between parties
    input_ids = ppd.device("P1")(lambda x: x)(input_ids)
    params = ppd.device("P2")(lambda x: x)(params)
    outputs = ppd.device("SPU")(
        text_generation,
    )(input_ids, params)
    outputs = ppd.get(outputs)
    return outputs


if __name__ == "__main__":
    # Load tokenizer
    model_dir_name = "/data/models/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir_name)

    flax_params_path = "/home/jjzhou/codes/github/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/flax_params.msgpack"
    model_params = init_model_params(flax_params_path)

    prompt = "What is 3 + 4? <think>\n"
    inputs = tokenizer(prompt, return_tensors="jax")
    input_ids = inputs["input_ids"]

    print('\n------\nRun on CPU')
    outputs = run_on_cpu(input_ids, model_params)
    print(tokenizer.decode(np.array(outputs[0])))

    print('\n------\nRun on SPU')
    outputs = run_on_spu(input_ids, model_params)
    print(tokenizer.decode(np.array(outputs[0])))
