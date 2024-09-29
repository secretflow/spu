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
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_gpt2/3pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_gpt2:flax_gpt2

import argparse
import json

import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Config

import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/flax_gpt2/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


tokenizer = AutoTokenizer.from_pretrained("gpt2")
pretrained_model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")


# greedy search
# ref: https://huggingface.co/blog/how-to-generate
def text_generation(input_ids, params, token_num=10):
    config = GPT2Config()
    model = FlaxGPT2LMHeadModel(config=config)

    for _ in range(token_num):
        outputs = model(input_ids=input_ids, params=params)
        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
    return input_ids


def run_on_cpu():
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode(
        'I enjoy walking with my cute dog', return_tensors='jax'
    )
    outputs_ids = text_generation(inputs_ids, pretrained_model.params)
    return outputs_ids


def run_on_spu():
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode(
        'I enjoy walking with my cute dog', return_tensors='jax'
    )

    input_ids = ppd.device("P1")(lambda x: x)(inputs_ids)
    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
    outputs_ids = ppd.device("SPU")(
        text_generation,
    )(input_ids, params)
    outputs_ids = ppd.get(outputs_ids)
    return outputs_ids


if __name__ == '__main__':
    print('\n------\nRun on CPU')
    outputs_ids = run_on_cpu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
    print('\n------\nRun on SPU')
    outputs_ids = run_on_spu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
