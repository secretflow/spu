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
# > bazel run -c opt //examples/python/ml/flax_gpt2:flax_gpt2

import argparse
import json
import time

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Config

import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2

from hack_functions import hack_gelu_context, hack_softmax_context

parser = argparse.ArgumentParser(description="distributed driver.")
parser.add_argument(
    "-c", "--config", default="examples/python/ml/experimental_mp/3pc.json"
)
parser.add_argument("--fp16_w", action="store_true")
parser.add_argument("--gelu", default="raw", help="['raw', 'quad', 'poly']")
parser.add_argument("--softmax", default="raw", help="['raw', '2relu', '2quad']")
args = parser.parse_args()

with open(args.config, "r") as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


whether_hack_gelu = False if args.gelu == "raw" else True
whether_hack_softmax = False if args.softmax == "raw" else True

prompt_text = "I enjoy walking with my cute dog"
max_seq_length = 32
token_num = 1

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

# NOTE: YOU SHOULD CHANGE THE MODEL_PATH TO YOUR PATH.
# Specificall, you should point to quantization-aware distilled models.
# model_path = "/checkpoints/DQ/gelu_new_softmax_gpt2"
# model_path = "/checkpoints/DQ/quan_quad_softmax_gpt2"
model_path = "/data/models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
pretrained_model = FlaxGPT2LMHeadModel.from_pretrained(model_path, from_pt=True)

# Converting model params to FP16
if args.fp16_w:
    print(f"==== Converting fp16 weights ====")
    from flax import traverse_util

    flat_params = traverse_util.flatten_dict(pretrained_model.params)
    # for path in flat_params:
    #     print(path)
    mask = {
        path: (
            path[-1] != "embedding"
            and path[-2] != "lm_head"
            and path[-2:] != ("LayerNorm", "scale")
            and path[-2:] != ("classifier", "bias")
            and path[-2:] != ("classifier", "kernel")  # should match to var name
        )
        for path in flat_params
    }
    mask = traverse_util.unflatten_dict(mask)
    print(mask)
    pretrained_model.params = pretrained_model.to_fp16(pretrained_model.params, mask)


# greedy search
# ref: https://huggingface.co/blog/how-to-generate
def text_generation(input_ids, params, token_num=token_num):
    # config = GPT2Config()
    # config.tie_word_embeddings = False
    # model = FlaxGPT2LMHeadModel(config=config)
    model = pretrained_model

    for _ in range(token_num):
        outputs = model(input_ids=input_ids, params=params, train=False)
        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
    return input_ids


def run_on_cpu():
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode(
        prompt_text,
        return_tensors="jax",
        padding="max_length",
        max_length=max_seq_length,
    )

    with hack_gelu_context(args.gelu, enabled=whether_hack_gelu), hack_softmax_context(
        args.softmax, enabled=whether_hack_softmax
    ):
        outputs_ids = text_generation(inputs_ids, pretrained_model.params)

    print(f"CPU: Output ids: {outputs_ids[0]}")
    return outputs_ids


def run_on_spu():
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode(
        prompt_text,
        return_tensors="jax",
        padding="max_length",
        max_length=max_seq_length,
    )

    input_ids = ppd.device("P1")(lambda x: x)(inputs_ids)
    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)

    start = time.time()
    with hack_gelu_context(args.gelu, enabled=whether_hack_gelu), hack_softmax_context(
        args.softmax, enabled=whether_hack_softmax
    ):
        outputs_ids = ppd.device("SPU")(text_generation, copts=copts)(input_ids, params)
    end = time.time()
    print(f"SPU GPT2 time: {end - start}s")

    outputs_ids = ppd.get(outputs_ids)
    print(f"SPU: Output ids: {outputs_ids[0]}")
    return outputs_ids


if __name__ == "__main__":
    print("\n------\nRun on CPU")
    outputs_ids = run_on_cpu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
    print("\n------\nRun on SPU")
    outputs_ids = run_on_spu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
