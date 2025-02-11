# Copyright 2024 Ant Group Co., Ltd.
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
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_whisper/3pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_whisper:flax_whisper

import argparse
import json
import os

import jax.numpy as jnp
from datasets import load_dataset
from transformers import FlaxWhisperForConditionalGeneration, WhisperProcessor

import spu.utils.distributed as ppd
from spu import libspu

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument(
    "-c", "--config", default="examples/python/ml/flax_whisper/3pc.json"
)
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
pretrained_model = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny.en", from_pt=True
)


def text_generation(input_features, params):
    pretrained_model.params = params
    generated_ids = pretrained_model.generate(input_features=input_features)
    return generated_ids.sequences


def run_on_cpu():
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
    generated_ids = pretrained_model.generate(input_features=inputs.input_features)
    return generated_ids.sequences


def run_on_spu():
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    inputs_ids = processor(ds[0]["audio"]["array"], return_tensors="np")

    # Enable rewrite for better performance
    copts = libspu.CompilerOptions()
    copts.enable_optimize_denominator_with_broadcast = True

    input_ids = ppd.device("P1")(lambda x: x)(inputs_ids.input_features)
    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
    outputs_ids = ppd.device("SPU")(text_generation, copts=copts)(input_ids, params)
    outputs_ids = ppd.get(outputs_ids)
    return outputs_ids


if __name__ == '__main__':
    print('\n------\nRun on CPU')
    generated_ids = run_on_cpu()
    print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
    print('\n------\nRun on SPU')
    generated_ids = run_on_spu()
    print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
