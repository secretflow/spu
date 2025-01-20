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

import jax

import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    FlaxResNetForImageClassification,
)

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = FlaxResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="jax")["pixel_values"]

from time import time


def inference(inputs, params):
    config = AutoConfig.from_pretrained("microsoft/resnet-50")
    model_cipher = FlaxResNetForImageClassification(config=config)
    outputs = model_cipher(pixel_values=inputs, params=params)
    return outputs


def run_on_spu(inputs, model):
    start = time()
    inputs = ppd.device("P1")(lambda x: x)(inputs)
    params = ppd.device("P2")(lambda x: x)(model.params)
    outputs = ppd.device("SPU")(inference)(inputs, params)
    outputs = ppd.get(outputs)
    outputs = outputs['logits']
    predicted_class_idx = jax.numpy.argmax(outputs, axis=-1)
    print(f"Elapsed time:{time() - start}")
    print("Predicted class:", model.config.id2label[predicted_class_idx.item()])


def run_on_cpu(inputs, model):
    start = time()
    outputs = inference(inputs, model.params)
    outputs = outputs['logits']
    predicted_class_idx = jax.numpy.argmax(outputs, axis=-1)
    print(f"Elapsed time:{time() - start}")
    print("Predicted class:", model.config.id2label[predicted_class_idx.item()])


if __name__ == "__main__":
    # print("Run on CPU\n------\n")
    # run_on_cpu(inputs, model)
    print("Run on SPU\n------\n")
    run_on_spu(inputs, model)
    ppd.print_status()
    ppd.clear_status()
