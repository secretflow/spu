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


from transformers import (
    AutoImageProcessor,
    AutoConfig,
    FlaxResNetForImageClassification,
)

from datasets import load_dataset

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
    print("Run on CPU\n------\n")
    run_on_cpu(inputs, model)
    print("Run on SPU\n------\n")
    run_on_spu(inputs, model)
