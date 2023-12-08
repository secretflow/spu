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


def infer_cipher(inputs, params):
    config = AutoConfig.from_pretrained("microsoft/resnet-50")
    model_cipher = FlaxResNetForImageClassification(config=config)
    outputs = model_cipher(pixel_values=inputs, params=params)
    return outputs


def main(inputs, model):
    inputs = ppd.device("P1")(lambda x: x)(inputs)
    params = ppd.device("P2")(lambda x: x)(model.params)
    outputs = ppd.device("SPU")(infer_cipher)(inputs, params)
    outputs = ppd.get(outputs)
    outputs = outputs['logits']
    predicted_class_idx = jax.numpy.argmax(outputs, axis=-1)
    print("Predicted class:", model.config.id2label[predicted_class_idx.item()])


if __name__ == "__main__":
    from time import time

    start = time()
    main(inputs, model)
    print(f"Elapsed time:{time() - start}")
