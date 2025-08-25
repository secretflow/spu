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
import urllib
from collections import OrderedDict

import torch
from jax.tree_util import tree_map
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

import examples.python.utils.distributed as ppd

# This is an experimental example to show legacy pytorch program could be run
# by SPU. Currently we rely on torch-xla to convert torch code into MLIR
# (specifically StableHLO) which is then consumed by SPU. To run this example,
# torch-xla python package should be installed.

# Start nodes.
# > python examples/python/utils/nodectl.py up
#
# Run this example script.
# > python examples/python/ml/torch_resnet_experiment/torch_resnet_experiment.py


def run_inference_on_cpu(model, image):
    print('Run on CPU\n------\n')
    output = model(image)
    # model predicts one of the 1000 ImageNet classes
    predicted_label = output.argmax(-1).item()
    print(f"predicted_label={predicted_label}\n------\n")
    return predicted_label


def run_inference_on_spu(model, image):
    print('Run on SPU\n------\n')
    params = ppd.device("P1")(
        lambda input: tree_map(lambda x: x.detach().numpy(), input)
    )(model.state_dict())
    image_hat = ppd.device("P2")(lambda x: x.detach().numpy())(image)
    res = ppd.device("SPU")(model)(params, image_hat)
    predicted_label = ppd.get(res).argmax(-1).item()
    print(f"predicted_label={predicted_label}\n------\n")
    return predicted_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"], framework=ppd.Framework.EXP_TORCH)

    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )

    urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet.eval()

    torch.manual_seed(0)
    run_inference_on_cpu(resnet, input_batch)
    run_inference_on_spu(resnet, input_batch)
