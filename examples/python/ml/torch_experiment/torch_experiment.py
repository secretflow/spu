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

import numpy as np
import torch
from sklearn import metrics

import spu.utils.distributed as ppd

# This is an experimental example to show legacy pytorch program could be run
# by SPU. Currently we rely on torch-mlir to convert torch code into MLIR
# (specifically MHLO) which is then consumed by SPU. To run this example,
# torch-mlir python package should be installed. This example here trains a
# linear regression model in plaintext and makes private inferences with joint
# features.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/torch_experiment:torch_experiment


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(30, 1)

    def forward(self, x1, x2):
        y_pred = self.linear(torch.cat((x1, x2), 1))
        return y_pred


def train(model, n_epochs=500, lr=0.01):
    print('Train model with plaintext features\n------\n')
    x, y = breast_cancer()
    x1, x2 = x[:, :15], x[:, 15:]
    x1 = torch.Tensor(x1)
    x2 = torch.Tensor(x2)
    y = torch.Tensor(y).view(-1, 1)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        pred_y = model(x1, x2)
        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train model finished\n------\n')


# prepare test datasets
def breast_cancer(
    col_slicer=slice(None, None, None),
    train: bool = True,
    *,
    normalize: bool = True,
):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']

    # only difference to dsutil.breast_cancer
    y = y.astype(dtype=np.float64)

    if normalize:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if train:
        x_ = x_train
        y_ = y_train
    else:
        x_ = x_test
        y_ = y_test
    x_ = x_[:, col_slicer]
    return x_.astype(dtype=np.float32), y_.astype(dtype=np.float32)


import time


def run_inference_on_cpu(model):
    print('Run on CPU\n------\n')
    x_test, y_test = breast_cancer(slice(None, None, None), False)
    x1, x2 = torch.Tensor(x_test[:, :15]), torch.Tensor(x_test[:, 15:])
    start_ts = time.time()
    y_pred = model.forward(x1, x2).cpu().detach().numpy()
    end_ts = time.time()
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(f"AUC(cpu)={auc}, time={end_ts-start_ts}\n------\n")


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"], framework=ppd.Framework.EXP_TORCH)


def run_inference_on_spu(model):
    print('Run on SPU\n------\n')
    x1, _ = ppd.device("P1")(breast_cancer)(slice(None, 15), False)
    x2, _ = ppd.device("P2")(breast_cancer)(slice(15, None), False)
    start_ts = time.time()
    y_pred_ciphertext = ppd.device('SPU')(model)(x1, x2)
    end_ts = time.time()
    y_pred_plaintext = ppd.get(y_pred_ciphertext)
    _, y_test = breast_cancer(slice(None, None, None), False)
    auc = metrics.roc_auc_score(y_test, y_pred_plaintext)
    print(f"AUC(cpu)={auc}, time={end_ts-start_ts}\n------\n")
    return auc


def compile_torch_to_mhlo(model):
    print('Compile torch program to mhlo test\n------\n')
    x_test, _ = breast_cancer(slice(None, None, None), False)
    x1, x2 = torch.Tensor(x_test[:, :15]), torch.Tensor(x_test[:, 15:])
    import torch_mlir

    module = torch_mlir.compile(
        model,
        [x1, x2],
        output_type=torch_mlir.OutputType.MHLO,
    )
    print(f"MHLO={module}\n------\n")


if __name__ == '__main__':
    # For reproducibility
    torch.manual_seed(0)

    model = LinearRegression()
    # Train model with plaintext features
    train(model)
    # Torch-mlho conversion test
    compile_torch_to_mhlo(model)
    # Native torch inference
    run_inference_on_cpu(model)
    # SPU inference
    run_inference_on_spu(model)
