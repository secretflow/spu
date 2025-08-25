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

import examples.python.utils.distributed as ppd

# Start nodes.
# > python examples/python/utils/nodectl.py up
#
# Run this example script.
# > python examples/python/ml/torch_lr_experiment/torch_lr_experiment.py


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(30, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def train(model, n_epochs=500, lr=0.01):
    print('Train model with plaintext features\n------\n')
    x, y = breast_cancer()
    x = torch.Tensor(x)
    y = torch.Tensor(y).view(-1, 1)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        pred_y = model(x)
        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train model finished\n------\n')


# prepare test datasets
def breast_cancer(
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
    return x_.astype(dtype=np.float32), y_.astype(dtype=np.float32)


import time


def run_inference_on_cpu(model):
    print('Run on CPU\n------\n')
    x_test, y_test = breast_cancer(False)
    x = torch.Tensor(x_test)
    start_ts = time.time()
    y_pred = model(x).cpu().detach().numpy()
    end_ts = time.time()
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(f"AUC(cpu)={auc}, time={end_ts-start_ts}\n------\n")


from jax.tree_util import tree_map


def run_inference_on_spu(model):
    print('Run on SPU\n------\n')

    # load state dict on P1
    params = ppd.device("P1")(
        lambda input: tree_map(lambda x: x.detach().numpy(), input)
    )(model.state_dict())

    # load inputs on P2
    x, _ = ppd.device("P2")(breast_cancer)(False)

    start_ts = time.time()
    y_pred_ciphertext = ppd.device('SPU')(model)(params, x)
    end_ts = time.time()
    y_pred_plaintext = ppd.get(y_pred_ciphertext)
    _, y_test = breast_cancer(False)
    auc = metrics.roc_auc_score(y_test, y_pred_plaintext)
    print(f"AUC(spu)={auc}, time={end_ts-start_ts}\n------\n")
    return auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"], framework=ppd.Framework.EXP_TORCH)

    # For reproducibility
    torch.manual_seed(0)

    model = LinearRegression()
    # Train model with plaintext features
    train(model)
    model.eval()
    # Native torch inference
    run_inference_on_cpu(model)
    # SPU inference
    run_inference_on_spu(model)
