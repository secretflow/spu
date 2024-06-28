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
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
import torch.optim as optim
from sklearn.datasets import make_blobs
import spu.utils.distributed as ppd


# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/torch_lr_experiment:torch_lr_experiment


def train( model, args):
    X, Y = data(True)
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    N = len(Y)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epoch):
        perm = torch.randperm(N)
        sum_loss = 0

        for i in range(0, N, args.batchsize):
            x = X[perm[i : i + args.batchsize]].to(args.device)
            y = Y[perm[i : i + args.batchsize]].to(args.device)

            optimizer.zero_grad()
            output = model(x).squeeze()
            weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - y * output, min=0))
            loss += args.c * (weight.t() @ weight) / 2.0

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))


def visualize(X, Y, model):
    W = model.weight.squeeze().detach().cpu().numpy()
    b = model.bias.squeeze().detach().cpu().numpy()

    delta = 0.001
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)
    z[np.where(z > 1.0)] = 4
    z[np.where((z > 0.0) & (z <= 1.0))] = 3
    z[np.where((z > -1.0) & (z <= 0.0))] = 2
    z[np.where(z <= -1.0)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
    plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
    plt.tight_layout()
    plt.savefig('result.png')
    plt.show()

def data(
    train: bool = True,
    *,
    normalize: bool = True,
):
    from sklearn.model_selection import train_test_split
    X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)
    X = (X - X.mean()) / X.std()
    Y[np.where(Y == 0)] = -1
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
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
    x_test, y_test= data(False)
    print('Run on CPU\n------\n')
    x = torch.Tensor(x_test)
    start_ts = time.time()
    y_pred = model(x).cpu().detach().numpy()
    end_ts = time.time()
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(f"AUC(cpu)={auc}, time={end_ts-start_ts}\n------\n")


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"], framework=ppd.Framework.EXP_TORCH)

from collections import OrderedDict
from jax.tree_util import tree_map


def run_inference_on_spu(model):
    print('Run on SPU\n------\n')

    # load state dict on P1
    params = ppd.device("P1")(
        lambda input: tree_map(lambda x: x.detach().numpy(), input)
    )(model.state_dict())

    # load inputs on P2
    x, _ = ppd.device("P2")(data)(False)

    start_ts = time.time()
    y_pred_ciphertext = ppd.device('SPU')(model)(params, x)
    end_ts = time.time()
    y_pred_plaintext = ppd.get(y_pred_ciphertext)
    _, y_test = data(False)
    auc = metrics.roc_auc_score(y_test, y_pred_plaintext)
    print(f"AUC(spu)={auc}, time={end_ts-start_ts}\n------\n")
    return auc


if __name__ == '__main__':
    # For reproducibility
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batchsize", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    print(args)

    X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)
    X = (X - X.mean()) / X.std()
    Y[np.where(Y == 0)] = -1

    model = torch.nn.Linear(2, 1)
    model.to(args.device)

    train( model, args)
    visualize(X, Y, model)

    # Train model with plaintext features
    # train(model)
    model.eval()
    # Native torch inference
    run_inference_on_cpu(model)
    # SPU inference
    run_inference_on_spu(model)
