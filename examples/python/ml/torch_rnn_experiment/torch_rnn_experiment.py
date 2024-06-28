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

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

import spu.utils.distributed as ppd


# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/torch_lr_experiment:torch_rnn_experiment


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data


# Mnist digital dataset
train_data = dsets.MNIST(
    root='./mnist/',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)

# plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]    # covert to numpy array



class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

class RNN_0(nn.Module):
    def __init__(self):
        super(RNN_0, self).__init__()

        self.rnn = nn.RNN(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, h_n = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out




def train(model, n_epochs=500, lr=0.01):
    print('Train model with plaintext features\n------\n')
    x, y = breast_cancer()
    x = torch.Tensor(x)
    y = torch.Tensor(y).view(-1, 1)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
            b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)

            output = model(b_x)  # rnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 50 == 0:
                # test_output = model(test_x)  # (samples, time_step, input_size)
                # pred_y = torch.max(test_output, 1)[1].data.numpy()
                # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())# , '| test accuracy: %.2f' % accuracy)

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

def mnist_test(
    train: bool = True,
    *,
    normalize: bool = True,
):
    x_test_numpy = test_x.cpu().numpy()
    y_test_numpy = test_y
    return x_test_numpy.astype(dtype=np.float32), y_test_numpy.astype(dtype=np.float32)


import time


def run_inference_on_cpu(model):
    print('Run on CPU\n------\n')
    x = torch.Tensor(test_x)
    start_ts = time.time()
    test_output = model(test_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
    end_ts = time.time()
    print(f"accuracy(cpu)={accuracy}, time={end_ts-start_ts}\n------\n")


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
    x, _ = ppd.device("P2")(mnist_test)(False)
    start_ts = time.time()
    y_pred_ciphertext = ppd.device('SPU')(model)(params, x)
    end_ts = time.time()
    y_pred_plaintext = ppd.get(y_pred_ciphertext)
    y_pred_tensor = torch.from_numpy(y_pred_plaintext)
    pred_y = torch.max(y_pred_tensor, 1)[1].data.numpy()
    accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
    print(f"accuracy(spu)={accuracy}, time={end_ts-start_ts}\n------\n")
    return y_pred_plaintext


if __name__ == '__main__':
    # For reproducibility
    torch.manual_seed(0)

    model = RNN()
    print(model)
    # Train model with plaintext features
    train(model)
    model.eval()
    # Native torch inference
    run_inference_on_cpu(model)
    # SPU inference
    run_inference_on_spu(model)
