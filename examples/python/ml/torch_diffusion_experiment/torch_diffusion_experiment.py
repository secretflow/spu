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

import torch.optim as optim
import torch.utils.data as data
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

import spu.utils.distributed as ppd
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/torch_lr_experiment:torch_diffusion_experiment


torch.manual_seed(1)    # reproducible
IMG_SIZE = 32     # input image size, CIFAR-10 is 32x32
BATCH_SIZE = 128  # for training batch size
timesteps = 16    # how many steps for a noisy image into clear
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1) # linspace for timesteps


plt.plot(time_bar, label='Noise')
plt.plot(1 - time_bar, label='Clarity')
plt.legend()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

all_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# filter training imgs
idx = [i for i, (img, label) in enumerate(all_trainset) if label == 1]
sub_trainset = torch.utils.data.Subset(all_trainset, idx)

trainloader = torch.utils.data.DataLoader(sub_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


def cvtImg(img):
    img = img.permute([0, 2, 3, 1])
    img = img - img.min()
    img = (img / img.max())
    return img.numpy().astype(np.float32)

def show_examples(x):
    plt.figure(figsize=(10, 10))
    imgs = cvtImg(x)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

x, _ = next(iter(trainloader))
show_examples(x)


def forward_noise(x, t):
    a = time_bar[t]  # base on t
    b = time_bar[t + 1]  # image for t + 1

    noise = np.random.normal(size=x.shape)  # noise mask
    a = a.reshape((-1, 1, 1, 1))
    b = b.reshape((-1, 1, 1, 1))
    img_a = x * (1 - a) + noise * a
    img_b = x * (1 - b) + noise * b
    return img_a, img_b


def generate_ts(num):
    return np.random.randint(0, timesteps, size=num)


# t = np.full((25,), timesteps - 1) # if you want see clarity
# t = np.full((25,), 0)             # if you want see noisy
t = generate_ts(25)  # random for training data
x, _ = next(iter(trainloader))
a, b = forward_noise(x[:25], t)
show_examples(a)


class Block(nn.Module):
    def __init__(self, in_channels=128, size=32):
        super(Block, self).__init__()

        self.conv_param = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)

        self.dense_ts = nn.Linear(192, 128)

        self.layer_norm = nn.LayerNorm([128, size, size])

    def forward(self, x_img, x_ts):
        x_parameter = F.relu(self.conv_param(x_img))

        time_parameter = F.relu(self.dense_ts(x_ts))
        time_parameter = time_parameter.view(-1, 128, 1, 1)
        x_parameter = x_parameter * time_parameter

        x_out = self.conv_out(x_img)
        x_out = x_out + x_parameter
        x_out = F.relu(self.layer_norm(x_out))

        return x_out


class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()

        self.l_ts = nn.Sequential(
            nn.Linear(1, 192),
            nn.LayerNorm([192]),
            nn.ReLU(),
        )

        self.down_x32 = Block(in_channels=3, size=32)
        self.down_x16 = Block(size=16)
        self.down_x8 = Block(size=8)
        self.down_x4 = Block(size=4)

        self.mlp = nn.Sequential(
            nn.Linear(2240, 128),
            nn.LayerNorm([128]),
            nn.ReLU(),

            nn.Linear(128, 32 * 4 * 4),  # make [-1, 32, 4, 4]
            nn.LayerNorm([32 * 4 * 4]),
            nn.ReLU(),
        )

        self.up_x4 = Block(in_channels=32 + 128, size=4)
        self.up_x8 = Block(in_channels=256, size=8)
        self.up_x16 = Block(in_channels=256, size=16)
        self.up_x32 = Block(in_channels=256, size=32)

        self.cnn_output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, padding=0)

        # make optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0008)

    def forward(self, x, x_ts):
        x_ts = self.l_ts(x_ts)

        # ----- left ( down ) -----
        blocks = [
            self.down_x32,
            self.down_x16,
            self.down_x8,
            self.down_x4,
        ]
        x_left_layers = []
        for i, block in enumerate(blocks):
            x = block(x, x_ts)
            x_left_layers.append(x)
            if i < len(blocks) - 1:
                x = F.max_pool2d(x, 2)

        # ----- MLP -----
        x = x.view(-1, 128 * 4 * 4)
        x = torch.cat([x, x_ts], dim=1)
        x = self.mlp(x)
        x = x.view(-1, 32, 4, 4)

        # ----- right ( up ) -----
        blocks = [
            self.up_x4,
            self.up_x8,
            self.up_x16,
            self.up_x32,
        ]

        for i, block in enumerate(blocks):
            # cat left
            x_left = x_left_layers[len(blocks) - i - 1]
            x = torch.cat([x, x_left], dim=1)

            x = block(x, x_ts)
            if i < len(blocks) - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear')

        # ----- output -----
        x = self.cnn_output(x)

        return x

def train_one(x_img,model):
    x_ts = generate_ts(len(x_img))
    x_a, x_b = forward_noise(x_img, x_ts)

    x_ts = torch.from_numpy(x_ts).view(-1, 1).float()
    x_a = x_a.float()
    x_b = x_b.float()

    y_p = model(x_a, x_ts)
    loss = torch.mean(torch.abs(y_p - x_b))
    model.opt.zero_grad()
    loss.backward()
    model.opt.step()

    return loss.item()

def train(model):
    R = 1
    bar = trange(R)
    total = len(trainloader)
    for i in bar:
        for j, (x_img, _) in enumerate(trainloader):
            loss = train_one(x_img, model)
            pg = (j / total) * 100
            if j % 5 == 0:
                bar.set_description(f'loss: {loss:.5f}, p: {pg:.2f}%')


x = torch.randn(32, 3, IMG_SIZE, IMG_SIZE)

# prepare test datasets
def data_loader(
    train: bool = True,
    *,
    normalize: bool = True,
):
    # src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    # tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    # src_data_numpy = src_data.float().numpy().astype(np.float32)
    # tgt_data_numpy = tgt_data[:, :-1].float().numpy().astype(np.float32)
    x_np = x.numpy()
    return x_np




import time


def run_inference_on_cpu(model,x):
    print('Run on CPU\n------\n')
    start_ts = time.time()
    with torch.no_grad():
        for i in trange(timesteps):
            t = i
            x = model(x, torch.full([32, 1], t, dtype=torch.float))

    show_examples(x.cpu())
    plt.savefig('predict_cpu.png')
    end_ts = time.time()
    print(f" time={end_ts-start_ts}\n------\n")


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
    x= ppd.device("P2")(data_loader)(False)
    start_ts = time.time()
    with torch.no_grad():
        for i in trange(timesteps):
            t = i
            # x = model(x, torch.full([32, 1], t, dtype=torch.float))
            y_pred_ciphertext = ppd.device('SPU')(model)(params, x, torch.full([32, 1], t, dtype=torch.float))
            y_pred_plaintext = ppd.get(y_pred_ciphertext)
    show_examples(y_pred_plaintext.cpu())
    plt.savefig('predict_spu.png')

    end_ts = time.time()
    # y_pred_plaintext = ppd.get(y_pred_ciphertext)
    # correct = (y_pred_plaintext == tgt_data[:, 1:]).sum().item()
    # total = (tgt_data[:, 1:] != 0).sum().item()
    # accuracy = correct / total
    print(f"a time={end_ts-start_ts}\n------\n")
    return y_pred_plaintext


if __name__ == '__main__':
    # For reproducibility
    torch.manual_seed(0)

    model = Diffusion()
    print(model)
    # Train model with plaintext features
    train(model)
    model.eval()
    # Native torch inference
    run_inference_on_cpu(model,x)
    # SPU inference
    run_inference_on_spu(model)



