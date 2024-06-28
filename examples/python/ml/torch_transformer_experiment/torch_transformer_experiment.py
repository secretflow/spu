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
import torch
import torch.optim as optim
import torch.utils.data as data
import math
import copy
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
EPOCH=3
src_vocab_size = 500 #0
tgt_vocab_size = 500
d_model = 512
num_heads = 4 #8
num_layers = 4
d_ff = 2048
max_seq_length = 100
dropout = 0.1


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

def train(model):
    print('Train model with plaintext features\n------\n')
    # x, y = data_loader()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(EPOCH):
        optimizer.zero_grad()
        output = model(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, dim=2)
        correct = (predicted == tgt_data[:, 1:]).sum().item()
        total = (tgt_data[:, 1:] != 0).sum().item()
        accuracy = correct / total

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()} ,Accuracy: {accuracy}")

    print('Train model finished\n------\n')


# prepare test datasets
def data_loader(
    train: bool = True,
    *,
    normalize: bool = True,
):
    # src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    # tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    src_data_numpy = src_data.numpy()
    tgt_data_numpy = tgt_data[:, :-1].numpy()
    return src_data_numpy,tgt_data_numpy




import time


def run_inference_on_cpu(model):
    print('Run on CPU\n------\n')
    start_ts = time.time()
    test_output = model(src_data, tgt_data[:, :-1])
    _, predicted = torch.max(test_output, dim=2)
    correct = (predicted == tgt_data[:, 1:]).sum().item()
    total = (tgt_data[:, 1:] != 0).sum().item()
    accuracy = correct / total
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
    x, y = ppd.device("P2")(data_loader)(False)
    start_ts = time.time()
    y_pred_ciphertext = ppd.device('SPU')(model)(params, x,y)
    end_ts = time.time()
    _,y_pred_plaintext = ppd.get(y_pred_ciphertext)
    correct = (y_pred_plaintext == tgt_data[:, 1:]).sum().item()
    total = (tgt_data[:, 1:] != 0).sum().item()
    accuracy = correct / total
    print(f"accuracy(spu)={accuracy}, time={end_ts-start_ts}\n------\n")
    return y_pred_plaintext


if __name__ == '__main__':
    # For reproducibility
    torch.manual_seed(0)

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    print(model)
    # Train model with plaintext features
    train(model)
    model.eval()
    # Native torch inference
    run_inference_on_cpu(model)
    # SPU inference
    run_inference_on_spu(model)
