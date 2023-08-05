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

import jax.numpy as jnp
from enum import Enum


class SigType(Enum):
    T1 = 't1'
    T3 = 't3'
    T5 = 't5'
    SEG3 = 'seg3'
    DF = 'df'
    SR = 'sr'
    LS7 = 'ls7'
    # DO NOT use this except in hessian case.
    MIX = 'mix'
    REAL = 'real'

# taylor series referenced from:
# https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/
def sigmoid_t1(x):
    T0 = 1.0 / 2
    T1 = 1.0 / 4
    return T0 + x * T1


def sigmoid_t3(x):
    T3 = -1.0 / 48
    return sigmoid_t1(x) + jnp.power(x, 3) * T3


def sigmoid_t5(x):
    T5 = 1.0 / 480
    return sigmoid_t3(x) + jnp.power(x, 5) * T5


# f(x) = 0.5 + 0.125x if -4 <= x <= 4
#        1            if       x > 4
#        0            if  -4 > x
def sigmoid_seg3(x):
    return jnp.select([x < -4, x > 4], [0, 1], 0.5 + x * 0.125)


# https://dergipark.org.tr/en/download/article-file/54559
# Dataflow implementation of sigmoid function:
# F(x) = 0.5 * ( x / ( 1 + |x| ) ) + 0.5
# sigmoid_df has higher precision than sigmoid_sr if x in [-2, 2]
def sigmoid_df(x):
    return 0.5 * (x / (1 + jnp.abs(x))) + 0.5


# https://en.wikipedia.org/wiki/Sigmoid_function#Examples
# Square Root approximation functions:
# F(x) = 0.5 * ( x / ( 1 + x^2 )^0.5 ) + 0.5
# sigmoid_sr almost perfect fit to sigmoid if x out of range [-3,3]
# highly recommended use this appr as GDBT's default sigmoid method.
def sigmoid_sr(x):
    return 0.5 * (x / jnp.sqrt(1 + jnp.power(x, 2))) + 0.5


# polynomial fitting of degree 7
def sigmoid_ls7(x):
    return (
        5.00052959e-01
        + 2.35176260e-01 * x
        - 3.97212202e-05 * jnp.power(x, 2)
        - 1.23407424e-02 * jnp.power(x, 3)
        + 4.04588962e-06 * jnp.power(x, 4)
        + 3.94330487e-04 * jnp.power(x, 5)
        - 9.74060972e-08 * jnp.power(x, 6)
        - 4.74674505e-06 * jnp.power(x, 7)
    )


#  mix ls7 & sr sig, use ls7 if |x| < 4 , else use sr.
#  has higher precision in all input range.
#  NOTICE: this method is very expensive, only use for hessian matrix.
def sigmoid_mix(x):
    ls7 = sigmoid_ls7(x)
    sr = sigmoid_sr(x)
    return jnp.select([x < -4, x > 4], [sr, sr], ls7)


# real computation of sigmoid
# NOTICE: should make sure x not too small
def sigmoid_real(x):
    return 1 / (1 + jnp.exp(-x))


def sigmoid(x, sig_type):
    if sig_type is SigType.T1:
        return sigmoid_t1(x)
    elif sig_type is SigType.T3:
        return sigmoid_t3(x)
    elif sig_type is SigType.T5:
        return sigmoid_t5(x)
    elif sig_type is SigType.SEG3:
        return sigmoid_seg3(x)
    elif sig_type is SigType.DF:
        return sigmoid_df(x)
    elif sig_type is SigType.SR:
        return sigmoid_sr(x)
    elif sig_type is SigType.LS7:
        return sigmoid_ls7(x)
    elif sig_type is SigType.MIX:
        return sigmoid_mix(x)
    elif sig_type is SigType.REAL:
        return sigmoid_real(x)