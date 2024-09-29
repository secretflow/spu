# Copyright 2022 Ant Group Co., Ltd.
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

### TODO: move to sf


# taylor series referenced from:
# https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/
def t1_sig(x):
    T0 = 1.0 / 2
    T1 = 1.0 / 4
    return T0 + x * T1


def t3_sig(x):
    T3 = -1.0 / 48
    return t1_sig(x) + jnp.power(x, 3) * T3


def t5_sig(x):
    T5 = 1.0 / 480
    return t3_sig(x) + jnp.power(x, 5) * T5


# f(x) = 0.5 + 0.125x if -4 <= x <= 4
#        1            if       x > 4
#        0            if  -4 > x
def seg3_sig(x):
    return jnp.select([x < -4, x > 4], [0, 1], 0.5 + x * 0.125)


# https://dergipark.org.tr/en/download/article-file/54559
# Dataflow implementation of sigmoid function:
# F(x) = 0.5 * ( x / ( 1 + |x| ) ) + 0.5
# df_sig has higher precision than sr_sig if x in [-2, 2]
def df_sig(x):
    return 0.5 * (x / (1 + jnp.abs(x))) + 0.5


# https://en.wikipedia.org/wiki/Sigmoid_function#Examples
# Square Root approximation functions:
# F(x) = 0.5 * ( x / ( 1 + x^2 )^0.5 ) + 0.5
# sr_sig almost perfect fit to sigmoid if x out of range [-3,3]
# highly recommended use this appr as GDBT's default sigmoid method.
def sr_sig(x):
    return 0.5 * (x / jnp.sqrt(1 + jnp.power(x, 2))) + 0.5
