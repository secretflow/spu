# Copyright 2021 Ant Group Co., Ltd.
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
import pdb
import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from flax.linen.linear import Array
from typing import Any, Optional, Tuple, Union

import spu
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd
import spu.intrinsic as si



def bumblebee_gelu(x: Array) -> Array:
    b0 = x < -5.0
    b1 = x < -1.97
    b2 = x > 3.0
    b3 = b1 ^ b2 ^ True  # x in [-1.97, 3.0)
    b4 = b0 ^ b1  # x in [-5.0, -1.97)
    a_coeffs = jnp.array(
        [-0.5054031199708174, -0.4222658115198386, -0.1180761295118195, -0.0110341340306157]
    )
    b_coeffs = jnp.array(
        [
            0.0085263215410380,
            0.5,
            0.3603292692789629,
            -0.037688200365904,
            0.0018067462606141,
        ]
    )
    x2 = jnp.square(x)
    x3 = x2 * x
    x4 = jnp.square(x2)
    x6 = x2 * x4
    seg1 = a_coeffs[3] * x3 + a_coeffs[2] * x2 + a_coeffs[1]*x + a_coeffs[0]
    seg2 = (
        b_coeffs[4] * x6
        + b_coeffs[3] * x4
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )
    ret = b2 * x + b4 * seg1 + b3 * seg2
    return ret

    
def ours_fake_gelu(x: Array):
    order=2
    left,right=-1.8, 0.5
    
    zero_mask=x<left
    right_mask=x>right
    # poly_mask=jnp.logical_not(zero_mask) & jnp.logical_not(right_mask)
    poly_mask=zero_mask^right_mask^True
    
    if order==2:
        coeff = jnp.array([0.17025471961795552, 0.32421075145645123, -0.006946478426716862])
        
        # tmp1=x*coeff[0]+coeff[1]
        # computed_v=tmp1*x+coeff[2]
        x2=jnp.square(x)
        computed_v=coeff[0]*x2+coeff[1]*x+coeff[2]
    else:
        raise("error")
    x=right_mask*x+poly_mask*computed_v # +zero_mask*0
    
    return x            


def bumblebee_softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
) -> Array:
    x_max = jnp.max(x, axis, keepdims=True)
    x = x - x_max
    
    nexp=bumblebee_exp(x)
    
    divisor = jnp.sum(nexp, axis, keepdims=True)
    return nexp / divisor


    
def ours_fake_softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
) -> Array:
    threshold=-3.9
    
    x = x - jnp.max(x, axis=-1, keepdims=True)
    # exp on large negative is clipped to zero
    compute_mask = x > threshold
    
    
    computed_v=ours_fake_exp(x)
    
    x=computed_v * compute_mask
    
    divisor = jnp.sum(x, axis, keepdims=True)
    return x / divisor


def bumblebee_exp(
    x: Array,
) -> Array:
    b = x > -14
    nexp = jnp.exp(x) * b
    return nexp

def ours_fake_exp(
    x: Array,
) -> Array:
    coeff = jnp.array([0.0291786417536755, 0.26332572942924737, 0.8318065569430768, 0.9708981428645548])
    
    
    tmp1=x*coeff[0]+coeff[1]
    tmp2=x*tmp1+coeff[2]
    computed_v=x*tmp2+coeff[3]
    
    # x2=jnp.square(x)
    # x3=x2*x
    # computed_v=coeff[0]*x3+coeff[1]*x2+coeff[2]*x+coeff[3]
    return computed_v



if __name__ == "__main__":
    """
    You can modify the code below for debug purpose only.
    Please DONT commit it unless it will cause build break.
    """
    with open("examples/python/ml/Nimbus/2pc.json", 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"])

    copts = spu_pb2.CompilerOptions()
    # Tweak compiler options
    copts.disable_div_sqrt_rewrite = True
    copts.enable_optimize_denominator_with_broadcast = True

    parser = argparse.ArgumentParser(description='Process the test flags.')
    parser.add_argument('--test_oursfake_gelu', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_oursfake_softmax', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_oursfake_exp', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_bumblebeefake_gelu', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_bumblebeefake_softmax', action='store_true', help='Set TEST_SOFTMAX to True')
    parser.add_argument('--test_bumblebeefake_exp', action='store_true', help='Set TEST_SOFTMAX to True')

    parser.add_argument('--b', type=int, help='Integer argument b')
    parser.add_argument('--token_num', type=int, help='Integer argument token_num')
    parser.add_argument('--hidden', type=int, default=768, help='Integer argument hidden')

    # Parse the arguments
    args = parser.parse_args()
    test_oursfake_gelu = args.test_oursfake_gelu
    test_oursfake_softmax = args.test_oursfake_softmax
    test_oursfake_exp = args.test_oursfake_exp
    test_bumblebeefake_gelu = args.test_bumblebeefake_gelu
    test_bumblebeefake_softmax = args.test_bumblebeefake_softmax
    test_bumblebeefake_exp = args.test_bumblebeefake_exp

    b = args.b
    token_num = args.token_num
    hidden = args.hidden




    if test_bumblebeefake_gelu:
        x = np.random.randn(b, token_num, hidden)

        cpu_out = bumblebee_gelu(x)

        # print(ppd.device("SPU")(bumblebee_gelu).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            bumblebee_gelu, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_bumblebeefake_softmax:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = bumblebee_softmax(x)

        # print(ppd.device("SPU")(bumblebee_softmax).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            bumblebee_softmax, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())


    if test_bumblebeefake_exp:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = bumblebee_exp(x)

        # print(ppd.device("SPU")(bumblebee_exp).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            bumblebee_exp, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())


    if test_oursfake_gelu:
        x = np.random.randn(b, token_num, hidden)

        cpu_out = ours_fake_gelu(x)

        # print(ppd.device("SPU")(ours_fake_gelu).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            ours_fake_gelu, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())


    if test_oursfake_softmax:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = ours_fake_softmax(x)

        # print(ppd.device("SPU")(ours_fake_softmax).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            ours_fake_softmax, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.001).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

    if test_oursfake_exp:
        head_num=12

        x = np.random.randn(b, head_num, token_num, token_num)

        cpu_out = ours_fake_exp(x)

        # print(ppd.device("SPU")(ours_fake_exp).dump_pphlo(x))
        x = ppd.device("P2")(lambda x: x)(x)
        spu_out = ppd.device("SPU")(
            ours_fake_exp, copts=copts
        )(x)
        spu_out = ppd.get(spu_out)

        print(np.isclose(cpu_out, spu_out, atol=0.01).all(),(np.absolute(cpu_out-spu_out).max()),np.absolute(cpu_out-spu_out).mean())

