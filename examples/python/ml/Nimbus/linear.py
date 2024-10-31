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

import sys
import json
import pdb
import unittest
import itertools
import math
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

import spu.intrinsic as si
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd


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

    if len(sys.argv) == 5:
        b = int(sys.argv[1])
        m = int(sys.argv[2])
        k = int(sys.argv[3])
        n = int(sys.argv[4])
    else:
        b=1
        m,k,n=3072,768,128

    pn,pk=8,8
    unused1 = np.ones((b, k, n), dtype=np.bool_)
    unused2 = np.array([pk, pn], dtype=np.int64)

    x = np.random.randn(b, m, k)
    y = np.random.randn(b, k, n)


    fn = lambda x, y: si.sparse_dot_general(
        x,
        y,
        unused1,
        unused2,
        dimension_numbers=(((x.ndim - 1,), (1,)), ((0,), (0,))),
    )

    cpu_out=np.matmul(x, y)
    x = ppd.device("P1")(lambda x: x)(x)
    y = ppd.device("P2")(lambda x: x)(y)
    spu_out = ppd.device("SPU")(
        fn, copts=copts
    )(x,y)
    spu_out = ppd.get(spu_out)


    # print("cpu_out", cpu_out)
    # print("spu_out", spu_out)

    diff=cpu_out-spu_out
    diff[np.absolute(diff)<0.001]=0
    diff[np.absolute(diff)>0.001]=1
    print("correctness ratio:",1-np.mean(diff))
    print("Overall correctness:",np.isclose(cpu_out, spu_out, atol=0.0005).all())
    print("maximal difference:",np.max(np.absolute(cpu_out-spu_out)))
    print("mean difference:",np.mean(np.absolute(cpu_out-spu_out)))
