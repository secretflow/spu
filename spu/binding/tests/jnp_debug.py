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


import unittest

import jax.numpy as jnp
import numpy as np

import spu.binding.util.simulation as ppsim
import spu.spu_pb2 as spu_pb2

if __name__ == "__main__":
    """
    You can modify the code below for debug purpose only.
    Please DONT commit it unless it will cause build break.
    """

    sim = ppsim.Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)

    x = np.random.randint(low=-100, high=100, size=(3, 5))
    fn = lambda x: jnp.argmax(x, axis=0)
    spu_fn = ppsim.sim_jax(sim, fn)
    z = spu_fn(x)

    print(f'spu out = {z}')
    print(f'cpu out = {fn(x)}')
