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

import jax.numpy as jnp
import numpy as np

import spu.libspu as libspu
import spu.utils.simulation as ppsim

if __name__ == "__main__":
    """
    You can modify the code below for debug purpose only.
    Please DONT commit it unless it will cause build break.
    """

    sim = ppsim.Simulator.simple(3, libspu.ProtocolKind.ABY3, libspu.FieldType.FM64)
    copts = libspu.CompilerOptions()
    # Tweak compiler options
    copts.disable_div_sqrt_rewrite = True

    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3]) * 1.0
    fn = lambda x, y: jnp.power(x, y)

    spu_fn = ppsim.sim_jax(sim, fn, copts=copts)
    z = spu_fn(x, y)

    print(spu_fn.pphlo)

    print(f"spu out = {z}")
    print(f"cpu out = {fn(x, y)}")
