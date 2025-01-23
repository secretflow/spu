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

import numpy as np

import spu.spu_pb2 as spu_pb2
import spu.utils.simulation as ppsim
from spu.tests.jnp_testbase import JnpTests


class JnpTestSemi2kFM128(JnpTests.JnpTestBase):
    def setUp(self):
        self._sim = ppsim.Simulator.simple(
            3, spu_pb2.ProtocolKind.SEMI2K, spu_pb2.FieldType.FM128
        )
        self._rng = np.random.RandomState()


class JnpTestSemi2kFM128TwoParty(JnpTests.JnpTestBase):
    def setUp(self):
        config = spu_pb2.RuntimeConfig(
            protocol=spu_pb2.ProtocolKind.SEMI2K, field=spu_pb2.FieldType.FM128
        )
        config.experimental_enable_exp_prime = True
        config.experimental_exp_prime_enable_upper_bound = True
        config.experimental_exp_prime_offset = 13
        config.experimental_exp_prime_disable_lower_bound = False
        config.fxp_exp_mode = spu_pb2.RuntimeConfig.ExpMode.EXP_PRIME
        self._sim = ppsim.Simulator(2, config)
        self._rng = np.random.RandomState()


if __name__ == "__main__":
    unittest.main()
