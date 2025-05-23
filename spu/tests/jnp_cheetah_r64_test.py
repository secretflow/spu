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

import spu.libspu as libspu
import spu.utils.simulation as ppsim
from spu.tests.jnp_testbase import JnpTests


class JnpTestCheetahFM64(JnpTests.JnpTestBase):
    def setUp(self):
        self._sim = ppsim.Simulator.simple(
            2, libspu.ProtocolKind.CHEETAH, libspu.FieldType.FM64
        )
        self._rng = np.random.RandomState()


if __name__ == "__main__":
    unittest.main()
