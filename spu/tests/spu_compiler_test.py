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


import os
import unittest

import numpy as np
import numpy.testing as npt

import spu.utils.frontend as spu_fe
import spu.spu_pb2 as spu_pb2


class UnitTests(unittest.TestCase):
    def test_compile_pb(self):
        def test():
            return 1

        result, *_ = spu_fe.compile(
            spu_fe.Kind.JAX,
            test,
            list(),
            dict(),
            [],
            [],
            lambda _: ["out1"],
        )

        # inspect compiled result
        ir = result.code.decode("utf-8")
        self.assertIn("@main", ir)
        self.assertIn("pphlo", ir)


if __name__ == '__main__':
    unittest.main()
