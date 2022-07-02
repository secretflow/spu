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


import multiprocessing
import os
import re
import sys
import threading
import unittest

import spu.binding._lib.link as link


class UnitTests(unittest.TestCase):
    def test_link_brpc(self):
        desc = link.Desc()
        desc.add_party("alice", "127.0.0.1:9927")
        desc.add_party("bob", "127.0.0.1:9928")

        # Pickle only works properly for top-level functions, so mark proc as global to workaround this limitation
        # See https://stackoverflow.com/questions/56533827/pool-apply-async-nested-function-is-not-executed/56534386#56534386
        global proc

        def proc(rank):
            data = "hello" if rank == 0 else "world"

            lctx = link.create_brpc(desc, rank)
            res = lctx.all_gather(data)

            self.assertEqual(res, ['hello', 'world'])

        # launch with multiprocessing
        jobs = [
            multiprocessing.Process(target=proc, args=(0,)),
            multiprocessing.Process(target=proc, args=(1,)),
        ]
        [job.start() for job in jobs]
        [job.join() for job in jobs]

    def test_link_mem(self):
        desc = link.Desc()
        desc.add_party("alice", "thread_0")
        desc.add_party("bob", "thread_1")

        def proc(rank):
            data = "hello" if rank == 0 else "world"

            lctx = link.create_mem(desc, rank)
            res = lctx.all_gather(data)

            self.assertEqual(res, ['hello', 'world'])

        # launch with threading
        jobs = [
            threading.Thread(target=proc, args=(0,)),
            threading.Thread(target=proc, args=(1,)),
        ]

        [job.start() for job in jobs]
        [job.join() for job in jobs]


if __name__ == '__main__':
    unittest.main()
