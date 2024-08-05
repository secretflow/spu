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
import re
import sys
import threading
import unittest

import multiprocess

import spu.libspu.link as link


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

            lctx.stop_link()

        # launch with multiprocess
        jobs = [
            multiprocess.Process(target=proc, args=(0,)),
            multiprocess.Process(target=proc, args=(1,)),
        ]
        [job.start() for job in jobs]
        [job.join() for job in jobs]

        for j in jobs:
            self.assertEqual(j.exitcode, 0)

    def test_link_mem(self):
        desc = link.Desc()
        desc.add_party("alice", "thread_0")
        desc.add_party("bob", "thread_1")

        def proc():
            def thread(rank):
                data = "hello" if rank == 0 else "world"

                lctx = link.create_mem(desc, rank)
                res = lctx.all_gather(data)

                self.assertEqual(res, ['hello', 'world'])

                lctx.stop_link()

            # launch with threading
            jobs = [
                threading.Thread(target=thread, args=(0,)),
                threading.Thread(target=thread, args=(1,)),
            ]

            [job.start() for job in jobs]
            [job.join() for job in jobs]

        job = multiprocess.Process(target=proc)
        job.start()
        job.join()
        self.assertEqual(job.exitcode, 0)

    def test_link_send_recv(self):
        desc = link.Desc()
        desc.add_party("alice", "127.0.0.1:9927")
        desc.add_party("bob", "127.0.0.1:9928")

        # Pickle only works properly for top-level functions, so mark proc as global to workaround this limitation
        # See https://stackoverflow.com/questions/56533827/pool-apply-async-nested-function-is-not-executed/56534386#56534386
        global proc

        def proc(rank):
            lctx = link.create_brpc(desc, rank)
            if rank == 0:
                lctx.send(1, b"hello world")
            else:
                s = lctx.recv(0)
                self.assertEqual(s, b"hello world")

            lctx.stop_link()

        # launch with multiprocess
        jobs = [
            multiprocess.Process(target=proc, args=(0,)),
            multiprocess.Process(target=proc, args=(1,)),
        ]
        [job.start() for job in jobs]
        [job.join() for job in jobs]

        for j in jobs:
            self.assertEqual(j.exitcode, 0)

    def test_link_send_async(self):
        desc = link.Desc()
        desc.add_party("alice", "127.0.0.1:9927")
        desc.add_party("bob", "127.0.0.1:9928")

        # Pickle only works properly for top-level functions, so mark proc as global to workaround this limitation
        # See https://stackoverflow.com/questions/56533827/pool-apply-async-nested-function-is-not-executed/56534386#56534386
        global proc

        def proc(rank):
            lctx = link.create_brpc(desc, rank)
            dst_rank = (rank + 1) % 2
            lctx.send_async(dst_rank, f"hello world {rank}".encode())
            self.assertEqual(lctx.recv(dst_rank), f"hello world {dst_rank}".encode())

            lctx.stop_link()

        # launch with multiprocess
        jobs = [
            multiprocess.Process(target=proc, args=(0,)),
            multiprocess.Process(target=proc, args=(1,)),
        ]
        [job.start() for job in jobs]
        [job.join() for job in jobs]

        for j in jobs:
            self.assertEqual(j.exitcode, 0)

    def test_link_next_rank(self):
        desc = link.Desc()
        desc.add_party("alice", "127.0.0.1:9927")
        desc.add_party("bob", "127.0.0.1:9928")

        # Pickle only works properly for top-level functions, so mark proc as global to workaround this limitation
        # See https://stackoverflow.com/questions/56533827/pool-apply-async-nested-function-is-not-executed/56534386#56534386
        global proc

        def proc(rank):
            lctx = link.create_brpc(desc, rank)
            next_rank = (rank + 1) % 2
            self.assertEqual(lctx.next_rank(), next_rank)
            self.assertEqual(lctx.next_rank(2), rank)

            lctx.stop_link()

        # launch with multiprocess
        jobs = [
            multiprocess.Process(target=proc, args=(0,)),
            multiprocess.Process(target=proc, args=(1,)),
        ]
        [job.start() for job in jobs]
        [job.join() for job in jobs]

        for j in jobs:
            self.assertEqual(j.exitcode, 0)


if __name__ == '__main__':
    unittest.main()
