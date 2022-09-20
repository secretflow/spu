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


import multiprocess
import subprocess
import time
import unittest

import spu.binding.psi as psi
import spu.binding._lib.link as link
from spu.binding.util.simulation import PropagatingThread
from socket import socket


def get_free_port():
    with socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def wc_count(file_name):
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])


class UnitTests(unittest.TestCase):
    def run_psi(self, fn):
        wsize = 2

        lctx_desc = link.Desc()
        for rank in range(wsize):
            lctx_desc.add_party(f"id_{rank}", f"thread_{rank}")

        def wrap(rank):
            lctx = link.create_mem(lctx_desc, rank)
            return fn(lctx)

        jobs = [PropagatingThread(target=wrap, args=(rank,)) for rank in range(wsize)]

        [job.start() for job in jobs]
        [job.join() for job in jobs]

    def run_streaming_psi(self, wsize, inputs, outputs, selected_fields, protocol):
        time_stamp = time.time()
        lctx_desc = link.Desc()
        lctx_desc.id = str(round(time_stamp * 1000))

        for rank in range(wsize):
            port = get_free_port()
            lctx_desc.add_party(f"id_{rank}", f"127.0.0.1:{port}")

        global wrap

        def wrap(rank, selected_fields, input_path, output_path, type):
            lctx = link.create_brpc(lctx_desc, rank)

            config = psi.BucketPsiConfig(
                psi_type=type,
                broadcast_result=True,
                input_params=psi.InputParams(
                    path=input_path, select_fields=selected_fields
                ),
                output_params=psi.OuputParams(path=output_path, need_sort=True),
            )
            report = psi.bucket_psi(lctx, config)

            source_count = wc_count(input_path)
            output_count = wc_count(output_path)
            print(
                f"id:{lctx.id()}, psi_type: {type}, original_count: {report.original_count}, intersection_count: {report.intersection_count}, source_count: {source_count}, output_count: {output_count}"
            )

            self.assertEqual(report.original_count, source_count - 1)
            self.assertEqual(report.intersection_count, output_count - 1)

        # launch with multiprocess
        jobs = [
            multiprocess.Process(
                target=wrap,
                args=(
                    rank,
                    selected_fields,
                    inputs[rank],
                    outputs[rank],
                    protocol,
                ),
            )
            for rank in range(wsize)
        ]
        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)

    def prep_data(self):
        data = [
            [f'r{idx}' for idx in range(1000) if idx % 3 == 0],
            [f'r{idx}' for idx in range(1000) if idx % 7 == 0],
        ]

        expected = [f'r{idx}' for idx in range(1000) if idx % 3 == 0 and idx % 7 == 0]

        return data, expected

    def test_reveal(self):
        data, expected = self.prep_data()
        expected.sort()

        def fn(lctx):
            config = psi.MemoryPsiConfig(
                psi_type=psi.PsiType.ECDH_PSI_2PC, broadcast_result=True
            )
            joint = psi.mem_psi(lctx, config, data[lctx.rank])
            joint.sort()
            return self.assertEqual(joint, expected)

        self.run_psi(fn)

    def test_reveal_to(self):
        data, expected = self.prep_data()
        expected.sort()

        reveal_to_rank = 0

        def fn(lctx):
            config = psi.MemoryPsiConfig(
                psi_type=psi.PsiType.KKRT_PSI_2PC,
                receiver_rank=reveal_to_rank,
                broadcast_result=False,
            )
            joint = psi.mem_psi(lctx, config, data[lctx.rank])

            joint.sort()

            if lctx.rank == reveal_to_rank:
                self.assertEqual(joint, expected)
            else:
                self.assertEqual(joint, [])

        self.run_psi(fn)

    def test_ecdh_3pc(self):
        print("----------test_ecdh_3pc-------------")

        inputs = [
            "spu/binding/tests/data/alice.csv",
            "spu/binding/tests/data/bob.csv",
            "spu/binding/tests/data/carol.csv",
        ]
        outputs = ["./alice-ecdh3pc.csv", "./bob-ecdh3pc.csv", "./carol-ecdh3pc.csv"]
        selected_fields = ["id", "idx"]

        self.run_streaming_psi(
            3, inputs, outputs, selected_fields, psi.PsiType.ECDH_PSI_3PC
        )

    def test_kkrt_2pc(self):
        print("----------test_kkrt_2pc-------------")

        inputs = ["spu/binding/tests/data/alice.csv", "spu/binding/tests/data/bob.csv"]
        outputs = ["./alice-kkrt.csv", "./bob-kkrt.csv"]
        selected_fields = ["id", "idx"]

        self.run_streaming_psi(
            2, inputs, outputs, selected_fields, psi.PsiType.KKRT_PSI_2PC
        )

    def test_ecdh_2pc(self):
        print("----------test_ecdh_2pc-------------")

        inputs = ["spu/binding/tests/data/alice.csv", "spu/binding/tests/data/bob.csv"]
        outputs = ["./alice-ecdh.csv", "./bob-ecdh.csv"]
        selected_fields = ["id", "idx"]

        self.run_streaming_psi(
            2, inputs, outputs, selected_fields, psi.PsiType.ECDH_PSI_2PC
        )


if __name__ == '__main__':
    unittest.main()
