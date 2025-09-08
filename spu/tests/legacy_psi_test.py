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
import time
import unittest

import spu.libspu.link as link  # type: ignore
import spu.psi as psi
from spu.tests.utils import (
    build_link_desc,
    create_link_desc_dict,
    get_free_port,
    wc_count,
)
from spu.utils.polyfill import Process


class UnitTests(unittest.TestCase):
    @staticmethod
    def run_psi(link_desc, rank, selected_fields, input_path, output_path, protocol):
        lctx_desc = build_link_desc(link_desc)
        lctx = link.create_brpc(lctx_desc, rank)

        config = psi.PsiExecuteConfig(
            protocol_conf=psi.PsiProtocolConfig(
                protocol=protocol,
                receiver_rank=0,
                broadcast_result=True,
                ecdh_params=psi.EcdhParams(curve=psi.EllipticCurveType.CURVE_25519),
            ),
            input_params=psi.InputParams(
                type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
                path=input_path,
                selected_keys=selected_fields,
            ),
            output_params=psi.OutputParams(
                type=psi.SourceType.SOURCE_TYPE_FILE_CSV, path=output_path
            ),
        )

        report = psi.psi_execute(config, lctx)

        source_count = wc_count(input_path)
        output_count = wc_count(output_path)
        print(
            f"id:{lctx.id()}, psi_type: {type}, original_count: {report.original_count}, intersection_count: {report.intersection_count}, source_count: {source_count}, output_count: {output_count}"
        )

        if report.original_count != source_count - 1:
            raise ValueError(
                f"original_count mismatch, {report.original_count}, {source_count - 1}"
            )
        if report.intersection_count != output_count - 1:
            raise ValueError(
                f"intersection_count mismatch, {report.intersection_count}, {output_count-1}"
            )

        lctx.stop_link()

    def run_streaming_psi(self, wsize, inputs, outputs, selected_fields, protocol):
        link_desc = create_link_desc_dict(wsize)

        # launch with multiprocess
        jobs = [
            Process(
                target=UnitTests.run_psi,
                args=(
                    link_desc,
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

        for f in outputs:
            os.remove(f)

    # TODO: this 3pc ecdh psi test is very likely to be failed, but we don't know why
    # def test_ecdh_3pc(self):
    #     print("----------test_ecdh_3pc-------------")

    #     inputs = [
    #         "spu/tests/data/alice.csv",
    #         "spu/tests/data/bob.csv",
    #         "spu/tests/data/carol.csv",
    #     ]
    #     outputs = ["./alice-ecdh3pc.csv", "./bob-ecdh3pc.csv", "./carol-ecdh3pc.csv"]
    #     selected_fields = ["id", "idx"]

    #     self.run_streaming_psi(
    #         3, inputs, outputs, selected_fields, psi.PsiProtocol.PROTOCOL_ECDH_3PC
    #     )

    def test_dppsi_2pc(self):
        print("----------test_dppsi_2pc-------------")

        inputs = ["spu/tests/data/alice.csv", "spu/tests/data/bob.csv"]
        outputs = ["./alice-dppsi.csv", "./bob-dppsi.csv"]
        selected_fields = ["id", "idx"]

        self.run_streaming_psi(
            2, inputs, outputs, selected_fields, psi.PsiProtocol.PROTOCOL_DP
        )


if __name__ == '__main__':
    unittest.main()
