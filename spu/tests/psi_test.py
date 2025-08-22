# Copyright 2024 Ant Group Co., Ltd.
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
from pathlib import Path
from tempfile import TemporaryDirectory

import spu.libspu.link as link  # type: ignore
import spu.psi as psi
from spu.tests.utils import build_link_desc, create_link_desc_dict, wc_count
from spu.utils.polyfill import Process


class UnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir_ = TemporaryDirectory()
        return super().setUp()

    def tearDown(self) -> None:
        self.tempdir_.cleanup()
        return super().tearDown()

    @staticmethod
    def run_psi(rank, link_desc, input_path, output_path):
        lctx_desc = build_link_desc(link_desc)
        link_ctx = link.create_brpc(lctx_desc, rank)
        config = psi.PsiExecuteConfig(
            protocol_conf=psi.PsiProtocolConfig(
                protocol=psi.PsiProtocol.PROTOCOL_ECDH,
                receiver_rank=0,
                ecdh_params=psi.EcdhParams(curve=psi.EllipticCurveType.CURVE_25519),
                broadcast_result=True,
            ),
            input_params=psi.InputParams(
                type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
                path=input_path,
                selected_keys=["id"],
                keys_unique=True,
            ),
            output_params=psi.OutputParams(
                type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
                path=output_path,
                disable_alignment=True,
            ),
        )
        psi.psi_execute(config, link_ctx)

    def test_psi(self):
        link_desc = create_link_desc_dict(2)
        jobs = []
        for rank in range(2):
            party = "alice" if rank == 0 else "bob"
            input_path = f"spu/tests/data/{party}.csv"
            output_path = f"{self.tempdir_.name}/spu_test_psi_{party}_psi_ouput.csv"
            job = Process(
                target=UnitTests.run_psi,
                args=(rank, link_desc, input_path, output_path),
            )
            jobs.append(job)

        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)

        self.assertEqual(
            wc_count(f"{self.tempdir_.name}/spu_test_psi_alice_psi_ouput.csv"),
            wc_count(f"{self.tempdir_.name}/spu_test_psi_bob_psi_ouput.csv"),
        )
        # clear file
        base_dir = Path(__file__).parent
        for name in ['bob.csv.meta', 'alice.csv.meta']:
            file_path = base_dir / 'data' / name
            if file_path.exists():
                file_path.unlink()


if __name__ == '__main__':
    unittest.main()
