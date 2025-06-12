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
from tempfile import TemporaryDirectory

import multiprocess

import spu.libspu.link as link
import spu.psi as psi
from spu.tests.utils import create_link_desc


class UnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir_ = TemporaryDirectory()
        return super().setUp()

    def tearDown(self) -> None:
        self.tempdir_.cleanup()
        return super().tearDown()

    def test_ub_psi(self):
        link_desc = create_link_desc(2)

        configs = [
            psi.UbPsiExecuteConfig(
                mode=psi.UbPsiMode.MODE_OFFLINE,
                role=psi.UbPsiRole.ROLE_SERVER,
                cache_path=f"{self.tempdir_.name}/spu_test_ub_psi_server_cache",
                input_params=psi.InputParams(
                    type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
                    path="spu/tests/data/alice.csv",
                    selected_keys=["id"],
                ),
            ),
            psi.UbPsiExecuteConfig(
                mode=psi.UbPsiMode.MODE_OFFLINE,
                role=psi.UbPsiRole.ROLE_CLIENT,
                cache_path=f"{self.tempdir_.name}/spu_test_ub_psi_client_cache",
            ),
        ]

        def wrap(rank, link_desc, configs):
            link_ctx = link.create_brpc(link_desc, rank)
            psi.ub_psi_execute(configs[rank], link_ctx)

        jobs = [
            multiprocess.Process(
                target=wrap,
                args=(rank, link_desc, configs),
            )
            for rank in range(2)
        ]
        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)

        configs = [
            psi.UbPsiExecuteConfig(
                mode=psi.UbPsiMode.MODE_ONLINE,
                role=psi.UbPsiRole.ROLE_SERVER,
                cache_path=f"{self.tempdir_.name}/spu_test_ub_psi_server_cache",
                output_params=psi.OutputParams(
                    type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
                ),
            ),
            psi.UbPsiExecuteConfig(
                mode=psi.UbPsiMode.MODE_ONLINE,
                role=psi.UbPsiRole.ROLE_CLIENT,
                cache_path=f"{self.tempdir_.name}/spu_test_ub_psi_client_cache",
                input_params=psi.InputParams(
                    type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
                    path="spu/tests/data/bob.csv",
                    selected_keys=["id"],
                ),
                output_params=psi.OutputParams(
                    type=psi.SourceType.SOURCE_TYPE_FILE_CSV,
                    path=f"{self.tempdir_.name}/spu_test_ubpsi_bob_psi_ouput.csv",
                ),
            ),
        ]

        link_desc = create_link_desc(2)

        def wrap(rank, link_desc, configs):
            link_ctx = link.create_brpc(link_desc, rank)
            psi.ub_psi_execute(configs[rank], link_ctx)

        jobs = [
            multiprocess.Process(
                target=wrap,
                args=(rank, link_desc, configs),
            )
            for rank in range(2)
        ]
        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)


if __name__ == '__main__':
    unittest.main()
