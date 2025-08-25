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

import spu.libspu.link as link  # type: ignore
import spu.psi as psi
from spu.tests.utils import build_link_desc, create_link_desc_dict
from spu.utils.polyfill import Process


class UnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir_ = TemporaryDirectory()
        return super().setUp()

    def tearDown(self) -> None:
        self.tempdir_.cleanup()
        return super().tearDown()

    @staticmethod
    def _run_psi(rank, link_desc, config_json: dict):
        input_params = config_json.get("input_params")
        if input_params:
            config_json["input_params"] = psi.InputParams(**input_params)
        output_params = config_json.get("output_params")
        if output_params:
            config_json["output_params"] = psi.OutputParams(**output_params)
        config = psi.UbPsiExecuteConfig(**config_json)

        lctx_desc = build_link_desc(link_desc)
        link_ctx = link.create_brpc(lctx_desc, rank)
        psi.ub_psi_execute(config, link_ctx)

    def test_ub_psi(self):
        link_desc = create_link_desc_dict(2)

        configs = [
            {
                "mode": psi.UbPsiMode.MODE_OFFLINE,
                "role": psi.UbPsiRole.ROLE_SERVER,
                "cache_path": f"{self.tempdir_.name}/spu_test_ub_psi_server_cache",
                "input_params": {
                    "type": psi.SourceType.SOURCE_TYPE_FILE_CSV,
                    "path": "spu/tests/data/alice.csv",
                    "selected_keys": ["id"],
                },
            },
            {
                "mode": psi.UbPsiMode.MODE_OFFLINE,
                "role": psi.UbPsiRole.ROLE_CLIENT,
                "cache_path": f"{self.tempdir_.name}/spu_test_ub_psi_client_cache",
            },
        ]

        jobs = [
            Process(
                target=UnitTests._run_psi,
                args=(rank, link_desc, configs[rank]),
            )
            for rank in range(2)
        ]

        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)

        configs = [
            {
                "mode": psi.UbPsiMode.MODE_ONLINE,
                "role": psi.UbPsiRole.ROLE_SERVER,
                "cache_path": f"{self.tempdir_.name}/spu_test_ub_psi_server_cache",
                "output_params": {
                    "type": psi.SourceType.SOURCE_TYPE_FILE_CSV,
                },
            },
            {
                "mode": psi.UbPsiMode.MODE_ONLINE,
                "role": psi.UbPsiRole.ROLE_CLIENT,
                "cache_path": f"{self.tempdir_.name}/spu_test_ub_psi_client_cache",
                "input_params": {
                    "type": psi.SourceType.SOURCE_TYPE_FILE_CSV,
                    "path": "spu/tests/data/bob.csv",
                    "selected_keys": ["id"],
                },
                "output_params": {
                    "type": psi.SourceType.SOURCE_TYPE_FILE_CSV,
                    "path": f"{self.tempdir_.name}/spu_test_ubpsi_bob_psi_ouput.csv",
                },
            },
        ]

        link_desc = create_link_desc_dict(2)

        jobs = [
            Process(
                target=UnitTests._run_psi,
                args=(rank, link_desc, configs[rank]),
            )
            for rank in range(2)
        ]
        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)


if __name__ == '__main__':
    unittest.main()
