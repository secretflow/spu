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

import json
import unittest

import multiprocess
from google.protobuf import json_format

import spu.libspu.link as link
import spu.psi as psi
from spu.tests.utils import create_link_desc
from tempfile import TemporaryDirectory


class UnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir_ = TemporaryDirectory()
        return super().setUp()

    def tearDown(self) -> None:
        self.tempdir_.cleanup()
        return super().tearDown()

    def test_ub_psi(self):
        link_desc = create_link_desc(2)

        # offline stage
        server_offline_config = f'''
        {{
            "mode": "MODE_OFFLINE",
            "role": "ROLE_SERVER",
            "cache_path": "{self.tempdir_.name}/spu_test_ub_psi_server_cache",
            "input_config": {{
                "path": "spu/tests/data/alice.csv"
            }},
            "keys": [
                "id"
            ],
            "server_secret_key_path": "{self.tempdir_.name}/spu_test_ub_psi_server_secret_key.key"
        }}
        '''

        client_offline_config = f'''
        {{
            "mode": "MODE_OFFLINE",
            "role": "ROLE_CLIENT",
            "cache_path": "{self.tempdir_.name}/spu_test_ub_psi_client_cache"
        }}
        '''

        with open(
            f"{self.tempdir_.name}/spu_test_ub_psi_server_secret_key.key", 'wb'
        ) as f:
            f.write(
                bytes.fromhex(
                    "000102030405060708090a0b0c0d0e0ff0e0d0c0b0a090807060504030201000"
                )
            )

        configs = [
            json_format.ParseDict(json.loads(server_offline_config), psi.UbPsiConfig()),
            json_format.ParseDict(json.loads(client_offline_config), psi.UbPsiConfig()),
        ]

        def wrap(rank, link_desc, configs):
            link_ctx = link.create_brpc(link_desc, rank)
            psi.ub_psi(configs[rank], link_ctx)

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

        # online stage
        server_online_config = f'''
        {{
            "mode": "MODE_ONLINE",
            "role": "ROLE_SERVER",
            "server_secret_key_path": "{self.tempdir_.name}/spu_test_ub_psi_server_secret_key.key",
            "cache_path": "{self.tempdir_.name}/spu_test_ub_psi_server_cache"
        }}
        '''

        client_online_config = f'''
        {{
            "mode": "MODE_ONLINE",
            "role": "ROLE_CLIENT",
            "input_config": {{
                "path": "spu/tests/data/bob.csv"
            }},
            "output_config": {{
                "path": "{self.tempdir_.name}/spu_test_ubpsi_bob_psi_ouput.csv"
            }},
            "keys": [
                "id"
            ],
            "cache_path": "{self.tempdir_.name}/spu_test_ub_psi_client_cache"
        }}
        '''

        configs = [
            json_format.ParseDict(json.loads(server_online_config), psi.UbPsiConfig()),
            json_format.ParseDict(json.loads(client_online_config), psi.UbPsiConfig()),
        ]

        link_desc = create_link_desc(2)

        def wrap(rank, link_desc, configs):
            link_ctx = link.create_brpc(link_desc, rank)
            psi.ub_psi(configs[rank], link_ctx)

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
