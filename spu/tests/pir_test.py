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
from tempfile import TemporaryDirectory

import multiprocess
from google.protobuf import json_format

import spu.libspu.link as link
import spu.psi as psi
from spu.tests.utils import create_link_desc, wc_count


class UnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir_ = TemporaryDirectory()
        return super().setUp()

    def tearDown(self) -> None:
        self.tempdir_.cleanup()
        return super().tearDown()

    def test_pir(self):
        # setup stage
        server_setup_config = f'''
        {{
            "mode": "MODE_SERVER_SETUP",
            "pir_protocol": "PIR_PROTOCOL_KEYWORD_PIR_APSI",
            "pir_server_config": {{
                "input_path": "spu/tests/data/alice.csv",
                "setup_path": "{self.tempdir_.name}/spu_test_pir_pir_server_setup",
                "key_columns": [
                    "id"
                ],
                "label_columns": [
                    "y"
                ],
                "label_max_len": 288,
                "bucket_size": 1000000,
                "apsi_server_config": {{
                    "oprf_key_path": "{self.tempdir_.name}/spu_test_pir_server_secret_key.bin",
                    "num_per_query": 1,
                    "compressed": false
                }}
            }}
        }}
        '''

        with open(
            f"{self.tempdir_.name}/spu_test_pir_server_secret_key.bin", 'wb'
        ) as f:
            f.write(
                bytes.fromhex(
                    "000102030405060708090a0b0c0d0e0ff0e0d0c0b0a090807060504030201000"
                )
            )

        psi.pir(json_format.ParseDict(json.loads(server_setup_config), psi.PirConfig()))

        server_online_config = f'''
        {{
            "mode": "MODE_SERVER_ONLINE",
            "pir_protocol": "PIR_PROTOCOL_KEYWORD_PIR_APSI",
            "pir_server_config": {{
                "setup_path": "{self.tempdir_.name}/spu_test_pir_pir_server_setup"
            }}
        }}
        '''

        client_online_config = f'''
        {{
            "mode": "MODE_CLIENT",
            "pir_protocol": "PIR_PROTOCOL_KEYWORD_PIR_APSI",
            "pir_client_config": {{
                "input_path": "{self.tempdir_.name}/spu_test_pir_pir_client.csv",
                "key_columns": [
                    "id"
                ],
                "output_path": "{self.tempdir_.name}/spu_test_pir_pir_output.csv"
            }}
        }}
        '''

        pir_client_input_content = '''id
user808
xxx
'''

        with open(f"{self.tempdir_.name}/spu_test_pir_pir_client.csv", 'w') as f:
            f.write(pir_client_input_content)

        configs = [
            json_format.ParseDict(json.loads(server_online_config), psi.PirConfig()),
            json_format.ParseDict(json.loads(client_online_config), psi.PirConfig()),
        ]

        link_desc = create_link_desc(2)

        def wrap(rank, link_desc, configs):
            link_ctx = link.create_brpc(link_desc, rank)
            psi.pir(configs[rank], link_ctx)

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

        # including title, actual matched item cnt is 1.
        self.assertEqual(
            wc_count(f"{self.tempdir_.name}/spu_test_pir_pir_output.csv"), 2
        )


if __name__ == '__main__':
    unittest.main()
