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

    def test_psi(self):
        link_desc = create_link_desc(2)

        receiver_config_json = f'''
        {{
            "protocol_config": {{
                "protocol": "PROTOCOL_ECDH",
                "ecdh_config": {{
                    "curve": "CURVE_25519"
                }},
                "role": "ROLE_RECEIVER",
                "broadcast_result": true
            }},
            "input_config": {{
                "type": "IO_TYPE_FILE_CSV",
                "path": "spu/tests/data/alice.csv"
            }},
            "output_config": {{
                "type": "IO_TYPE_FILE_CSV",
                "path": "{self.tempdir_.name}/spu_test_psi_alice_psi_ouput.csv"
            }},
            "keys": [
                "id"
            ],
            "skip_duplicates_check": true,
            "disable_alignment": true
        }}
        '''

        sender_config_json = f'''
        {{
            "protocol_config": {{
                "protocol": "PROTOCOL_ECDH",
                "ecdh_config": {{
                    "curve": "CURVE_25519"
                }},
                "role": "ROLE_SENDER",
                "broadcast_result": true
            }},
            "input_config": {{
                "type": "IO_TYPE_FILE_CSV",
                "path": "spu/tests/data/bob.csv"
            }},
            "output_config": {{
                "type": "IO_TYPE_FILE_CSV",
                "path": "{self.tempdir_.name}/spu_test_psi_bob_psi_ouput.csv"
            }},
            "keys": [
                "id"
            ],
            "skip_duplicates_check": true,
            "disable_alignment": true
        }}
        '''

        configs = [
            json_format.ParseDict(json.loads(receiver_config_json), psi.PsiConfig()),
            json_format.ParseDict(json.loads(sender_config_json), psi.PsiConfig()),
        ]

        def wrap(rank, link_desc, configs):
            link_ctx = link.create_brpc(link_desc, rank)
            psi.psi(configs[rank], link_ctx)

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

        self.assertEqual(
            wc_count(f"{self.tempdir_.name}/spu_test_psi_alice_psi_ouput.csv"),
            wc_count(f"{self.tempdir_.name}/spu_test_psi_bob_psi_ouput.csv"),
        )


if __name__ == '__main__':
    unittest.main()
