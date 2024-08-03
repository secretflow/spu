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
import tempfile
import unittest

import multiprocess
import spu.libspu.link as link
import spu.psi as psi
from google.protobuf import json_format
from spu.tests.utils import create_link_desc


class UnitTests(unittest.TestCase):

    def test_pir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # setup stage
            sender_setup_config_json = f'''
            {{
                "db_file": "spu/tests/data/db.csv",
                "params_file": "spu/tests/data/100K-1-16.json",
                "sdb_out_file": "{temp_dir}/sdb",
                "save_db_only": true
            }}
            '''

            psi.apsi_send(
                json_format.ParseDict(
                    json.loads(sender_setup_config_json), psi.ApsiSenderConfig()
                )
            )

            sender_online_config_json = f'''
            {{
                "db_file": "{temp_dir}/sdb"
            }}
            '''

            receiver_online_config_json = f'''
            {{
                "query_file": "spu/tests/data/query.csv",
                "output_file": "{temp_dir}/result.csv",
                "params_file": "spu/tests/data/100K-1-16.json"
            }}
            '''

            sender_online_config = json_format.ParseDict(
                json.loads(sender_online_config_json), psi.ApsiSenderConfig()
            )

            receiver_online_config = json_format.ParseDict(
                json.loads(receiver_online_config_json), psi.ApsiReceiverConfig()
            )

            link_desc = create_link_desc(2)

            def sender_wrap(rank, link_desc, config):
                link_ctx = link.create_brpc(link_desc, rank)
                psi.apsi_send(config, link_ctx)

            def receiver_wrap(rank, link_desc, config):
                link_ctx = link.create_brpc(link_desc, rank)
                psi.apsi_receive(config, link_ctx)

            jobs = [
                multiprocess.Process(
                    target=sender_wrap, args=(0, link_desc, sender_online_config)
                ),
                multiprocess.Process(
                    target=receiver_wrap, args=(1, link_desc, receiver_online_config)
                ),
            ]

            [job.start() for job in jobs]
            for job in jobs:
                job.join()
                self.assertEqual(job.exitcode, 0)

            import pandas as pd

            df1 = pd.read_csv(f'{temp_dir}/result.csv')
            df2 = pd.read_csv('spu/tests/data/ground_truth.csv')

            self.assertTrue(df1.equals(df2))


if __name__ == '__main__':
    unittest.main()
