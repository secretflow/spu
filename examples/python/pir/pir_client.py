# Copyright 2023 Ant Group Co., Ltd.
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


# To run the example, start two terminals:
# > python pir_client.py --rank 0 --in_path examples/data/pir_client_data.csv --key_columns id --out_path /tmp/pir_client_out.csv
#

from absl import app, flags

import spu.pir as pir
import spu.libspu.link as link
import spu.libspu.logging as logging

flags.DEFINE_integer("rank", 0, "rank: 0/1/2...")
flags.DEFINE_string("party_ips", "127.0.0.1:9307,127.0.0.1:9308", "party addresses")
flags.DEFINE_string("in_path", "data.csv", "data input path")
flags.DEFINE_string("key_columns", "id", "csv file filed name")
flags.DEFINE_string("out_path", "simple_psi_out.csv", "data output path")

FLAGS = flags.FLAGS


def setup_link(rank):
    lctx_desc = link.Desc()
    lctx_desc.id = f"root"

    lctx_desc.recv_timeout_ms = 2 * 60 * 1000
    # lctx_desc.connect_retry_times = 180

    ips = FLAGS.party_ips.split(",")
    for i, ip in enumerate(ips):
        lctx_desc.add_party(f"id_{i}", ip)
        print(f"id_{i} = {ip}")

    return link.create_brpc(lctx_desc, rank)


def main(_):
    opts = logging.LogOptions()
    opts.system_log_path = "./tmp/spu.log"
    opts.enable_console_logger = True
    opts.log_level = logging.LogLevel.INFO
    logging.setup_logging(opts)

    key_columns = FLAGS.key_columns.split(",")

    config = pir.PirClientConfig(
        pir_protocol=pir.PirProtocol.Value('KEYWORD_PIR_LABELED_PSI'),
        input_path=FLAGS.in_path,
        key_columns=key_columns,
        output_path=FLAGS.out_path,
    )

    report = pir.pir_client(setup_link(FLAGS.rank), config)

    print(f"data_count: {report.data_count}")


if __name__ == '__main__':
    app.run(main)
