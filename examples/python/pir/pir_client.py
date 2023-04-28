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
flags.DEFINE_bool("enable_tls", False, "whether to enable tls for link")
flags.DEFINE_string("link_server_certificate", "", "link server certificate file path")
flags.DEFINE_string("link_server_private_key", "", "link server private key file path")
flags.DEFINE_string(
    "link_server_ca", "", "ca file used to verify other's link server certificate"
)
flags.DEFINE_string("link_client_certificate", "", "link client certificate file path")
flags.DEFINE_string("link_client_private_key", "", "link client private key file path")
flags.DEFINE_string(
    "link_client_ca", "", "ca file used to verify other's link client certificate"
)

FLAGS = flags.FLAGS


def setup_link(rank):
    lctx_desc = link.Desc()
    lctx_desc.id = f"root"

    lctx_desc.recv_timeout_ms = 30 * 60 * 1000
    # lctx_desc.connect_retry_times = 180

    ips = FLAGS.party_ips.split(",")
    for i, ip in enumerate(ips):
        lctx_desc.add_party(f"id_{i}", ip)
        print(f"id_{i} = {ip}")

    # config link tls
    if FLAGS.enable_tls:
        # two-way authentication
        lctx_desc.server_ssl_opts.cert.certificate_path = FLAGS.link_server_certificate
        lctx_desc.server_ssl_opts.cert.private_key_path = FLAGS.link_server_private_key
        lctx_desc.server_ssl_opts.verify.ca_file_path = FLAGS.link_server_ca
        lctx_desc.server_ssl_opts.verify.verify_depth = 1
        lctx_desc.client_ssl_opts.cert.certificate_path = FLAGS.link_client_certificate
        lctx_desc.client_ssl_opts.cert.private_key_path = FLAGS.link_client_private_key
        lctx_desc.client_ssl_opts.verify.ca_file_path = FLAGS.link_client_ca
        lctx_desc.client_ssl_opts.verify.verify_depth = 1

    return link.create_brpc(lctx_desc, rank)


def main(_):
    opts = logging.LogOptions()
    opts.system_log_path = "./tmp/spu.log"
    opts.trace_log_path = "./tmp/trace.log"
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

    link_ctx = setup_link(FLAGS.rank)
    report = pir.pir_client(link_ctx, config)

    print(f"data_count: {report.data_count}")


if __name__ == '__main__':
    app.run(main)
