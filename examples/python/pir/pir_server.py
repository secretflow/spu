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
# > python pir_server.py --rank 1 --setup_path setup_path --oprf_key_path oprf_key.bin
#

from absl import app, flags

import spu.pir as pir
import spu.libspu.link as link
import spu.libspu.logging as logging

flags.DEFINE_integer("rank", 0, "rank: 0/1/2...")
flags.DEFINE_string("party_ips", "127.0.0.1:9307,127.0.0.1:9308", "party addresses")
flags.DEFINE_string("in_path", "data.csv", "data input path")
flags.DEFINE_string("oprf_key_path", "oprf_key.bin", "oprf key file path")
flags.DEFINE_string("setup_path", "setup_path", "data output path")
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
    opts.enable_console_logger = True
    opts.log_level = logging.LogLevel.INFO
    logging.setup_logging(opts)

    config = pir.PirServerConfig(
        pir_protocol=pir.PirProtocol.Value('KEYWORD_PIR_LABELED_PSI'),
        store_type=pir.KvStoreType.Value('LEVELDB_KV_STORE'),
        oprf_key_path=FLAGS.oprf_key_path,
        setup_path=FLAGS.setup_path,
    )

    link_ctx = setup_link(FLAGS.rank)
    report = pir.pir_server(link_ctx, config)

    print(f"data_count: {report.data_count}")


if __name__ == '__main__':
    app.run(main)
