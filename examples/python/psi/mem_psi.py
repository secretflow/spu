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
# > bazel run //examples/python/psi:mem_psi -- --rank 0 --protocol ECDH_PSI_2PC --in_path examples/data/psi_1.csv --field_name id --out_path /tmp/p1.out
# > bazel run //examples/python/psi:mem_psi -- --rank 1 --protocol ECDH_PSI_2PC --in_path examples/data/psi_2.csv --field_name id --out_path /tmp/p2.out

from absl import app, flags

import pandas as pd
import spu.psi as psi
import spu.libspu.link as link
import spu.libspu.logging as logging

flags.DEFINE_string("protocol", "ECDH_PSI_2PC", "psi protocol, see `spu/psi/psi.proto`")
flags.DEFINE_integer("rank", 0, "rank: 0/1/2...")
flags.DEFINE_string("party_ips", "127.0.0.1:9307,127.0.0.1:9308", "party addresses")
flags.DEFINE_string("in_path", "data.csv", "data input path")
flags.DEFINE_string("field_name", "id", "csv file filed name")
flags.DEFINE_string("out_path", "mem_psi_out.csv", "data output path")
flags.DEFINE_integer("receiver_rank", 0, "main party for psi, will get result")
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

    lctx_desc.recv_timeout_ms = 2 * 60 * 1000
    lctx_desc.connect_retry_times = 180

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
    logging.setup_logging()

    # read csv
    in_df = pd.read_csv(FLAGS.in_path)
    in_data = in_df[FLAGS.field_name].astype(str).tolist()

    config = psi.MemoryPsiConfig(
        psi_type=psi.PsiType.Value(FLAGS.protocol),
        broadcast_result=False,
        receiver_rank=FLAGS.receiver_rank if FLAGS.receiver_rank >= 0 else 0,
        curve_type=psi.CurveType.CURVE_25519,
    )

    if FLAGS.protocol == "DP_PSI_2PC":
        config.dppsi_params.bob_sub_sampling = 0.9
        config.dppsi_params.epsilon = 3

    intersection = psi.mem_psi(setup_link(FLAGS.rank), config, in_data)

    out_df = pd.DataFrame(columns=[FLAGS.field_name])
    out_df[FLAGS.field_name] = intersection
    out_df.to_csv(FLAGS.out_path, index=False)

    print(f"original_count: {len(in_data)}, intersection_count: {len(intersection)}")


if __name__ == '__main__':
    app.run(main)
