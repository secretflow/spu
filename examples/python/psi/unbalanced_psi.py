# Copyright 2021 Ant Group Co., Ltd.
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
# > bazel run //examples/python/psi:unbalanced_psi -- --rank 0 --in_path examples/data/psi_1.csv --field_names id --out_path /tmp/p1.out
# > bazel run //examples/python/psi:unbalanced_psi -- --rank 1 --in_path examples/data/psi_2.csv --field_names id --out_path /tmp/p2.out

from absl import app, flags

import spu.psi as psi
import spu.libspu.link as link
import time

flags.DEFINE_integer("rank", 0, "rank: 0/1/2...")
flags.DEFINE_string("in_path", "data.csv", "data input path")
flags.DEFINE_string("field_names", "id", "csv file filed name")
flags.DEFINE_string("out_path", "data.out", "data output path")
flags.DEFINE_integer("receiver_rank", 0, "main party for psi, will get result")
flags.DEFINE_bool("output_sort", False, "whether to sort result")
flags.DEFINE_integer("bucket_size", 1048576, "hash bucket size")
FLAGS = flags.FLAGS


def setup_link(rank, port):
    lctx_desc = link.Desc()
    lctx_desc.id = f"desc_id"
    lctx_desc.recv_timeout_ms = 3600 * 1000  # 3600 seconds

    lctx_desc.add_party(f"id_0", f"127.0.0.1:{port}")
    lctx_desc.add_party(f"id_1", f"127.0.0.1:{port+10}")

    return link.create_brpc(lctx_desc, rank)


def main(_):
    selected_fields = FLAGS.field_names.split(",")

    # one-way PSI, just one party get result
    broadcast_result = False

    secret_key_path = "secret_key.bin"
    with open(secret_key_path, 'wb') as f:
        f.write(
            bytes.fromhex(
                "000102030405060708090a0b0c0d0e0ff0e0d0c0b0a090807060504030201000"
            )
        )

    link_ctx = setup_link(FLAGS.rank, 9827)
    config = psi.BucketPsiConfig(
        psi_type=psi.PsiType.Value('ECDH_OPRF_UNBALANCED_PSI_2PC_OFFLINE'),
        broadcast_result=broadcast_result,
        receiver_rank=FLAGS.receiver_rank,
        input_params=psi.InputParams(
            path=FLAGS.in_path,
            select_fields=selected_fields,
            precheck=FLAGS.precheck_input,
        ),
        output_params=psi.OutputParams(
            path=FLAGS.out_path, need_sort=FLAGS.output_sort
        ),
        bucket_size=10000000,
        curve_type=psi.CurveType.CURVE_FOURQ,
    )

    if FLAGS.receiver_rank == link_ctx.rank:
        config.preprocess_path = 'tmp/preprocess_path.csv'
        config.input_params.path = 'fake.csv'
    else:
        config.ecdh_secret_key_path = secret_key_path

    start = time.time()
    report = psi.bucket_psi(link_ctx, config)
    print(f"offline cost time: {time.time() - start}")
    print(
        f"offline: rank: {FLAGS.rank} original_count: {report.original_count}, intersection_count: {report.intersection_count}"
    )

    print("===== online phase =====")

    config_offline = psi.BucketPsiConfig(
        psi_type=psi.PsiType.Value('ECDH_OPRF_UNBALANCED_PSI_2PC_ONLINE'),
        broadcast_result=broadcast_result,
        receiver_rank=FLAGS.receiver_rank,
        input_params=psi.InputParams(
            path=FLAGS.in_path,
            select_fields=selected_fields,
            precheck=FLAGS.precheck_input,
        ),
        output_params=psi.OutputParams(
            path=FLAGS.out_path, need_sort=FLAGS.output_sort
        ),
        bucket_size=100000000,
        curve_type=psi.CurveType.CURVE_FOURQ,
    )

    print(f"input path:{FLAGS.in_path}")
    if FLAGS.receiver_rank == link_ctx.rank:
        config_offline.preprocess_path = 'tmp/preprocess_path.csv'
    else:
        config_offline.input_params.path = 'fake.csv'
        config_offline.ecdh_secret_key_path = secret_key_path

    start = time.time()
    report_online = psi.bucket_psi(link_ctx, config_offline)
    print(f"online cost time: {time.time() - start}")
    print(
        f"online: rank:{FLAGS.rank} original_count: {report_online.original_count}, intersection_count: {report_online.intersection_count}"
    )


if __name__ == '__main__':
    app.run(main)
