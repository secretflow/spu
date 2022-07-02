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
# > bazel run //examples/python/psi:simple_psi -- --rank 0 --protocol ecdh --in_path examples/data/psi_1.csv --field_names id --out_path /tmp/p1.out
# > bazel run //examples/python/psi:simple_psi -- --rank 1 --protocol ecdh --in_path examples/data/psi_2.csv --field_names id --out_path /tmp/p2.out

from absl import app, flags
import re
import subprocess

import spu.binding._lib.libs as libs
import spu.binding._lib.link as link


def wc_count(file_path):
    assert re.match(r'([a-zA-Z0-9\s_\\.\-\(\)\\/])+', file_path)
    out = subprocess.getoutput("wc -l %s" % file_path)
    return int(out.split()[0])


class PsiTests:
    def run_streaming_psi(self, fn, rank, in_path, out_path, selected_fields):
        lctx_desc = link.Desc()
        lctx_desc.id = f"desc_id"
        port = 9727

        lctx_desc.add_party(f"id_0", f"127.0.0.1:{port}")
        lctx_desc.add_party(f"id_1", f"127.0.0.1:{port+10}")

        lctx = link.create_brpc(lctx_desc, rank)

        fn(lctx, selected_fields, in_path, out_path)

    def test_kkrt_2pc(self, rank, in_path, out_path, field_names, broadcast_result):
        print("----------test_kkrt_2pc-------------")

        selected_fields = field_names.split(",")

        def fn(lctx, selected_fields, input_path, output_path):
            report = libs.PsiReport()
            libs.kkrt_2pc_psi(
                lctx, selected_fields, input_path, output_path, True, report, broadcast_result
            )
            source_count = wc_count(input_path)
            output_count = wc_count(output_path)
            id = lctx.id()
            print(
                f"test_kkrt_2pc---id:{id}, original_count: {report.original_count}, intersection_count: {report.intersection_count}, source_count: {source_count}, output_count: {output_count}"
            )

        self.run_streaming_psi(fn, rank, in_path, out_path, selected_fields)

    def test_ecdh_2pc(self, rank, in_path, out_path, field_names):
        print("----------test_ecdh_2pc-------------")

        selected_fields = field_names.split(",")

        def fn(lctx, selected_fields, input_path, output_path):
            report = libs.PsiReport()
            libs.ecdh_2pc_psi(
                lctx, selected_fields, input_path, output_path, 64, True, report
            )
            source_count = wc_count(input_path)
            output_count = wc_count(output_path)
            id = lctx.id()
            print(
                f"test_ecdh_2pc---id:{id}, original_count: {report.original_count}, intersection_count: {report.intersection_count}, source_count: {source_count}, output_count: {output_count}"
            )

        self.run_streaming_psi(fn, rank, in_path, out_path, selected_fields)


flags.DEFINE_string("protocol", "ecdh", "psi protocol: ecdh/kkrt")
flags.DEFINE_integer("rank", 0, "rank: 0/1")
flags.DEFINE_string("in_path", "data.csv", "data input path")
flags.DEFINE_string("field_names", "id", "csv file filed name")
flags.DEFINE_string("out_path", "data.out", "data output path")
FLAGS = flags.FLAGS


def main(_):
    psi_test = PsiTests()
    if FLAGS.protocol == 'ecdh':
        PsiTests.test_ecdh_2pc(
            psi_test, FLAGS.rank, FLAGS.in_path, FLAGS.out_path, FLAGS.field_names
        )
    elif FLAGS.protocol == 'kkrt':
        PsiTests.test_kkrt_2pc(
            psi_test, FLAGS.rank, FLAGS.in_path, FLAGS.out_path, FLAGS.field_names, False
        )
    else:
        print(f"unsupport psi protocol: {FLAGS.protocol}")


if __name__ == '__main__':
    app.run(main)
