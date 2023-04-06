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
# > python pir_setup.py --in_path examples/data/pir_server_data.csv --key_columns id --label_columns label \
# > --count_per_query 1 -max_label_length 256 \
# > --oprf_key_path oprf_key.bin --setup_path setup_path

from absl import app, flags

import spu.pir as pir
import spu.libspu.link as link
import spu.libspu.logging as logging

flags.DEFINE_string("in_path", "data.csv", "data input path")
flags.DEFINE_string("key_columns", "id", "csv file key filed name")
flags.DEFINE_string("label_columns", "label", "csv file label filed name")
flags.DEFINE_integer("count_per_query", 1, "count_per_query")
flags.DEFINE_integer("max_label_length", 256, "max_label_length")
flags.DEFINE_string("oprf_key_path", "oprf_key.bin", "oprf key file")
flags.DEFINE_string("setup_path", "setup_path", "data output path")
FLAGS = flags.FLAGS


def main(_):
    opts = logging.LogOptions()
    opts.system_log_path = "./tmp/spu.log"
    opts.enable_console_logger = True
    opts.log_level = logging.LogLevel.INFO
    logging.setup_logging(opts)

    key_columns = FLAGS.key_columns.split(",")
    label_columns = FLAGS.label_columns.split(",")

    config = pir.PirSetupConfig(
        pir_protocol=pir.PirProtocol.Value('KEYWORD_PIR_LABELED_PSI'),
        store_type=pir.KvStoreType.Value('LEVELDB_KV_STORE'),
        input_path=FLAGS.in_path,
        key_columns=key_columns,
        label_columns=label_columns,
        num_per_query=FLAGS.count_per_query,
        label_max_len=FLAGS.max_label_length,
        oprf_key_path=FLAGS.oprf_key_path,
        setup_path=FLAGS.setup_path,
    )

    report = pir.pir_setup(config)
    print(f"data_count: {report.data_count}")


if __name__ == '__main__':
    app.run(main)
