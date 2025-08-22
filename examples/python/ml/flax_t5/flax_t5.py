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

# Start nodes.
# > python examples/python/utils/nodectl.py --config `pwd`/examples/python/ml/flax_t5/3pc.json up
#
# Run this example script.
# > python examples/python/ml/flax_t5/flax_t5.py

import argparse
import json

import examples.python.utils.distributed as ppd

DEFAULT_CONF_FILE = "examples/python/ml/flax_t5/3pc.json"

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("t5-small")
pretrained_model = FlaxT5ForConditionalGeneration.from_pretrained("t5-small")

ARTICLE_TO_SUMMARIZE = "summarize: My friends are cool but they eat too many carbs."
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors="np")


def summary_generation(input_ids, params):
    pretrained_model.params = params
    summary_ids = pretrained_model.generate(input_ids, max_new_tokens=20)
    return summary_ids.sequences


def run_on_cpu():
    summary_ids = pretrained_model.generate(inputs["input_ids"], max_new_tokens=20)
    return summary_ids.sequences


def run_on_spu(config=DEFAULT_CONF_FILE):
    with open(config, 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"])

    input_ids = ppd.device("P1")(lambda x: x)(inputs["input_ids"])
    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
    summary_ids = ppd.device("SPU")(
        summary_generation,
    )(input_ids, params)
    summary_ids = ppd.get(summary_ids)
    return summary_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument("-c", "--config", default=DEFAULT_CONF_FILE)
    args = parser.parse_args()

    print('\n------\nRun on CPU')
    summary_ids = run_on_cpu()
    print(
        tokenizer.decode(
            summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    )
    print('\n------\nRun on SPU')
    summary_ids = run_on_spu(args.config)
    print(
        tokenizer.decode(
            summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    )
