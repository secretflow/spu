#! /usr/bin/python3

# Copyright 2022 Ant Group Co., Ltd.
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


import argparse
from pytablewriter import MarkdownTableWriter
import json


def main():
    parser = argparse.ArgumentParser(
        description='Convert Complexity JSON to MarkDown file.'
    )
    parser.add_argument("--input", type=str, help="Json file.", default="")
    parser.add_argument("--output", type=str, help="Markdown file.", default="")

    args = parser.parse_args()

    with open(args.input, 'r') as f:
        complexity = json.load(f)

    writer = MarkdownTableWriter()
    writer.headers = ['kernel', 'latency', 'comm']

    res = ''

    for report in complexity['reports']:
        writer.table_name = report['protocol']
        value_matrix = []
        for entry in report['entries']:
            value_matrix.append(
                [
                    entry['kernel'],
                    entry['latency'].replace('*', '\*'),
                    entry['comm'].replace('*', '\*'),
                ]
            )
        writer.value_matrix = value_matrix
        writer.write_table()
        writer.write_null_line()
        res += writer.dumps()

    with open(args.output, 'w') as f:
        f.write(res)


if __name__ == "__main__":
    main()
