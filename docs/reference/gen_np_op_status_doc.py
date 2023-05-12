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
from mdutils.mdutils import MdUtils


def main():
    parser = argparse.ArgumentParser(
        description='Convert Complexity JSON to MarkDown file.'
    )
    parser.add_argument("--input", type=str, help="Json file.", default="")
    parser.add_argument("--output", type=str, help="Markdown file.", default="")

    args = parser.parse_args()

    mdFile = MdUtils(file_name=args.output, title='JAX NumPy Operators Status')

    mdFile.new_header(level=1, title='Overview')

    mdFile.new_paragraph(
        "SPU recommends users to use JAX as frontend language to write logics. "
        "We found most of users would utilize **jax.numpy** modules in their programs. "
        "We have conducted tests with some *selected* Operators for reference."
    )

    mdFile.new_paragraph(
        "Just keep in mind, if you couldn't find a **jax.numpy** operator in this list, "
        "it doesn't mean it's not supported. We just haven't test it, e.g. jax.numpy.sort. "
        "And we don't test other **JAX** modules at this moment."
    )

    mdFile.new_paragraph("Please contact us if")
    mdFile.new_list(
        items=[
            "You need to confirm the status of another **jax.numpy** operator not listed here.",
            "You find a **jax.numpy** is not working even it is marked as **PASS**. e.g. The precision is bad.",
        ]
    )

    mdFile.new_paragraph()

    mdFile.new_header(level=1, title='Tested Operators List')

    with open(args.input, 'r') as f:
        status_doc = json.load(f)

    for op in status_doc:
        name = op['name']
        status = op['status']
        if status == "Status.PassNoGen":
            continue

        mdFile.new_header(level=2, title=name)
        mdFile.new_line(
            f"JAX NumPy Document link: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.{name}.html"
        )
        mdFile.new_header(level=3, title='Status')

        if status == "Status.Pass":
            mdFile.new_line("**PASS**")
            mdFile.new_line("Please check *Supported Dtypes* as well.")
        elif status == "Status.UnSupport":
            mdFile.new_line(
                "Not supported by design. We couldn't fix this in near future."
            )
            mdFile.new_line("Please check *Note* for details.")
        elif status == "Status.SysError":
            mdFile.new_line(
                "Not supported by compiler or runtime. But we could implement on demand in future."
            )
            mdFile.new_line("Please check *Note* for details.")
        elif status == "Status.Failed":
            mdFile.new_line("Result is incorrect.")
            mdFile.new_line("Please check *Note* for details.")

        if op['note']:
            mdFile.new_header(level=3, title='Note')
            mdFile.new_paragraph(op['note'])
        if status == "Status.Pass":
            mdFile.new_header(level=3, title='Supported Dtypes')
            mdFile.new_list(items=op['dtypes'])

    mdFile.create_md_file()


if __name__ == "__main__":
    main()
