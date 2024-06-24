#! /usr/bin/env python3

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

import collections
import json

Data = collections.namedtuple(
    "Data",
    ["data", "shape", "dtype"],
)

Case = collections.namedtuple(
    "TestCase",
    ["inputs", "expected", "checker", "tol"],
)

Record = collections.namedtuple(
    "Record",
    ["name", "template", "cases"],
)


def TestCase(inputs, expected, checker='expect_eq', tol=None):
    return Case(inputs, expected, checker, tol)


TESTS = [
    "abs",
    "add",
    "and",
    "atan2",
    "ceil",
    "cosine",
    "divide",
    "equal",
    "not_equal",
    "greater_equal",
    "greater",
    "less",
    "less_equal",
    "exponential_minus_one",
    "exponential",
    "floor",
    "log_plus_one",
    "log",
    "logistic",
    "max",
    "min",
    "multiply",
    "negate",
    "not",
    "or",
    # "popcnt",
    "power",
    "reshape",
    "round_afz",
    "rsqrt",
    "arshift",
    "rshift",
    "sign",
    "sine",
    "sqrt",
    "subtract",
    "tanh",
    "xor",
]

for test in TESTS:
    with open(f"test_json/{test}.json", "r") as f:
        test_contents = json.loads(f.read())

    test_name = test_contents["name"]
    template_name = test_contents["template"]

    with open(f"template/{template_name}.template", "r") as f:
        template = f.read()

    with open(f"{test_name}.mlir", "w+") as f:
        # emit run command
        f.write("// RUN: spu-translate --interpret -split-input-file %s\n")
        f.write("// AUTO GENERATED, DO NOT EDIT\n\n")

        # Emit cases
        cases = []
        for case in test_contents["testcases"]:
            c = template.replace('%OP%', test_name)
            for idx, input in enumerate(case["inputs"]):
                c = c.replace(f'%INPUT{idx}%', input["data"])
                if not input["shape"]:
                    c = c.replace(f'%IN{idx}_SHAPE%x', '')
                else:
                    c = c.replace(f'%IN{idx}_SHAPE%', input["shape"])
                c = c.replace(f'%IN{idx}_DTYPE%', input["dtype"])
            for idx, expect in enumerate(case["expected"]):
                c = c.replace(f'%EXPECTED{idx}%', expect["data"])
                if not expect["shape"]:
                    c = c.replace(f'%OUT{idx}_SHAPE%x', '')
                else:
                    c = c.replace(f'%OUT{idx}_SHAPE%', expect["shape"])
                c = c.replace(f'%OUT{idx}_DTYPE%', expect["dtype"])
            if "checker" in case:
                c = c.replace('%CHECKER%', case["checker"])
            else:
                c = c.replace('%CHECKER%', 'expect_eq')
            if "tol" in case:
                c = c.replace('%ATTR%', f'{{ tol = {case["tol"]} }}')
            else:
                c = c.replace('%ATTR%', '')
            cases.append(c)

        f.write("\n// -----\n\n".join(cases))
