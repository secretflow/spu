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
import json

from spu.tests.jnp_testbase import (
    ARGMINMAX_RECORDS,
    BITWISE_OP_RECORDS,
    COMPOUND_OP_RECORDS,
    JAX_ONE_TO_ONE_OP_RECORDS,
    REC,
    REDUCER_INITIAL_RECORDS,
    REDUCER_NO_DTYPE_RECORDS,
    REDUCER_RECORDS,
    REDUCER_WHERE_NO_INITIAL_RECORDS,
    SHIFT_RECORDS,
    OpRecord,
    Status,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate numpy operators status.')
    parser.add_argument(
        "-o",
        "--output",
        default="np_op_status.json",
        help="the output file path.",
    )

    args = parser.parse_args()

    all_records = []

    for records in [
        JAX_ONE_TO_ONE_OP_RECORDS,
        COMPOUND_OP_RECORDS,
        BITWISE_OP_RECORDS,
        REDUCER_RECORDS,
        SHIFT_RECORDS,
        ARGMINMAX_RECORDS,
    ]:
        for record in records:
            all_records.append(
                {
                    "name": record.name,
                    "dtypes": sorted([item.__name__ for item in record.dtypes]),
                    "status": str(record.status),
                    "note": record.note,
                }
            )

    all_records = sorted(all_records, key=lambda d: d['name'])

    with open(args.output, "w") as f:
        json.dump(all_records, f)
