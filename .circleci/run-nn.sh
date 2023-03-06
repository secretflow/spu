#!/usr/bin/bash
#
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
#

for net in network_a network_b network_c network_d; do
    for opt in sgd; do
        echo "Start training "${net}" "${opt}" "${start_ts}""
        bazel-bin/examples/python/ml/stax_nn/stax_nn --model ${net} --optimizer ${opt} -e 1 -c .circleci/benchmark.json
        echo "Finish training "${net}" "${opt}" "${end_ts}""
    done
done

# adam is slow so run network_a only
bazel-bin/examples/python/ml/stax_nn/stax_nn --model network_a --optimizer adam -e 1 -c .circleci/benchmark.json
