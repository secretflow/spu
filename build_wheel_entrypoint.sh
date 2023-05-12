#! /bin/bash
#
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
#

pip install numpy

python setup.py bdist_wheel

# Ensure binary safety
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    UTIL_DIR="devtools"
    git clone https://github.com/secretflow/devtools.git
    sh $UTIL_DIR/check-binary.sh bazel-bin/spu/libspu.so
    rm -rf $UTIL_DIR
fi
