#!/bin/bash
# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cd "$(dirname "$(readlink -f "$0")")"

# set -x
set -e

CURRENT_DIR=$(pwd)

pip install grpcio==1.56.2 grpcio-tools==1.56.2

cd ../../../
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. examples/python/utils/distributed.proto