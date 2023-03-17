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

echo "1. Update protocol complexity doc."
bash reference/update_complexity_doc.sh

echo "2. Update pphlo doc."
bash reference/update_pphlo_doc.sh

echo "3. Update runtime config doc."
# runtime_config_md.tmpl is adapted from https://github.com/pseudomuto/protoc-gen-doc/blob/master/examples/templates/grpc-md.tmpl.
docker run --rm -v $(pwd)/reference/:/out \
                -v $(pwd)/../libspu:/protos \
                pseudomuto/protoc-gen-doc \
                --doc_opt=/out/runtime_config_md.tmpl,runtime_config.md spu.proto

echo "4. Update psi config doc."
# psi_config_md.tmpl is adapted from https://github.com/pseudomuto/protoc-gen-doc/blob/master/examples/templates/grpc-md.tmpl.
docker run --rm -v $(pwd)/reference/:/out \
                -v $(pwd)/../libspu/psi:/protos \
                pseudomuto/protoc-gen-doc \
                --doc_opt=/out/psi_config_md.tmpl,psi_config.md psi.proto

echo "5. Update numpy op status doc."
bash reference/update_np_op_status_doc.sh
