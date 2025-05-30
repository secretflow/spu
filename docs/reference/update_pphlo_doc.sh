#! /bin/sh
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

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd `bazelisk info workspace`/src && bazelisk build //libspu/dialect/pphlo/IR:op_doc

cp `bazelisk info workspace`/bazel-bin/libspu/dialect/pphlo/IR/op_doc.md $SCRIPTPATH/pphlo_op_doc.md

