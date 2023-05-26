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
REPOPATH=`realpath $SCRIPTPATH/../../`

bazel run //spu/tests:np_op_status -- --out="$REPOPATH/docs/reference/np_op_status.json"
python $REPOPATH/docs/reference/gen_np_op_status_doc.py --in="$REPOPATH/docs/reference/np_op_status.json" --out="$REPOPATH/docs/reference/np_op_status.md"

