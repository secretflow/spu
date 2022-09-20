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


if test -n "${BAZEL_MAX_JOBS-}"; then
  parallel_flag="--jobs=${BAZEL_MAX_JOBS}"
else
  parallel_flag=""
fi

echo ${parallel_flag}

bazel build --ui_event_filters=-info,-debug,-warning //spu:spu_wheel -c opt ${parallel_flag}
spu_wheel_name=$(<bazel-bin/spu/spu_wheel.name)
spu_wheel_path="bazel-bin/spu/${spu_wheel_name//sf-spu/sf_spu}"

cp -rf $spu_wheel_path ./
cp -rf bazel-bin/spu/spu_wheel.name ./
