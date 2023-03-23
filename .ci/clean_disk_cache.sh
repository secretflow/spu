#!/usr/bin/env bash
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

set -eu

: ${BAZEL_DISK_CACHE_PATH:=~/.cache/spu_build_cache}

# As a courtesy, compute and print some approximate stats.
total_file_count=$(find "$BAZEL_DISK_CACHE_PATH" -type f | wc -l)
stale_file_count=$(find "$BAZEL_DISK_CACHE_PATH" -type f -atime +5 | wc -l)
echo "Removing $stale_file_count files out of $total_file_count total."

# Just re-running the find is simpler than managing any state.
#find "$BAZEL_DISK_CACHE_PATH" -type f -atime +30 -delete
