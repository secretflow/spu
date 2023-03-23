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

set -e
set -o pipefail

pip install pwntools

BINARY=$1

CHECK_STATUS=$(pwn checksec $1 2>&1)

(echo "$CHECK_STATUS" | grep -Eq  "RELRO:\s*Full\sRELRO") && echo "relro enabled" || exit 1
(echo "$CHECK_STATUS" | grep -Eq  "Stack:\s*Canary\sfound") && echo "has canary" || exit 1
(echo "$CHECK_STATUS" | grep -Eq  "NX:\s*NX\senabled") && echo "nx enabled" || exit 1
(echo "$CHECK_STATUS" | grep -Eq  "PIE:\s*PIE\senabled") && echo "pie enabled" || exit 1
(echo "$CHECK_STATUS" | grep -Eq  "FORTIFY:\s*Enabled") && echo "FORTIFY enabled" || exit 1

